"""
ResizeBatch (Loop507)
----------------------
Streamlit app per il ridimensionamento batch di immagini a una risoluzione
target, con quattro modalità di adattamento, rinomina batch e correzione
colore opzionale.

Modalità resize:
  - stretch      : resize diretto (deforma se i rapporti differiscono)
  - crop         : center-crop al rapporto target, poi resize (nessuna
                   distorsione, ma parte dell'immagine viene tagliata)
  - pad          : letterbox/pillarbox con colore a scelta
  - pad_glitch   : bande procedurali stile Loop507 / Glitch Brutalista

Correzione colore (opzionale, applicata prima del resize):
  - Auto White Balance (gray-world)
  - Auto Livelli / contrasto (stretch dell'istogramma su nero/bianco)
  - Luminosità, Contrasto, Saturazione (fattori manuali)
  - Curva a S (aumenta il contrasto tonale in modo morbido)

Dipendenze: streamlit, pillow, numpy
"""

import io
import zipfile
from dataclasses import dataclass

import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageOps

# ---------------------------------------------------------------------------
# Config pagina
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ResizeBatch :: Loop507", page_icon="▦", layout="wide")

st.title("▦ ResizeBatch")
st.caption(
    "Carica più foto, scegli la risoluzione target, la modalità di adattamento e, "
    "se vuoi, una correzione colore. Upload multiple photos, pick a target "
    "resolution, a fit mode, and an optional color correction."
)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "processed" not in st.session_state:
    st.session_state.processed = []  # list of (filename, bytes, orig_img, out_img)
if "zip_bytes" not in st.session_state:
    st.session_state.zip_bytes = None


@dataclass
class Settings:
    target_w: int
    target_h: int
    mode: str
    pad_color: tuple
    output_format: str
    jpeg_quality: int
    rename_base: str


@dataclass
class ColorSettings:
    enabled: bool
    auto_white_balance: bool
    auto_levels: bool
    auto_levels_cutoff: float
    brightness: float
    contrast: float
    saturation: float
    s_curve_strength: float


# ---------------------------------------------------------------------------
# Sidebar :: parametri resize
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header(":: parametri / settings")

    col_w, col_h = st.columns(2)
    with col_w:
        target_w = st.number_input("Larghezza target (px)", min_value=16, max_value=8000, value=1280, step=1)
    with col_h:
        target_h = st.number_input("Altezza target (px)", min_value=16, max_value=8000, value=720, step=1)

    mode_labels = {
        "stretch": "Stretch — resize diretto (può deformare)",
        "crop": "Crop to fill — center-crop, nessuna deformazione",
        "pad": "Pad / Letterbox — bande di colore",
        "pad_glitch": "Pad Glitch — bande procedurali Loop507",
    }
    mode = st.radio(
        "Modalità di adattamento",
        options=list(mode_labels.keys()),
        format_func=lambda k: mode_labels[k],
        index=1,
    )

    pad_color = (0, 0, 0)
    if mode == "pad":
        pad_color_hex = st.color_picker("Colore banda (pad)", value="#000000")
        pad_color = tuple(int(pad_color_hex.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))

    st.divider()
    output_format = st.selectbox("Formato output", options=["JPEG", "PNG"], index=0)
    jpeg_quality = 95
    if output_format == "JPEG":
        jpeg_quality = st.slider("Qualità JPEG", min_value=50, max_value=100, value=95)

    st.divider()
    st.subheader(":: rinomina batch")
    rename_base = st.text_input(
        "Nome base (vuoto = mantieni nome originale)",
        value="",
        placeholder="es. lavoro",
        help="Con 'lavoro' otterrai lavoro_1, lavoro_2, lavoro_3... "
             "Puoi usare {n} per posizionare il numero manualmente, es. 'foto_{n}_finale'.",
    )

    st.divider()
    st.subheader(":: correzione colore")
    cc_enabled = st.checkbox("Applica correzione colore", value=False)
    cc_awb = cc_auto_levels = False
    cc_cutoff = 1.0
    cc_brightness = cc_contrast = cc_saturation = 1.0
    cc_scurve = 0.0
    if cc_enabled:
        cc_awb = st.checkbox("Auto White Balance (bilanciamento del bianco)", value=False)
        cc_auto_levels = st.checkbox("Auto livelli (stretch nero/bianco)", value=False)
        if cc_auto_levels:
            cc_cutoff = st.slider("Cutoff auto livelli (%)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
        cc_brightness = st.slider("Luminosità", min_value=0.5, max_value=1.5, value=1.0, step=0.05)
        cc_contrast = st.slider("Contrasto", min_value=0.5, max_value=1.5, value=1.0, step=0.05)
        cc_saturation = st.slider("Saturazione", min_value=0.0, max_value=2.0, value=1.0, step=0.05)
        cc_scurve = st.slider("Curva a S (contrasto tonale)", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

    settings = Settings(
        target_w=int(target_w),
        target_h=int(target_h),
        mode=mode,
        pad_color=pad_color,
        output_format=output_format,
        jpeg_quality=jpeg_quality,
        rename_base=rename_base.strip(),
    )

    color_settings = ColorSettings(
        enabled=cc_enabled,
        auto_white_balance=cc_awb,
        auto_levels=cc_auto_levels,
        auto_levels_cutoff=cc_cutoff,
        brightness=cc_brightness,
        contrast=cc_contrast,
        saturation=cc_saturation,
        s_curve_strength=cc_scurve,
    )

    st.divider()
    st.caption(f":: target ratio = {settings.target_w / settings.target_h:.3f}")

    st.divider()
    st.subheader(":: anteprima")
    preview_limit = st.slider(
        "Numero di anteprime da mostrare",
        min_value=1,
        max_value=20,
        value=5,
        help="Con batch grandi mostrare tutte le anteprime rallenta l'app. "
             "Il download (singolo o ZIP) include comunque tutte le immagini.",
    )

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Carica le immagini (JPG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)


# ---------------------------------------------------------------------------
# Funzioni di correzione colore
# ---------------------------------------------------------------------------
def auto_white_balance(img: Image.Image) -> Image.Image:
    """Bilanciamento del bianco gray-world: scala i canali R/G/B in modo
    che le loro medie coincidano, assumendo che la media dell'immagine
    dovrebbe essere neutra (grigia)."""
    arr = np.asarray(img).astype(np.float32)
    means = arr.reshape(-1, 3).mean(axis=0)
    gray_mean = means.mean()
    # evita divisioni per zero su immagini quasi nere
    gains = np.where(means > 1e-3, gray_mean / means, 1.0)
    arr = arr * gains
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def auto_levels(img: Image.Image, cutoff_pct: float) -> Image.Image:
    """Stretch dell'istogramma: porta il punto nero e il punto bianco
    ai limiti 0/255, ignorando una piccola percentuale di outlier."""
    return ImageOps.autocontrast(img, cutoff=cutoff_pct)


def apply_s_curve(img: Image.Image, strength: float) -> Image.Image:
    """Applica una curva a S per aumentare il contrasto tonale in modo
    morbido: scurisce le ombre e schiarisce le luci senza clipping duro."""
    if strength <= 0:
        return img
    x = np.linspace(0, 1, 256)
    # curva a S basata su funzione smoothstep, miscelata con l'identità
    s = x * x * (3 - 2 * x)
    lut_norm = x * (1 - strength) + s * strength
    lut = np.clip(lut_norm * 255, 0, 255).astype(np.uint8)
    lut_full = list(lut) * 3  # stessa curva su R, G, B
    return img.point(lut_full)


def apply_color_correction(img: Image.Image, cs: ColorSettings) -> Image.Image:
    if not cs.enabled:
        return img
    out = img
    if cs.auto_white_balance:
        out = auto_white_balance(out)
    if cs.auto_levels:
        out = auto_levels(out, cs.auto_levels_cutoff)
    if cs.brightness != 1.0:
        out = ImageEnhance.Brightness(out).enhance(cs.brightness)
    if cs.contrast != 1.0:
        out = ImageEnhance.Contrast(out).enhance(cs.contrast)
    if cs.saturation != 1.0:
        out = ImageEnhance.Color(out).enhance(cs.saturation)
    if cs.s_curve_strength > 0:
        out = apply_s_curve(out, cs.s_curve_strength)
    return out


# ---------------------------------------------------------------------------
# Funzioni di trasformazione (resize)
# ---------------------------------------------------------------------------
def resize_stretch(img: Image.Image, tw: int, th: int) -> Image.Image:
    return img.resize((tw, th), Image.LANCZOS)


def resize_crop(img: Image.Image, tw: int, th: int) -> Image.Image:
    src_w, src_h = img.size
    target_ratio = tw / th
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        new_w = int(src_h * target_ratio)
        new_h = src_h
        x0 = (src_w - new_w) // 2
        y0 = 0
    else:
        new_w = src_w
        new_h = int(src_w / target_ratio)
        x0 = 0
        y0 = (src_h - new_h) // 2

    cropped = img.crop((x0, y0, x0 + new_w, y0 + new_h))
    return cropped.resize((tw, th), Image.LANCZOS)


def _fit_inside(img: Image.Image, tw: int, th: int) -> tuple:
    src_w, src_h = img.size
    scale = min(tw / src_w, th / src_h)
    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))
    scaled = img.resize((new_w, new_h), Image.LANCZOS)
    off_x = (tw - new_w) // 2
    off_y = (th - new_h) // 2
    return scaled, off_x, off_y


def resize_pad(img: Image.Image, tw: int, th: int, color: tuple) -> Image.Image:
    scaled, off_x, off_y = _fit_inside(img, tw, th)
    canvas = Image.new("RGB", (tw, th), color)
    canvas.paste(scaled, (off_x, off_y))
    return canvas


def _make_glitch_pattern(w: int, h: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = rng.integers(10, 40, size=(h, w, 1), dtype=np.uint8)
    arr = np.repeat(base, 3, axis=2).astype(np.int16)

    arr[::3, :, :] += 12

    n_bands = max(1, w // 40)
    for _ in range(n_bands):
        bx = rng.integers(0, max(1, w - 6))
        bw = rng.integers(1, 6)
        channel = rng.integers(0, 3)
        arr[:, bx:bx + bw, channel] += rng.integers(20, 60)

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def resize_pad_glitch(img: Image.Image, tw: int, th: int, seed: int) -> Image.Image:
    scaled, off_x, off_y = _fit_inside(img, tw, th)
    canvas = _make_glitch_pattern(tw, th, seed)
    canvas.paste(scaled, (off_x, off_y))
    return canvas


def resize_only(img: Image.Image, s: Settings, seed: int) -> Image.Image:
    if s.mode == "stretch":
        return resize_stretch(img, s.target_w, s.target_h)
    elif s.mode == "crop":
        return resize_crop(img, s.target_w, s.target_h)
    elif s.mode == "pad":
        return resize_pad(img, s.target_w, s.target_h, s.pad_color)
    elif s.mode == "pad_glitch":
        return resize_pad_glitch(img, s.target_w, s.target_h, seed)
    else:
        raise ValueError(f"Modalità sconosciuta: {s.mode}")


def process_image(img: Image.Image, s: Settings, cs: ColorSettings, seed: int) -> Image.Image:
    img = img.convert("RGB")
    img = apply_color_correction(img, cs)
    return resize_only(img, s, seed)


def image_to_bytes(img: Image.Image, fmt: str, quality: int) -> bytes:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=quality)
    else:
        img.save(buf, format="PNG")
    return buf.getvalue()


def build_output_name(original_name: str, rename_base: str, index: int, tw: int, th: int, ext: str) -> str:
    """Genera il nome del file di output. Se rename_base è vuoto, mantiene
    il nome originale con suffisso dimensione. Altrimenti applica il
    pattern di rinomina (supporta {n} come placeholder esplicito)."""
    if not rename_base:
        base = original_name.rsplit(".", 1)[0]
        return f"{base}_{tw}x{th}.{ext}"
    if "{n}" in rename_base:
        base = rename_base.replace("{n}", str(index))
    else:
        base = f"{rename_base}_{index}"
    return f"{base}_{tw}x{th}.{ext}"


# ---------------------------------------------------------------------------
# Anteprima live (solo prima foto, si aggiorna ad ogni modifica parametri)
# ---------------------------------------------------------------------------
if uploaded_files:
    st.divider()
    st.subheader(":: anteprima live / live preview")
    st.caption(
        "Si aggiorna in tempo reale sulla prima foto caricata mentre muovi gli slider "
        "nella sidebar. Il batch completo viene elaborato solo quando premi 'Elabora tutte'."
    )

    ref_file = uploaded_files[0]
    ref_img = Image.open(ref_file)
    live_out = process_image(ref_img, settings, color_settings, seed=0)

    lc1, lc2 = st.columns(2)
    with lc1:
        st.image(ref_img, caption=f"originale — {ref_img.size[0]}x{ref_img.size[1]} — {ref_file.name}", use_container_width=True)
    with lc2:
        st.image(live_out, caption=f"anteprima — {live_out.size[0]}x{live_out.size[1]}", use_container_width=True)

# ---------------------------------------------------------------------------
# Elaborazione
# ---------------------------------------------------------------------------
if uploaded_files:
    total_weight_mb = sum(uf.size for uf in uploaded_files) / (1024 * 1024)
    st.write(
        f":: {len(uploaded_files)} file caricati (~{total_weight_mb:.1f} MB totali) — "
        f"target {settings.target_w}x{settings.target_h} — modalità: {mode_labels[settings.mode]}"
    )
    if total_weight_mb > 300:
        st.warning(
            f"Il batch pesa ~{total_weight_mb:.0f} MB. Il piano gratuito di Streamlit Cloud ha "
            "circa 1GB di RAM condivisa tra upload e immagini elaborate in memoria: sopra i "
            "300-400MB totali rischi l'errore 'resource limits'. Se succede, prova a dividere "
            "il batch in gruppi più piccoli — non conta il numero di foto in sé, ma il loro peso."
        )

    if st.button("▶ Elabora tutte", type="primary"):
        st.session_state.processed = []
        progress = st.progress(0.0)
        ext = "jpg" if settings.output_format == "JPEG" else "png"

        for i, uf in enumerate(uploaded_files):
            img = Image.open(uf)
            out_img = process_image(img, settings, color_settings, seed=i)
            out_bytes = image_to_bytes(out_img, settings.output_format, settings.jpeg_quality)
            out_name = build_output_name(uf.name, settings.rename_base, i + 1, settings.target_w, settings.target_h, ext)
            st.session_state.processed.append((out_name, out_bytes, img, out_img))
            progress.progress((i + 1) / len(uploaded_files))

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, data, _, _ in st.session_state.processed:
                zf.writestr(name, data)
        st.session_state.zip_bytes = zip_buf.getvalue()

    if st.session_state.processed:
        st.divider()
        total = len(st.session_state.processed)
        shown = min(preview_limit, total)
        st.subheader(f":: anteprima / preview ({shown} di {total})")

        for name, data, orig_img, out_img in st.session_state.processed[:preview_limit]:
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                st.image(orig_img, caption=f"originale {orig_img.size[0]}x{orig_img.size[1]}", use_container_width=True)
            with c2:
                st.image(out_img, caption=f"output {out_img.size[0]}x{out_img.size[1]}", use_container_width=True)
            with c3:
                st.write(name)
                st.download_button(
                    "⬇ scarica singola",
                    data=data,
                    file_name=name,
                    mime=f"image/{'jpeg' if settings.output_format == 'JPEG' else 'png'}",
                    key=f"dl_{name}",
                )
            st.divider()

        if total > preview_limit:
            st.caption(
                f":: altre {total - preview_limit} immagini elaborate ma non mostrate in anteprima "
                "— sono comunque incluse nello ZIP qui sotto."
            )

        if st.session_state.zip_bytes:
            st.download_button(
                "⬇ scarica tutte (ZIP)",
                data=st.session_state.zip_bytes,
                file_name=f"resizebatch_{settings.target_w}x{settings.target_h}.zip",
                mime="application/zip",
                type="primary",
            )
else:
    st.info("Carica una o più immagini per iniziare.")
