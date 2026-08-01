"""
Loop507 :: Resize Batch
------------------------
Streamlit app per il ridimensionamento batch di immagini a una risoluzione
target, con quattro modalità di adattamento quando il rapporto d'aspetto
sorgente != rapporto d'aspetto target.

Modalità:
  - stretch      : resize diretto (deforma se i rapporti differiscono)
  - crop         : center-crop al rapporto target, poi resize (nessuna
                   distorsione, ma parte dell'immagine viene tagliata)
  - pad          : letterbox/pillarbox con colore a scelta (immagine intera
                   preservata, bande aggiunte)
  - pad_glitch   : come pad, ma le bande sono riempite con un pattern
                   procedurale (scanline + rumore + bande di colore),
                   coerente con l'estetica Loop507 / Glitch Brutalista

Dipendenze: streamlit, pillow, numpy (leggere, compatibili Streamlit Cloud)
"""

import io
import zipfile
from dataclasses import dataclass

import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Config pagina
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Loop507 :: Resize Batch", page_icon="▦", layout="wide")

st.title("▦ Loop507 :: Resize Batch")
st.caption(
    "Carica più foto, scegli la risoluzione target e la modalità di adattamento. "
    "Upload multiple photos, pick a target resolution and a fit mode."
)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "processed" not in st.session_state:
    st.session_state.processed = []  # list of (filename, bytes)
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


# ---------------------------------------------------------------------------
# Sidebar :: parametri
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

    settings = Settings(
        target_w=int(target_w),
        target_h=int(target_h),
        mode=mode,
        pad_color=pad_color,
        output_format=output_format,
        jpeg_quality=jpeg_quality,
    )

    st.divider()
    st.caption(f":: target ratio = {settings.target_w / settings.target_h:.3f}")

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Carica le immagini (JPG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)


# ---------------------------------------------------------------------------
# Funzioni di trasformazione
# ---------------------------------------------------------------------------
def resize_stretch(img: Image.Image, tw: int, th: int) -> Image.Image:
    return img.resize((tw, th), Image.LANCZOS)


def resize_crop(img: Image.Image, tw: int, th: int) -> Image.Image:
    src_w, src_h = img.size
    target_ratio = tw / th
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        # sorgente più larga del target -> crop orizzontale (larghezza)
        new_w = int(src_h * target_ratio)
        new_h = src_h
        x0 = (src_w - new_w) // 2
        y0 = 0
    else:
        # sorgente più stretta/alta del target -> crop verticale (altezza)
        new_w = src_w
        new_h = int(src_w / target_ratio)
        x0 = 0
        y0 = (src_h - new_h) // 2

    cropped = img.crop((x0, y0, x0 + new_w, y0 + new_h))
    return cropped.resize((tw, th), Image.LANCZOS)


def _fit_inside(img: Image.Image, tw: int, th: int) -> tuple:
    """Scala l'immagine mantenendo le proporzioni, in modo che entri
    interamente dentro (tw, th). Ritorna (immagine_scalata, offset_x, offset_y)."""
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
    """Genera un pattern procedurale (scanline + bande + rumore) da usare
    come riempimento delle bande di padding, in stile Loop507."""
    rng = np.random.default_rng(seed)
    base = rng.integers(10, 40, size=(h, w, 1), dtype=np.uint8)
    arr = np.repeat(base, 3, axis=2).astype(np.int16)

    # scanline orizzontali
    arr[::3, :, :] += 12

    # bande colorate verticali casuali (accenti glitch)
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


def process_image(img: Image.Image, s: Settings, seed: int) -> Image.Image:
    img = img.convert("RGB")
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


def image_to_bytes(img: Image.Image, fmt: str, quality: int) -> bytes:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=quality)
    else:
        img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Elaborazione
# ---------------------------------------------------------------------------
if uploaded_files:
    st.write(f":: {len(uploaded_files)} file caricati — target {settings.target_w}x{settings.target_h} — modalità: {mode_labels[settings.mode]}")

    if st.button("▶ Elabora tutte", type="primary"):
        st.session_state.processed = []
        progress = st.progress(0.0)
        ext = "jpg" if settings.output_format == "JPEG" else "png"

        for i, uf in enumerate(uploaded_files):
            img = Image.open(uf)
            out_img = process_image(img, settings, seed=i)
            out_bytes = image_to_bytes(out_img, settings.output_format, settings.jpeg_quality)
            base_name = uf.name.rsplit(".", 1)[0]
            out_name = f"{base_name}_{settings.target_w}x{settings.target_h}.{ext}"
            st.session_state.processed.append((out_name, out_bytes, img, out_img))
            progress.progress((i + 1) / len(uploaded_files))

        # crea zip in memoria
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, data, _, _ in st.session_state.processed:
                zf.writestr(name, data)
        st.session_state.zip_bytes = zip_buf.getvalue()

    if st.session_state.processed:
        st.divider()
        st.subheader(":: anteprima / preview")

        for name, data, orig_img, out_img in st.session_state.processed:
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

        if st.session_state.zip_bytes:
            st.download_button(
                "⬇ scarica tutte (ZIP)",
                data=st.session_state.zip_bytes,
                file_name=f"loop507_resize_{settings.target_w}x{settings.target_h}.zip",
                mime="application/zip",
                type="primary",
            )
else:
    st.info("Carica una o più immagini per iniziare.")
