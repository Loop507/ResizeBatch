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
import json
import math
import os
import re
import zipfile
from dataclasses import dataclass, replace

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
FONT_PATHS = {
    "Regular": os.path.join(FONT_DIR, "IBMPlexMono-Regular.ttf"),
    "Bold": os.path.join(FONT_DIR, "IBMPlexMono-Bold.ttf"),
}

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
# PROCESSED_SCHEMA_VERSION va incrementata ogni volta che cambia la forma
# delle tuple salvate in st.session_state.processed (es. numero di campi).
# Senza questo controllo, un redeploy con una struttura dati diversa può far
# crashare l'app con un ValueError/unpacking error se la sessione del
# browser conservava ancora risultati elaborati con la vecchia struttura.
PROCESSED_SCHEMA_VERSION = 2  # (nome, bytes, thumb_orig, thumb_out, dim_orig, dim_out)

if st.session_state.get("processed_schema_version") != PROCESSED_SCHEMA_VERSION:
    st.session_state.processed = []
    st.session_state.zip_bytes = None
    st.session_state.processed_schema_version = PROCESSED_SCHEMA_VERSION

if "processed" not in st.session_state:
    st.session_state.processed = []  # list of (filename, bytes, orig_thumb, out_thumb, orig_size, out_size)
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
    transform: str = "none"  # none | rot90cw | rot90ccw | rot180 | flip_h | flip_v


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


@dataclass
class ProSettings:
    enabled: bool
    exposure: float          # stop, -2..+2
    shadows: float           # -100..100
    highlights: float        # -100..100
    black_point: int         # 0..50
    white_point: int         # 205..255
    clarity: float           # -100..100
    hsl_enabled: bool
    hsl_range: str            # una delle chiavi di HSL_RANGES
    hsl_hue_shift: float      # -30..30 gradi
    hsl_sat_shift: float      # -100..100
    hsl_light_shift: float    # -100..100
    split_tone_enabled: bool = False
    shadow_color: tuple = (0, 0, 0)      # colore tinta ombre (RGB)
    shadow_amount: int = 0               # 0..100
    highlight_color: tuple = (255, 255, 255)  # colore tinta luci (RGB)
    highlight_amount: int = 0            # 0..100


# centro tonalità (in gradi, 0-360) per ciascuna banda colore selezionabile
# nell'HSL selettivo — stesse bande concettuali di Lightroom/Capture One
HSL_RANGES = {
    "Rossi": 0,
    "Arancioni": 30,
    "Gialli": 60,
    "Verdi": 120,
    "Ciano": 180,
    "Blu": 240,
    "Viola": 270,
    "Magenta": 315,
}

# Whitelist delle chiavi controllabili da un preset (built-in o caricato da JSON).
# Usata anche per validare i file .json caricati dall'utente, evitando di scrivere
# in session_state chiavi arbitrarie non previste.
ALL_PRESET_KEYS = {
    "cc_enabled_k", "cc_awb_k", "cc_auto_levels_k", "cc_brightness_k", "cc_contrast_k",
    "cc_saturation_k", "cc_scurve_k",
    "pro_enabled_k", "pro_exposure_k", "pro_shadows_k", "pro_highlights_k",
    "pro_black_k", "pro_white_k", "pro_clarity_k",
    "pro_hsl_enabled_k", "pro_hsl_range_k", "pro_hsl_hue_k", "pro_hsl_sat_k", "pro_hsl_light_k",
    "split_tone_enabled_k", "split_shadow_color_k", "split_shadow_amount_k",
    "split_highlight_color_k", "split_highlight_amount_k",
    "denoise_enabled_k", "denoise_strength_k", "sharpen_enabled_k", "sharpen_amount_k",
    "vignette_enabled_k", "vignette_amount_k", "vignette_feather_k",
}

# Preset di stile predefiniti. Calibrati prendendo spunto da trend di color grading
# reali (teal & orange cinematico, moody film con ombre calde, simulazioni pellicola
# stile Kodak Portra, pastello/matte da social) più un preset firmato Loop507.
BUILTIN_PRESETS = {
    "Pop Art": {
        "cc_enabled_k": True, "cc_saturation_k": 1.6, "cc_contrast_k": 1.2, "cc_brightness_k": 1.05,
        "pro_enabled_k": True, "pro_clarity_k": 40,
        "pro_hsl_enabled_k": True, "pro_hsl_range_k": "Rossi", "pro_hsl_hue_k": 0, "pro_hsl_sat_k": 60, "pro_hsl_light_k": 10,
    },
    "Cyberpunk Neon": {
        "cc_enabled_k": True, "cc_saturation_k": 1.3, "cc_contrast_k": 1.15,
        "pro_enabled_k": True, "pro_clarity_k": 30,
        "pro_hsl_enabled_k": True, "pro_hsl_range_k": "Blu", "pro_hsl_hue_k": -10, "pro_hsl_sat_k": 70, "pro_hsl_light_k": 10,
        "vignette_enabled_k": True, "vignette_amount_k": 55, "vignette_feather_k": 30,
    },
    "Teal & Orange Cinematic": {
        "cc_enabled_k": True, "cc_saturation_k": 1.05, "cc_contrast_k": 1.15,
        "pro_enabled_k": True, "pro_exposure_k": -0.1, "pro_clarity_k": 15,
        "split_tone_enabled_k": True, "split_shadow_color_k": "#1B4F72", "split_shadow_amount_k": 40,
        "split_highlight_color_k": "#E8934A", "split_highlight_amount_k": 35,
        "vignette_enabled_k": True, "vignette_amount_k": 25, "vignette_feather_k": 50,
    },
    "Moody Film": {
        "cc_enabled_k": True, "cc_saturation_k": 0.85, "cc_contrast_k": 1.05,
        "pro_enabled_k": True, "pro_shadows_k": 20, "pro_highlights_k": -25, "pro_black_k": 10, "pro_clarity_k": -10,
        "split_tone_enabled_k": True, "split_shadow_color_k": "#4A3728", "split_shadow_amount_k": 25,
        "split_highlight_color_k": "#D9C7A3", "split_highlight_amount_k": 15,
    },
    "Kodak Portra Warm": {
        "cc_enabled_k": True, "cc_saturation_k": 1.1, "cc_brightness_k": 1.05,
        "pro_enabled_k": True, "pro_exposure_k": 0.1, "pro_highlights_k": -15, "pro_clarity_k": 10,
    },
    "Pastel Matte": {
        "cc_enabled_k": True, "cc_saturation_k": 0.8, "cc_contrast_k": 0.9,
        "pro_enabled_k": True, "pro_black_k": 15, "pro_highlights_k": -20,
    },
    "Noir B&N": {
        "cc_enabled_k": True, "cc_saturation_k": 0.0, "cc_scurve_k": 0.5,
        "pro_enabled_k": True, "pro_clarity_k": 20,
        "vignette_enabled_k": True, "vignette_amount_k": 60, "vignette_feather_k": 35,
    },
    "Clean Portrait": {
        "cc_enabled_k": True, "cc_awb_k": True, "cc_auto_levels_k": True,
        "denoise_enabled_k": True, "denoise_strength_k": 15,
        "sharpen_enabled_k": True, "sharpen_amount_k": 40,
    },
    "Loop507 Glitch": {
        "cc_enabled_k": True, "cc_saturation_k": 1.2, "cc_contrast_k": 1.1,
        "pro_enabled_k": True, "pro_clarity_k": 45,
        "pro_hsl_enabled_k": True, "pro_hsl_range_k": "Magenta", "pro_hsl_hue_k": 10, "pro_hsl_sat_k": 50, "pro_hsl_light_k": 0,
        "vignette_enabled_k": True, "vignette_amount_k": 45, "vignette_feather_k": 25,
    },
}


# ---------------------------------------------------------------------------
# Callback per i pulsanti che scrivono su session_state per chiavi di widget
# già istanziati. IMPORTANTE: Streamlit vieta di scrivere in session_state[key]
# per un widget con quella key DOPO che è già stato creato nella stessa run
# (StreamlitAPIException). La soluzione corretta è on_click: il callback gira
# PRIMA che lo script (e quindi i widget) vengano ri-eseguiti, quindi non c'è
# alcun conflitto — mai fare l'assegnazione diretta seguita da un st.rerun()
# manuale per questi casi.
# ---------------------------------------------------------------------------
def _reset_all_preset_keys():
    for k in ALL_PRESET_KEYS:
        st.session_state.pop(k, None)
    st.session_state.pop("auto_optimize_report", None)


def apply_builtin_preset_cb(preset_name: str):
    if preset_name == "(nessuno)":
        return
    _reset_all_preset_keys()
    for k, v in BUILTIN_PRESETS[preset_name].items():
        st.session_state[k] = v


def apply_loaded_preset_cb(uploaded_file):
    if uploaded_file is None:
        return
    try:
        uploaded_file.seek(0)
        loaded = json.loads(uploaded_file.read())
    except (json.JSONDecodeError, UnicodeDecodeError):
        st.session_state["_preset_load_error"] = "File non valido: deve essere un JSON di preset esportato da questa app."
        return
    applied = 0
    _reset_all_preset_keys()
    for k, v in loaded.items():
        if k in ALL_PRESET_KEYS:
            st.session_state[k] = v
            applied += 1
    st.session_state["_preset_load_error"] = None if applied > 0 else "Il file non contiene chiavi di preset riconosciute."


def reset_manual_settings_cb():
    _reset_all_preset_keys()


@dataclass
class RetouchSettings:
    straighten_angle: float    # -45..45 gradi
    crop_top: int              # % 0..40
    crop_bottom: int
    crop_left: int
    crop_right: int
    denoise_enabled: bool
    denoise_strength: int      # 0..100
    sharpen_enabled: bool
    sharpen_amount: int        # 0..200 (percent unsharp mask)
    vignette_enabled: bool
    vignette_amount: int       # 0..100
    vignette_feather: int      # 0..100


@dataclass
class TitleSettings:
    enabled: bool
    text: str
    position: str  # es. "bottom-center"
    font_style: str  # "Regular" o "Bold"
    font_size: int
    color: tuple
    opacity: float  # 0-1
    bg_enabled: bool
    bg_color: tuple
    bg_opacity: float  # 0-1
    margin: int


def sanitize_filename(name: str) -> str:
    """Rimuove caratteri non validi nei nomi file (separatori di percorso,
    virgolette, ecc.) per evitare problemi nello ZIP o sul filesystem."""
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", name).strip()
    return cleaned or "foto"


# ---------------------------------------------------------------------------
# Sidebar :: parametri resize
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader(":: preset")
    preset_choice = st.selectbox(
        "Preset di stile predefiniti",
        options=["(nessuno)"] + list(BUILTIN_PRESETS.keys()),
        help="Applica un punto di partenza di colore/stile pronto. Puoi sempre affinare "
             "a mano dopo, negli slider qui sotto.",
    )
    st.button("Applica preset selezionato", on_click=apply_builtin_preset_cb, args=(preset_choice,))

    with st.expander("Preset personalizzati (carica/salva .json)"):
        uploaded_preset_file = st.file_uploader("Carica preset (.json)", type=["json"], key="preset_upload_k")
        if uploaded_preset_file is not None:
            st.button("Applica preset caricato", on_click=apply_loaded_preset_cb, args=(uploaded_preset_file,))
            if st.session_state.get("_preset_load_error"):
                st.error(st.session_state["_preset_load_error"])
        st.caption("Per salvare il preset con le impostazioni attuali, scorri in fondo alla sidebar.")

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
    st.subheader(":: trasformazione geometrica")
    transform_labels = {
        "none": "Nessuna",
        "rot90cw": "Ruota 90° orario",
        "rot90ccw": "Ruota 90° antiorario",
        "rot180": "Ruota 180°",
        "flip_h": "Rifletti orizzontale (specchio sx/dx)",
        "flip_v": "Rifletti verticale (specchio alto/basso)",
    }
    transform = st.selectbox(
        "Ruota / rifletti tutte le foto",
        options=list(transform_labels.keys()),
        format_func=lambda k: transform_labels[k],
        index=0,
        help="Applicata PRIMA del resize. Con le rotazioni a 90° larghezza e "
             "altezza target vengono scambiate automaticamente, così una foto "
             "1280x720 ruotata diventa 720x1280 invece di essere forzata nel "
             "rapporto originale.",
    )
    if transform in ("rot90cw", "rot90ccw"):
        st.caption(f":: con questa rotazione l'output sarà {int(target_h)}x{int(target_w)} invece di {int(target_w)}x{int(target_h)}")

    st.divider()
    output_format = st.selectbox(
        "Formato output", options=["JPEG", "PNG", "TIFF"], index=0,
        help="TIFF: senza perdita, per stampa/archivio professionale (file più pesanti). "
             "PNG: senza perdita, supporta trasparenza. JPEG: compresso, file più leggeri.",
    )
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
    zip_name_default = sanitize_filename(rename_base) if rename_base else "resizebatch"
    zip_name = st.text_input(
        "Nome file ZIP (senza estensione)",
        value=zip_name_default,
        placeholder="es. lavoro_cliente_finale",
        help="Il nome dello ZIP scaricabile. Di default riprende il 'Nome base' qui sopra.",
    )

    st.divider()
    st.subheader(":: correzione colore")
    cc_enabled = st.checkbox("Applica correzione colore", value=False, key="cc_enabled_k")
    cc_awb = cc_auto_levels = False
    cc_cutoff = 1.0
    cc_brightness = cc_contrast = cc_saturation = 1.0
    cc_scurve = 0.0
    if cc_enabled:
        cc_awb = st.checkbox("Auto White Balance (bilanciamento del bianco)", value=False, key="cc_awb_k")
        cc_auto_levels = st.checkbox("Auto livelli (stretch nero/bianco)", value=False, key="cc_auto_levels_k")
        if cc_auto_levels:
            cc_cutoff = st.slider("Cutoff auto livelli (%)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
        cc_brightness = st.slider("Luminosità", min_value=0.5, max_value=1.5, value=1.0, step=0.05, key="cc_brightness_k")
        cc_contrast = st.slider("Contrasto", min_value=0.5, max_value=1.5, value=1.0, step=0.05, key="cc_contrast_k")
        cc_saturation = st.slider("Saturazione", min_value=0.0, max_value=2.0, value=1.0, step=0.05, key="cc_saturation_k")
        cc_scurve = st.slider("Curva a S (contrasto tonale)", min_value=0.0, max_value=1.0, value=0.0, step=0.05, key="cc_scurve_k")

    settings = Settings(
        target_w=int(target_w),
        target_h=int(target_h),
        mode=mode,
        pad_color=pad_color,
        output_format=output_format,
        jpeg_quality=jpeg_quality,
        rename_base=rename_base.strip(),
        transform=transform,
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
    st.subheader(":: regolazioni pro")
    pro_enabled = st.checkbox("Applica regolazioni pro", value=False, key="pro_enabled_k")
    pro_exposure = 0.0
    pro_shadows = pro_highlights = 0.0
    pro_black = 0
    pro_white = 255
    pro_clarity = 0.0
    pro_hsl_enabled = False
    pro_hsl_range = "Rossi"
    pro_hsl_hue = pro_hsl_sat = pro_hsl_light = 0.0
    split_tone_enabled = False
    shadow_hex = "#1B4F72"
    highlight_hex = "#E67E22"
    split_shadow_color = (0, 0, 0)
    split_shadow_amount = 0
    split_highlight_color = (255, 255, 255)
    split_highlight_amount = 0
    if pro_enabled:
        pro_exposure = st.slider("Esposizione (stop)", min_value=-2.0, max_value=2.0, value=0.0, step=0.1, key="pro_exposure_k")
        pro_shadows = st.slider("Ombre", min_value=-100, max_value=100, value=0, step=5, key="pro_shadows_k")
        pro_highlights = st.slider("Luci", min_value=-100, max_value=100, value=0, step=5, key="pro_highlights_k")
        col_bp, col_wp = st.columns(2)
        with col_bp:
            pro_black = st.slider("Punto neri", min_value=0, max_value=50, value=0, step=1, key="pro_black_k")
        with col_wp:
            pro_white = st.slider("Punto bianchi", min_value=205, max_value=255, value=255, step=1, key="pro_white_k")
        pro_clarity = st.slider(
            "Chiarezza (contrasto locale)", min_value=-100, max_value=100, value=0, step=5,
            help="Positiva: aumenta il 'pop' dei dettagli. Negativa: effetto morbido/dreamy.",
            key="pro_clarity_k",
        )

        st.caption(":: HSL selettivo — regola una singola banda di colore")
        pro_hsl_enabled = st.checkbox("Applica HSL selettivo", value=False, key="pro_hsl_enabled_k")
        if pro_hsl_enabled:
            pro_hsl_range = st.selectbox("Banda colore", options=list(HSL_RANGES.keys()), index=0, key="pro_hsl_range_k")
            pro_hsl_hue = st.slider("Tonalità (°)", min_value=-30, max_value=30, value=0, step=1, key="pro_hsl_hue_k")
            pro_hsl_sat = st.slider("Saturazione", min_value=-100, max_value=100, value=0, step=5, key="pro_hsl_sat_k")
            pro_hsl_light = st.slider("Luminosità", min_value=-100, max_value=100, value=0, step=5, key="pro_hsl_light_k")

        st.caption(":: color grading (split toning) — tinge ombre e luci separatamente")
        split_tone_enabled = st.checkbox("Applica color grading", value=False, key="split_tone_enabled_k")
        if split_tone_enabled:
            col_sh, col_hi = st.columns(2)
            with col_sh:
                st.caption("Ombre")
                shadow_hex = st.color_picker("Colore ombre", value="#1B4F72", key="split_shadow_color_k")
                split_shadow_color = tuple(int(shadow_hex.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
                split_shadow_amount = st.slider("Intensità ombre", min_value=0, max_value=100, value=30, step=5, key="split_shadow_amount_k")
            with col_hi:
                st.caption("Luci")
                highlight_hex = st.color_picker("Colore luci", value="#E67E22", key="split_highlight_color_k")
                split_highlight_color = tuple(int(highlight_hex.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
                split_highlight_amount = st.slider("Intensità luci", min_value=0, max_value=100, value=30, step=5, key="split_highlight_amount_k")

    pro_settings = ProSettings(
        enabled=pro_enabled,
        exposure=pro_exposure,
        shadows=pro_shadows,
        highlights=pro_highlights,
        black_point=pro_black,
        white_point=pro_white,
        clarity=pro_clarity,
        hsl_enabled=pro_hsl_enabled,
        hsl_range=pro_hsl_range,
        hsl_hue_shift=pro_hsl_hue,
        hsl_sat_shift=pro_hsl_sat,
        hsl_light_shift=pro_hsl_light,
        split_tone_enabled=split_tone_enabled,
        shadow_color=split_shadow_color,
        shadow_amount=split_shadow_amount,
        highlight_color=split_highlight_color,
        highlight_amount=split_highlight_amount,
    )

    st.divider()
    st.subheader(":: ritocco tecnico")

    with st.expander("Raddrizza / Crop"):
        straighten_angle = st.slider("Raddrizza (°)", min_value=-45.0, max_value=45.0, value=0.0, step=0.5)
        st.caption("Taglio manuale dai bordi (%)")
        col_ct, col_cb = st.columns(2)
        with col_ct:
            crop_top = st.slider("Alto", min_value=0, max_value=40, value=0, step=1)
        with col_cb:
            crop_bottom = st.slider("Basso", min_value=0, max_value=40, value=0, step=1)
        col_cl, col_cr = st.columns(2)
        with col_cl:
            crop_left = st.slider("Sinistra", min_value=0, max_value=40, value=0, step=1)
        with col_cr:
            crop_right = st.slider("Destra", min_value=0, max_value=40, value=0, step=1)

    with st.expander("Nitidezza / Rumore"):
        sharpen_enabled = st.checkbox("Applica nitidezza", value=False, key="sharpen_enabled_k")
        sharpen_amount = 0
        if sharpen_enabled:
            sharpen_amount = st.slider("Quantità nitidezza (%)", min_value=0, max_value=200, value=80, step=10, key="sharpen_amount_k")

        denoise_enabled = st.checkbox("Applica riduzione rumore", value=False, key="denoise_enabled_k")
        denoise_strength = 0
        if denoise_enabled:
            denoise_strength = st.slider(
                "Intensità denoise", min_value=0, max_value=100, value=30, step=5,
                help="Valori alti riducono di più il rumore ma possono ammorbidire i dettagli fini.",
                key="denoise_strength_k",
            )

    with st.expander("Vignettatura"):
        vignette_enabled = st.checkbox("Applica vignettatura", value=False, key="vignette_enabled_k")
        vignette_amount = 0
        vignette_feather = 50
        if vignette_enabled:
            vignette_amount = st.slider("Intensità vignetta", min_value=0, max_value=100, value=40, step=5, key="vignette_amount_k")
            vignette_feather = st.slider(
                "Morbidezza bordo", min_value=0, max_value=100, value=50, step=5,
                help="Bassa = vignetta che parte più vicino al centro (più marcata). Alta = falloff più graduale.",
                key="vignette_feather_k",
            )

    retouch_settings = RetouchSettings(
        straighten_angle=straighten_angle,
        crop_top=crop_top,
        crop_bottom=crop_bottom,
        crop_left=crop_left,
        crop_right=crop_right,
        denoise_enabled=denoise_enabled,
        denoise_strength=denoise_strength,
        sharpen_enabled=sharpen_enabled,
        sharpen_amount=sharpen_amount,
        vignette_enabled=vignette_enabled,
        vignette_amount=vignette_amount,
        vignette_feather=vignette_feather,
    )

    st.divider()
    st.subheader(":: titolo / testo")
    title_enabled = st.checkbox("Aggiungi titolo a tutte le foto", value=False)
    title_text = ""
    title_position = "bottom-center"
    title_font_style = "Bold"
    title_font_size = 48
    title_color = (255, 255, 255)
    title_opacity = 1.0
    title_bg_enabled = False
    title_bg_color = (0, 0, 0)
    title_bg_opacity = 0.5
    title_margin = 40
    if title_enabled:
        title_text = st.text_input("Testo del titolo", value="LOOP507")
        position_labels = {
            "top-left": "alto sinistra", "top-center": "alto centro", "top-right": "alto destra",
            "center-left": "centro sinistra", "center": "centro", "center-right": "centro destra",
            "bottom-left": "basso sinistra", "bottom-center": "basso centro", "bottom-right": "basso destra",
        }
        title_position = st.selectbox(
            "Posizione", options=list(position_labels.keys()),
            format_func=lambda k: position_labels[k], index=7,
        )
        title_font_style = st.radio("Stile font", options=["Regular", "Bold"], index=1, horizontal=True)
        title_font_size = st.slider("Dimensione (px)", min_value=12, max_value=300, value=48, step=2)
        title_color_hex = st.color_picker("Colore testo", value="#FFFFFF")
        title_color = tuple(int(title_color_hex.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        title_opacity = st.slider("Opacità testo (%)", min_value=10, max_value=100, value=100, step=5) / 100
        title_margin = st.slider("Margine dai bordi (px)", min_value=0, max_value=200, value=40, step=5)
        title_bg_enabled = st.checkbox("Sfondo dietro al testo", value=False)
        if title_bg_enabled:
            title_bg_color_hex = st.color_picker("Colore sfondo", value="#000000")
            title_bg_color = tuple(int(title_bg_color_hex.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
            title_bg_opacity = st.slider("Opacità sfondo (%)", min_value=10, max_value=100, value=50, step=5) / 100

    title_settings = TitleSettings(
        enabled=title_enabled,
        text=title_text,
        position=title_position,
        font_style=title_font_style,
        font_size=title_font_size,
        color=title_color,
        opacity=title_opacity,
        bg_enabled=title_bg_enabled,
        bg_color=title_bg_color,
        bg_opacity=title_bg_opacity,
        margin=title_margin,
    )

    st.divider()
    if settings.transform in ("rot90cw", "rot90ccw"):
        st.caption(f":: target ratio (dopo rotazione) = {settings.target_h / settings.target_w:.3f}")
    else:
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

    st.divider()
    with st.expander("💾 Salva impostazioni attuali come preset"):
        current_preset = {
            "cc_enabled_k": cc_enabled, "cc_awb_k": cc_awb, "cc_auto_levels_k": cc_auto_levels,
            "cc_brightness_k": cc_brightness, "cc_contrast_k": cc_contrast,
            "cc_saturation_k": cc_saturation, "cc_scurve_k": cc_scurve,
            "pro_enabled_k": pro_enabled, "pro_exposure_k": pro_exposure,
            "pro_shadows_k": pro_shadows, "pro_highlights_k": pro_highlights,
            "pro_black_k": pro_black, "pro_white_k": pro_white, "pro_clarity_k": pro_clarity,
            "pro_hsl_enabled_k": pro_hsl_enabled, "pro_hsl_range_k": pro_hsl_range,
            "pro_hsl_hue_k": pro_hsl_hue, "pro_hsl_sat_k": pro_hsl_sat, "pro_hsl_light_k": pro_hsl_light,
            "split_tone_enabled_k": split_tone_enabled, "split_shadow_color_k": shadow_hex,
            "split_shadow_amount_k": split_shadow_amount, "split_highlight_color_k": highlight_hex,
            "split_highlight_amount_k": split_highlight_amount,
            "denoise_enabled_k": denoise_enabled, "denoise_strength_k": denoise_strength,
            "sharpen_enabled_k": sharpen_enabled, "sharpen_amount_k": sharpen_amount,
            "vignette_enabled_k": vignette_enabled, "vignette_amount_k": vignette_amount,
            "vignette_feather_k": vignette_feather,
        }
        st.download_button(
            "⬇ scarica preset (.json)",
            data=json.dumps(current_preset, indent=2),
            file_name="mio_preset_loop507.json",
            mime="application/json",
        )
        st.caption("Ricaricalo in qualsiasi sessione futura con 'Preset personalizzati' in cima alla sidebar.")

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Carica le immagini (JPG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

custom_names = {}
if uploaded_files:
    with st.expander(":: rinomina individuale (avanzata)"):
        st.caption(
            "Sovrascrive il 'Nome base' della sidebar file per file. Lascia vuoto per usare "
            "il nome automatico. Utile per dare nomi diversi a foto diverse dello stesso batch "
            "(es. 'foto_x', 'foto_y')."
        )
        rename_df = pd.DataFrame({
            "file originale": [uf.name for uf in uploaded_files],
            "nome personalizzato": ["" for _ in uploaded_files],
        })
        edited_df = st.data_editor(
            rename_df,
            key="rename_table",
            hide_index=True,
            use_container_width=True,
            disabled=["file originale"],
        )
        for _, row in edited_df.iterrows():
            if row["nome personalizzato"].strip():
                custom_names[row["file originale"]] = row["nome personalizzato"].strip()


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
# Regolazioni pro (esposizione, ombre/luci, punti nero/bianco, chiarezza, HSL)
# ---------------------------------------------------------------------------
def apply_exposure(img: Image.Image, stops: float) -> Image.Image:
    """Esposizione in stop fotografici: ogni stop raddoppia/dimezza la
    luce, coerente con la convenzione fotografica (non una semplice
    somma lineare come 'luminosità')."""
    if stops == 0:
        return img
    arr = np.asarray(img).astype(np.float32) * (2.0 ** stops)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def apply_shadows_highlights(img: Image.Image, shadows: float, highlights: float) -> Image.Image:
    """Solleva/abbassa selettivamente le ombre e le luci in base alla
    luminanza del pixel, lasciando i mezzitoni relativamente intatti —
    stesso principio dei cursori Ombre/Luci di Lightroom."""
    if shadows == 0 and highlights == 0:
        return img
    arr = np.asarray(img).astype(np.float32)
    lum = arr.mean(axis=2)  # luminanza approssimata

    shadow_weight = np.clip(1.0 - lum / 128.0, 0.0, 1.0)       # forte su pixel scuri
    highlight_weight = np.clip((lum - 128.0) / 127.0, 0.0, 1.0)  # forte su pixel chiari

    delta = (shadows / 100.0) * 80.0 * shadow_weight + (highlights / 100.0) * 80.0 * highlight_weight
    arr = arr + delta[:, :, None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def apply_black_white_point(img: Image.Image, black: int, white: int) -> Image.Image:
    """Stira i livelli: il punto nero diventa 0, il punto bianco diventa
    255, tutto il resto viene rimappato proporzionalmente (levels)."""
    if black <= 0 and white >= 255:
        return img
    white = max(white, black + 1)  # evita divisione per zero
    arr = np.asarray(img).astype(np.float32)
    arr = (arr - black) / (white - black) * 255.0
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def apply_clarity(img: Image.Image, amount: float) -> Image.Image:
    """Contrasto locale: sottrae una versione molto sfocata dell'immagine
    da se stessa per isolare i dettagli a media frequenza, poi li
    riaggiunge amplificati. Valori negativi ammorbidiscono (effetto
    'dreamy')."""
    if amount == 0:
        return img
    arr = np.asarray(img).astype(np.float32)
    blurred = np.asarray(img.filter(ImageFilter.GaussianBlur(radius=25))).astype(np.float32)
    detail = arr - blurred
    arr = arr + detail * (amount / 100.0)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def _hue_distance(h: np.ndarray, center_deg: float) -> np.ndarray:
    """Distanza angolare circolare tra ogni hue (0-255, scala PIL) e un
    centro dato in gradi (0-360). Ritorna gradi 0-180."""
    h_deg = h.astype(np.float32) * (360.0 / 255.0)
    diff = np.abs(h_deg - center_deg) % 360.0
    return np.minimum(diff, 360.0 - diff)


def apply_hsl_selective(img: Image.Image, color_range: str, hue_shift: float, sat_shift: float, light_shift: float) -> Image.Image:
    """Regola tonalità/saturazione/luminosità solo per i pixel la cui
    tonalità ricade in una banda di colore (es. 'solo i rossi'), con una
    dissolvenza morbida ai bordi della banda per evitare transizioni dure."""
    if hue_shift == 0 and sat_shift == 0 and light_shift == 0:
        return img

    center = HSL_RANGES.get(color_range, 0)
    hsv = np.asarray(img.convert("HSV")).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    dist = _hue_distance(h, center)
    band_half_width = 35.0  # gradi di banda con falloff morbido
    mask = np.clip(1.0 - dist / band_half_width, 0.0, 1.0)
    mask = mask * mask * (3 - 2 * mask)  # smoothstep

    if hue_shift != 0:
        h_deg = h * (360.0 / 255.0)
        h_deg = (h_deg + hue_shift * mask) % 360.0
        h = h_deg * (255.0 / 360.0)
    if sat_shift != 0:
        s = s * (1.0 + (sat_shift / 100.0) * mask)
    if light_shift != 0:
        v = v * (1.0 + (light_shift / 100.0) * mask)

    hsv_out = np.stack([
        np.clip(h, 0, 255),
        np.clip(s, 0, 255),
        np.clip(v, 0, 255),
    ], axis=2).astype(np.uint8)
    return Image.fromarray(hsv_out, mode="HSV").convert("RGB")


def apply_split_toning(img: Image.Image, shadow_color: tuple, shadow_amount: int, highlight_color: tuple, highlight_amount: int) -> Image.Image:
    """Color grading a due bande (split toning): tinge le ombre con un
    colore e le luci con un altro, sfumando in base alla luminanza di ogni
    pixel — lo stesso principio usato per il vero look 'teal & orange'
    cinematico, dove ombre e luci ricevono una grana di colore indipendente
    (diverso dall'HSL selettivo, che seleziona per tonalità, non per
    luminanza)."""
    if shadow_amount <= 0 and highlight_amount <= 0:
        return img

    arr = np.asarray(img).astype(np.float32)
    lum = arr.mean(axis=2)
    shadow_weight = np.clip(1.0 - lum / 128.0, 0.0, 1.0)
    highlight_weight = np.clip((lum - 128.0) / 127.0, 0.0, 1.0)

    MAX_BLEND = 0.5  # tetto di miscelazione: mantiene un effetto di 'tinta', non un colore piatto
    out = arr.copy()

    if shadow_amount > 0:
        shadow_arr = np.array(shadow_color, dtype=np.float32)
        w = (shadow_weight * (shadow_amount / 100.0) * MAX_BLEND)[:, :, None]
        out = out + w * (shadow_arr - out)

    if highlight_amount > 0:
        highlight_arr = np.array(highlight_color, dtype=np.float32)
        w = (highlight_weight * (highlight_amount / 100.0) * MAX_BLEND)[:, :, None]
        out = out + w * (highlight_arr - out)

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def apply_pro_adjustments(img: Image.Image, ps: ProSettings) -> Image.Image:
    if not ps.enabled:
        return img
    out = img
    out = apply_exposure(out, ps.exposure)
    out = apply_shadows_highlights(out, ps.shadows, ps.highlights)
    out = apply_black_white_point(out, ps.black_point, ps.white_point)
    out = apply_clarity(out, ps.clarity)
    if ps.hsl_enabled:
        out = apply_hsl_selective(out, ps.hsl_range, ps.hsl_hue_shift, ps.hsl_sat_shift, ps.hsl_light_shift)
    if ps.split_tone_enabled:
        out = apply_split_toning(out, ps.shadow_color, ps.shadow_amount, ps.highlight_color, ps.highlight_amount)
    return out


def compute_rgb_histogram(img: Image.Image, bins: int = 64) -> dict:
    """Istogramma RGB a risoluzione ridotta (default 64 bin) pronto per
    essere passato a st.line_chart — mostra la distribuzione tonale
    dell'immagine corrente in tempo reale."""
    arr = np.asarray(img.convert("RGB"))
    hist = {}
    for i, channel in enumerate(["R", "G", "B"]):
        counts, _ = np.histogram(arr[:, :, i], bins=bins, range=(0, 255))
        hist[channel] = counts
    return hist


# ---------------------------------------------------------------------------
# Ottimizzazione automatica (analizza una foto e suggerisce le regolazioni)
# ---------------------------------------------------------------------------
ANALYSIS_MAX_SIDE = 700  # l'analisi lavora su una copia ridotta, per velocità


def analyze_photo(img: Image.Image) -> dict:
    """Analizza una foto di riferimento e restituisce impostazioni consigliate
    per correzione colore, regolazioni pro e ritocco tecnico — un'euristica
    ispirata a come un fotografo valuta rapidamente uno scatto: esposizione,
    bilanciamento colore, contrasto, saturazione, rumore/nitidezza."""
    small = flatten_to_rgb(img).copy()
    small.thumbnail((ANALYSIS_MAX_SIDE, ANALYSIS_MAX_SIDE), Image.LANCZOS)
    arr = np.asarray(small).astype(np.float32)

    lum = arr.mean(axis=2)
    mean_lum = float(lum.mean())
    p2, p98 = np.percentile(lum, [2, 98])
    contrast_std = float(lum.std())

    means = arr.reshape(-1, 3).mean(axis=0)
    color_cast = float(means.max() - means.min())

    hsv = np.asarray(small.convert("HSV")).astype(np.float32)
    mean_sat = float(hsv[:, :, 1].mean())

    gray = cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    result = {
        "mean_lum": mean_lum, "contrast_std": contrast_std,
        "color_cast": color_cast, "mean_sat": mean_sat, "laplacian_var": laplacian_var,
    }

    # --- esposizione: riporta la luminanza media verso ~128 ---
    target_lum = 128.0
    if mean_lum < 95:
        result["exposure"] = round(min(1.2, (target_lum - mean_lum) / 90), 2)
    elif mean_lum > 165:
        result["exposure"] = round(max(-1.2, (target_lum - mean_lum) / 90), 2)
    else:
        result["exposure"] = 0.0

    # --- white balance: dominante di colore evidente tra i canali medi ---
    result["awb"] = bool(color_cast > 15)

    # --- auto livelli: se il range dinamico (2°-98° percentile) è compresso ---
    result["auto_levels"] = bool((p98 - p2) < 180)

    # --- ombre/luci in base ai percentili estremi ---
    result["shadows"] = 30 if p2 > 40 else 0
    result["highlights"] = -30 if p98 < 215 else 0

    # --- chiarezza: se il contrasto locale è basso, dai un po' di 'pop' ---
    result["clarity"] = 25 if contrast_std < 40 else 0

    # --- saturazione: se la foto appare smorta, alzala leggermente ---
    result["saturation"] = 1.15 if mean_sat < 70 else 1.0

    # --- denoise/nitidezza in base all'energia dei bordi (varianza Laplaciana) ---
    # euristica empirica: molto rumore/texture fine -> varianza alta e "sporca";
    # immagine morbida/sfocata -> varianza bassa
    result["denoise_strength"] = 25 if laplacian_var > 800 else 0
    result["sharpen_amount"] = 60 if laplacian_var < 150 else 0

    return result


# ---------------------------------------------------------------------------
# Ritocco tecnico (raddrizza/crop, denoise, nitidezza, vignettatura)
# ---------------------------------------------------------------------------
def _rotated_rect_max_area(w: float, h: float, angle_rad: float) -> tuple:
    """Calcola le dimensioni del più grande rettangolo assiale che sta
    interamente dentro un rettangolo WxH ruotato di angle_rad, senza
    includere gli angoli vuoti/neri generati dalla rotazione."""
    if w <= 0 or h <= 0:
        return 0, 0
    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    sin_a = abs(math.sin(angle_rad))
    cos_a = abs(math.cos(angle_rad))

    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * side_short
        if width_is_longer:
            wr, hr = x / sin_a, x / cos_a
        else:
            wr, hr = x / cos_a, x / sin_a
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr = (w * cos_a - h * sin_a) / cos_2a
        hr = (h * cos_a - w * sin_a) / cos_2a

    return wr, hr


def apply_straighten(img: Image.Image, angle_deg: float) -> Image.Image:
    """Ruota l'immagine per raddrizzarla e ritaglia automaticamente il
    rettangolo più grande possibile senza bordi neri agli angoli."""
    if angle_deg == 0:
        return img
    w, h = img.size
    rotated = img.rotate(-angle_deg, resample=Image.BICUBIC, expand=True)
    wr, hr = _rotated_rect_max_area(w, h, math.radians(angle_deg))
    rw, rh = rotated.size
    wr, hr = min(wr, rw), min(hr, rh)
    x0 = (rw - wr) / 2.0
    y0 = (rh - hr) / 2.0
    return rotated.crop((x0, y0, x0 + wr, y0 + hr))


def apply_manual_crop(img: Image.Image, top: int, bottom: int, left: int, right: int) -> Image.Image:
    """Ritaglia percentuali fisse dai 4 bordi (0-40% ciascuno)."""
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return img
    w, h = img.size
    x0 = int(w * left / 100.0)
    x1 = w - int(w * right / 100.0)
    y0 = int(h * top / 100.0)
    y1 = h - int(h * bottom / 100.0)
    x1 = max(x1, x0 + 1)
    y1 = max(y1, y0 + 1)
    return img.crop((x0, y0, x1, y1))


def apply_denoise(img: Image.Image, strength: int) -> Image.Image:
    """Riduzione del rumore con bilateral filter (OpenCV): leviga il
    rumore preservando i bordi netti molto meglio di un blur uniforme."""
    if strength <= 0:
        return img
    arr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    d = 5 + int(strength / 10)              # diametro del vicinato, cresce con l'intensità
    sigma = 10 + strength * 1.5             # sigma colore/spazio
    denoised = cv2.bilateralFilter(arr, d=d, sigmaColor=sigma, sigmaSpace=sigma)
    return Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB), mode="RGB")


def apply_sharpen(img: Image.Image, amount: int) -> Image.Image:
    """Nitidezza via unsharp mask: amount è la percentuale (0-200%) di
    contrasto aggiunto ai bordi rilevati."""
    if amount <= 0:
        return img
    return img.filter(ImageFilter.UnsharpMask(radius=2, percent=int(amount), threshold=3))


def apply_vignette(img: Image.Image, amount: int, feather: int) -> Image.Image:
    """Scurisce gradualmente i bordi dell'immagine verso gli angoli,
    con un raggio di partenza (feather) regolabile per un falloff più
    o meno morbido."""
    if amount <= 0:
        return img
    w, h = img.size
    arr = np.asarray(img).astype(np.float32)

    y_idx, x_idx = np.ogrid[:h, :w]
    cx, cy = w / 2.0, h / 2.0
    max_dist = math.sqrt(cx ** 2 + cy ** 2)
    dist = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2) / max_dist  # 0 centro -> 1 angolo

    start = 1.0 - (feather / 100.0) * 0.9
    falloff = np.clip((dist - start) / max(1.0 - start, 1e-6), 0.0, 1.0)
    falloff = falloff * falloff * (3 - 2 * falloff)  # smoothstep

    darken = 1.0 - (amount / 100.0) * falloff
    arr = arr * darken[:, :, None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def apply_retouch(img: Image.Image, rs: RetouchSettings) -> Image.Image:
    """Ordine: raddrizza -> crop manuale -> denoise (prima del resize, a
    piena risoluzione, dove funziona meglio). Nitidezza e vignettatura
    vengono applicate altrove nella pipeline (dopo il resize)."""
    out = apply_straighten(img, rs.straighten_angle)
    out = apply_manual_crop(out, rs.crop_top, rs.crop_bottom, rs.crop_left, rs.crop_right)
    if rs.denoise_enabled:
        out = apply_denoise(out, rs.denoise_strength)
    return out


def apply_retouch_pair(img: Image.Image, alpha, rs: RetouchSettings) -> tuple:
    """Come apply_retouch, ma applica anche a una maschera alpha (se
    presente) le stesse trasformazioni geometriche (raddrizza, crop) — il
    denoise invece resta solo sull'RGB, la trasparenza non va 'denoisata'."""
    out = apply_straighten(img, rs.straighten_angle)
    out = apply_manual_crop(out, rs.crop_top, rs.crop_bottom, rs.crop_left, rs.crop_right)
    if rs.denoise_enabled:
        out = apply_denoise(out, rs.denoise_strength)

    if alpha is None:
        return out, None
    out_alpha = apply_straighten(alpha, rs.straighten_angle)
    out_alpha = apply_manual_crop(out_alpha, rs.crop_top, rs.crop_bottom, rs.crop_left, rs.crop_right)
    return out, out_alpha


def apply_retouch_post_resize(img: Image.Image, rs: RetouchSettings) -> Image.Image:
    """Nitidezza e vignettatura si applicano dopo il resize: la nitidezza
    per essere calibrata sulla risoluzione finale, la vignetta per
    seguire correttamente la geometria del frame di output."""
    out = img
    if rs.sharpen_enabled:
        out = apply_sharpen(out, rs.sharpen_amount)
    if rs.vignette_enabled:
        out = apply_vignette(out, rs.vignette_amount, rs.vignette_feather)
    return out


# ---------------------------------------------------------------------------
# Trasformazione geometrica (rotazione / riflessione) — applicata PRIMA del
# resize, sia all'RGB sia (a monte, sull'immagine intera con alpha) alla
# trasparenza, così restano allineati senza logica separata.
# ---------------------------------------------------------------------------
def apply_geometric_transform(img: Image.Image, transform: str) -> Image.Image:
    """Ruota/riflette l'immagine. Le rotazioni a 90° scambiano larghezza e
    altezza: la geometria del canvas target va scambiata di conseguenza a
    valle (vedi process_image), altrimenti il resize successivo forzerebbe
    l'immagine ruotata dentro il rapporto w/h originale, deformandola o
    ritagliandola in modo indesiderato."""
    if transform == "rot90cw":
        return img.transpose(Image.ROTATE_270)  # 270° antiorario = 90° orario
    elif transform == "rot90ccw":
        return img.transpose(Image.ROTATE_90)
    elif transform == "rot180":
        return img.transpose(Image.ROTATE_180)
    elif transform == "flip_h":
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif transform == "flip_v":
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


# ---------------------------------------------------------------------------
# Funzioni di trasformazione (resize)
# ---------------------------------------------------------------------------
def resize_stretch(img: Image.Image, tw: int, th: int) -> Image.Image:
    return img.resize((tw, th), Image.LANCZOS)


def _crop_box_for_ratio(src_w: int, src_h: int, target_ratio: float) -> tuple:
    """Calcola il box di center-crop (x0, y0, w, h) per portare un'immagine
    src_w x src_h al rapporto target_ratio, senza deformarla."""
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
    return x0, y0, new_w, new_h


def resize_crop(img: Image.Image, tw: int, th: int) -> Image.Image:
    x0, y0, cw, ch = _crop_box_for_ratio(*img.size, tw / th)
    cropped = img.crop((x0, y0, x0 + cw, y0 + ch))
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


def resize_pair(img: Image.Image, alpha, s: Settings, seed: int) -> tuple:
    """Applica il resize sia all'immagine RGB sia (se presente) alla maschera
    alpha, con la STESSA geometria — fondamentale per tenere la trasparenza
    allineata pixel per pixel dopo crop/pad. Nelle modalità pad/pad_glitch,
    le aree aggiunte (letterbox) diventano opache (alpha=255): sono nuovo
    contenuto scelto dall'utente (colore o pattern), non 'buchi' del PNG
    originale."""
    tw, th = s.target_w, s.target_h
    if alpha is None:
        return resize_only(img, s, seed), None

    if s.mode == "stretch":
        out_img = resize_stretch(img, tw, th)
        out_alpha = alpha.resize((tw, th), Image.LANCZOS)
    elif s.mode == "crop":
        x0, y0, cw, ch = _crop_box_for_ratio(*img.size, tw / th)
        out_img = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((tw, th), Image.LANCZOS)
        out_alpha = alpha.crop((x0, y0, x0 + cw, y0 + ch)).resize((tw, th), Image.LANCZOS)
    elif s.mode in ("pad", "pad_glitch"):
        out_img = resize_pad(img, tw, th, s.pad_color) if s.mode == "pad" else resize_pad_glitch(img, tw, th, seed)
        scaled_alpha, off_x, off_y = _fit_inside(alpha, tw, th)
        canvas_alpha = Image.new("L", (tw, th), 255)
        canvas_alpha.paste(scaled_alpha, (off_x, off_y))
        out_alpha = canvas_alpha
    else:
        raise ValueError(f"Modalità sconosciuta: {s.mode}")
    return out_img, out_alpha


def load_font(style: str, size: int) -> ImageFont.FreeTypeFont:
    """Carica il font bundlato in fonts/. Se il file manca (es. repo GitHub
    senza la cartella fonts/), ricade sul font di default di Pillow."""
    path = FONT_PATHS.get(style, FONT_PATHS["Bold"])
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _anchor_xy(position: str, canvas_w: int, canvas_h: int, text_w: int, text_h: int, margin: int) -> tuple:
    """Calcola la posizione (x, y) del testo in base a uno dei 9 ancoraggi."""
    if position == "center":
        v, h = "center", "center"
    else:
        v, h = position.split("-")

    if h == "left":
        x = margin
    elif h == "right":
        x = canvas_w - text_w - margin
    else:  # center
        x = (canvas_w - text_w) // 2

    if v == "top":
        y = margin
    elif v == "bottom":
        y = canvas_h - text_h - margin
    else:  # center
        y = (canvas_h - text_h) // 2

    return x, y


def add_title_text(img: Image.Image, ts: TitleSettings, keep_alpha: bool = False) -> Image.Image:
    """Disegna un titolo testuale sull'immagine, con posizione a 9 ancoraggi,
    colore/opacità regolabili e sfondo opzionale semi-trasparente. Se
    keep_alpha=True ritorna RGBA (usato quando si preserva la trasparenza
    di un PNG); altrimenti appiattisce su RGB come prima."""
    if not ts.enabled or not ts.text.strip():
        return img if keep_alpha else img.convert("RGB")

    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font = load_font(ts.font_style, ts.font_size)
    bbox = draw.textbbox((0, 0), ts.text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x, y = _anchor_xy(ts.position, base.width, base.height, text_w, text_h, ts.margin)

    if ts.bg_enabled:
        pad = max(6, ts.font_size // 6)
        bg_alpha = int(255 * ts.bg_opacity)
        draw.rectangle(
            [x - pad, y - pad, x + text_w + pad, y + text_h + pad],
            fill=(*ts.bg_color, bg_alpha),
        )

    text_alpha = int(255 * ts.opacity)
    # offset verticale: textbbox include un top-offset (bbox[1]) da compensare
    draw.text((x - bbox[0], y - bbox[1]), ts.text, font=font, fill=(*ts.color, text_alpha))

    composited = Image.alpha_composite(base, overlay)
    return composited if keep_alpha else composited.convert("RGB")


def flatten_to_rgb(img: Image.Image) -> Image.Image:
    """Converte in RGB gestendo correttamente la trasparenza: le immagini
    con canale alpha (RGBA/LA, o P con trasparenza) vengono composte su
    uno sfondo bianco invece di un semplice convert('RGB'), che altrimenti
    lascia spesso valori RGB azzerati (neri) nelle zone trasparenti."""
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        alpha = img.convert("RGBA").split()[-1]
        background.paste(img.convert("RGB"), mask=alpha)
        return background
    return img.convert("RGB")


def has_transparency(img: Image.Image) -> bool:
    """Vero se l'immagine ha un canale alpha effettivo (RGBA/LA, o P con
    trasparenza indicizzata)."""
    return img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)


def process_image(img: Image.Image, s: Settings, cs: ColorSettings, ps: ProSettings, rs: RetouchSettings, ts: TitleSettings, seed: int) -> Image.Image:
    """Pipeline completa. Se il formato di output è PNG e la foto originale
    ha trasparenza, l'alpha viene preservato end-to-end (raddrizza/crop/
    resize applicati identicamente alla maschera; le aree di padding
    diventano opache). Regolazioni colore/pro/nitidezza/vignetta lavorano
    solo sull'RGB: la trasparenza non va né corretta né 'denoisata'."""
    # La rotazione/riflessione va applicata per prima, sull'immagine intera
    # (RGB + eventuale alpha insieme via transpose), così la trasparenza
    # resta allineata senza bisogno di una pipeline separata.
    img = apply_geometric_transform(img, s.transform)
    if s.transform in ("rot90cw", "rot90ccw"):
        # con una rotazione a 90° l'immagine scambia larghezza/altezza:
        # il canvas target segue lo scambio, altrimenti il resize successivo
        # rimetterebbe l'immagine ruotata nel rapporto w/h originale.
        s = replace(s, target_w=s.target_h, target_h=s.target_w)

    preserve_alpha = s.output_format in ("PNG", "TIFF") and has_transparency(img)
    alpha = img.convert("RGBA").split()[-1] if preserve_alpha else None

    rgb = flatten_to_rgb(img)
    rgb = apply_color_correction(rgb, cs)
    rgb = apply_pro_adjustments(rgb, ps)
    rgb, alpha = apply_retouch_pair(rgb, alpha, rs)
    rgb, alpha = resize_pair(rgb, alpha, s, seed)
    rgb = apply_retouch_post_resize(rgb, rs)

    if preserve_alpha and alpha is not None:
        rgba_out = Image.merge("RGBA", (*rgb.split(), alpha))
        return add_title_text(rgba_out, ts, keep_alpha=True)
    return add_title_text(rgb, ts, keep_alpha=False)


def image_to_bytes(img: Image.Image, fmt: str, quality: int) -> bytes:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=quality)
    elif fmt == "TIFF":
        img.save(buf, format="TIFF")  # senza perdita, nessuna compressione lossy
    else:
        img.save(buf, format="PNG")
    return buf.getvalue()


PREVIEW_THUMB_MAX = 480


def make_preview_thumbnail(img: Image.Image) -> Image.Image:
    """Crea una miniatura leggera per la sola visualizzazione in anteprima.
    Evita di tenere in memoria le immagini a piena risoluzione (originale +
    elaborata) per l'intero batch — solo i byte già codificati (out_bytes,
    usati per download/ZIP) mantengono la qualità piena."""
    thumb = img.copy()
    thumb.thumbnail((PREVIEW_THUMB_MAX, PREVIEW_THUMB_MAX), Image.LANCZOS)
    return thumb


def dedupe_filename(name: str, used_names: set) -> str:
    """Se il nome è già stato usato in questo batch, aggiunge un suffisso
    numerico (_2, _3...) per evitare collisioni nello ZIP e crash dei
    pulsanti di download (Streamlit richiede chiavi univoche)."""
    if name not in used_names:
        used_names.add(name)
        return name
    base, ext = name.rsplit(".", 1) if "." in name else (name, "")
    counter = 2
    while True:
        candidate = f"{base}_{counter}.{ext}" if ext else f"{base}_{counter}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        counter += 1


def build_output_name(original_name: str, rename_base: str, custom_name: str, index: int, tw: int, th: int, ext: str) -> str:
    """Genera il nome del file di output, in ordine di priorità:
    1. nome personalizzato (dalla tabella di rinomina individuale)
    2. pattern 'nome base' della sidebar (con {n} come placeholder esplicito)
    3. nome originale con suffisso dimensione, se nessuno dei due è impostato
    """
    if custom_name and custom_name.strip():
        base = sanitize_filename(custom_name.strip())
        return f"{base}.{ext}"
    if not rename_base:
        base = sanitize_filename(original_name.rsplit(".", 1)[0])
        return f"{base}.{ext}"
    if "{n}" in rename_base:
        base = rename_base.replace("{n}", str(index))
    else:
        base = f"{rename_base}_{index}"
    return f"{sanitize_filename(base)}.{ext}"


# ---------------------------------------------------------------------------
# Ottimizzazione automatica
# ---------------------------------------------------------------------------
def auto_optimize_cb(ref_file):
    ref_for_analysis = Image.open(ref_file)
    analysis = analyze_photo(ref_for_analysis)

    st.session_state["cc_enabled_k"] = True
    st.session_state["cc_awb_k"] = analysis["awb"]
    st.session_state["cc_auto_levels_k"] = analysis["auto_levels"]
    st.session_state["cc_saturation_k"] = analysis["saturation"]

    st.session_state["pro_enabled_k"] = True
    st.session_state["pro_exposure_k"] = analysis["exposure"]
    st.session_state["pro_shadows_k"] = analysis["shadows"]
    st.session_state["pro_highlights_k"] = analysis["highlights"]
    st.session_state["pro_clarity_k"] = analysis["clarity"]

    st.session_state["denoise_enabled_k"] = analysis["denoise_strength"] > 0
    st.session_state["denoise_strength_k"] = analysis["denoise_strength"]
    st.session_state["sharpen_enabled_k"] = analysis["sharpen_amount"] > 0
    st.session_state["sharpen_amount_k"] = analysis["sharpen_amount"]

    st.session_state["auto_optimize_report"] = analysis


if uploaded_files:
    st.divider()
    st.subheader(":: ottimizzazione automatica")
    st.caption(
        "Analizza la prima foto caricata (esposizione, colore, contrasto, rumore) e propone "
        "regolazioni per tutto il batch — un punto di partenza automatico, sempre modificabile "
        "a mano dopo negli slider della sidebar."
    )
    col_auto1, col_auto2 = st.columns([1, 1])
    with col_auto1:
        st.button(
            "✨ Analizza e ottimizza automaticamente", type="primary",
            on_click=auto_optimize_cb, args=(uploaded_files[0],),
        )
    with col_auto2:
        st.button("↺ Ripristina impostazioni manuali", on_click=reset_manual_settings_cb)

    if "auto_optimize_report" in st.session_state:
        a = st.session_state.auto_optimize_report
        with st.expander("✨ Cosa ha rilevato l'analisi automatica", expanded=False):
            notes = []
            notes.append(f"Luminosità media: {a['mean_lum']:.0f}/255 → esposizione consigliata {a['exposure']:+.2f} stop")
            notes.append(f"Dominante colore rilevata: {a['color_cast']:.0f} → White Balance {'attivato' if a['awb'] else 'non necessario'}")
            notes.append(f"Range dinamico → Auto livelli {'attivati' if a['auto_levels'] else 'non necessari'}")
            notes.append(f"Contrasto locale: {a['contrast_std']:.0f} → Chiarezza {'+' + str(a['clarity']) if a['clarity'] else 'invariata'}")
            notes.append(f"Saturazione media: {a['mean_sat']:.0f}/255 → {'leggero boost' if a['saturation'] > 1 else 'invariata'}")
            notes.append(f"Nitidezza/rumore rilevati → denoise {a['denoise_strength']}%, nitidezza {a['sharpen_amount']}%")
            for n in notes:
                st.write(f"- {n}")
            st.caption("Le impostazioni sono già state applicate agli slider della sidebar — puoi affinarle liberamente.")


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
    live_out = process_image(ref_img, settings, color_settings, pro_settings, retouch_settings, title_settings, seed=0)

    lc1, lc2 = st.columns(2)
    with lc1:
        st.image(ref_img, caption=f"originale — {ref_img.size[0]}x{ref_img.size[1]} — {ref_file.name}", use_container_width=True)
    with lc2:
        st.image(live_out, caption=f"anteprima — {live_out.size[0]}x{live_out.size[1]}", use_container_width=True)

    with st.expander(":: istogramma / histogram"):
        hist_data = compute_rgb_histogram(live_out)
        st.line_chart(hist_data, color=["#FF4B4B", "#3DD56D", "#3D9DF3"])

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
        ext_map = {"JPEG": "jpg", "PNG": "png", "TIFF": "tiff"}
        ext = ext_map[settings.output_format]
        used_names = set()

        for i, uf in enumerate(uploaded_files):
            img = Image.open(uf)
            out_img = process_image(img, settings, color_settings, pro_settings, retouch_settings, title_settings, seed=i)
            out_bytes = image_to_bytes(out_img, settings.output_format, settings.jpeg_quality)
            custom_name = custom_names.get(uf.name, "")
            raw_name = build_output_name(uf.name, settings.rename_base, custom_name, i + 1, settings.target_w, settings.target_h, ext)
            out_name = dedupe_filename(raw_name, used_names)

            # miniature leggere per l'anteprima: l'output a piena qualità resta
            # comunque disponibile in out_bytes per il download singolo/ZIP.
            # Le dimensioni vere si salvano a parte: la miniatura NON le rappresenta.
            orig_size = img.size
            out_size = out_img.size
            orig_thumb = make_preview_thumbnail(img)
            out_thumb = make_preview_thumbnail(out_img)
            st.session_state.processed.append((out_name, out_bytes, orig_thumb, out_thumb, orig_size, out_size))
            progress.progress((i + 1) / len(uploaded_files))

        # JPEG è già compresso: ZIP_STORED evita di sprecare CPU ricomprimendo
        # dati incomprimibili. Per PNG, ZIP_DEFLATED può ancora aiutare un po'.
        zip_compression = zipfile.ZIP_DEFLATED if settings.output_format == "PNG" else zipfile.ZIP_STORED
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zip_compression) as zf:
            for name, data, _, _, _, _ in st.session_state.processed:
                zf.writestr(name, data)
        st.session_state.zip_bytes = zip_buf.getvalue()

    if st.session_state.processed:
        st.divider()
        total = len(st.session_state.processed)
        shown = min(preview_limit, total)
        st.subheader(f":: anteprima / preview ({shown} di {total})")

        for i, (name, data, orig_img, out_img, orig_size, out_size) in enumerate(st.session_state.processed[:preview_limit]):
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                st.image(orig_img, caption=f"originale {orig_size[0]}x{orig_size[1]}", use_container_width=True)
            with c2:
                st.image(out_img, caption=f"output {out_size[0]}x{out_size[1]}", use_container_width=True)
            with c3:
                st.write(name)
                st.download_button(
                    "⬇ scarica singola",
                    data=data,
                    file_name=name,
                    mime={"JPEG": "image/jpeg", "PNG": "image/png", "TIFF": "image/tiff"}[settings.output_format],
                    key=f"dl_{i}_{name}",
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
                file_name=f"{sanitize_filename(zip_name)}_{settings.target_w}x{settings.target_h}.zip",
                mime="application/zip",
                type="primary",
            )
else:
    st.info("Carica una o più immagini per iniziare.")
