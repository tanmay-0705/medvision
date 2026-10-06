from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import os
import random
import re
import tempfile
import threading
import time
from xml.sax.saxutils import escape as xml_escape

import numpy as np
import streamlit as st
from PIL import Image

# Must be the first Streamlit command.
st.set_page_config(page_title="MedVisionAI", layout="wide", page_icon="🩺")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(APP_DIR, ".env"))
except ImportError:  # python-dotenv is optional (e.g. in the cloud, env vars are set directly)
    pass

import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

import torch
import torchxrayvision as xrv
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

try:
    from google import genai
    from google.genai import errors as gerrors
    from google.genai import types as gtypes

    GENAI_AVAILABLE = True
except Exception:  # package missing -> app still runs, LLM features show a clear message
    GENAI_AVAILABLE = False

RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


# =====================================================================================
# 1. SAMPLE DICOM GENERATOR (synthetic data, written to a temp dir so it works in containers)
# =====================================================================================
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_dicoms")
SAMPLES = {
    "Chest X-Ray Sample1": "chest xray sample1.dcm",
    "Chest X-Ray Sample2": "chest xray sample2.dcm",
    "Chest X-Ray Sample3": "chest xray sample3.dcm",
}


def _make_base_ds(modality, rows, cols, num_frames, patient_name, study_desc, body_part):
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientName = patient_name
    ds.PatientID = "DEMO1"
    ds.PatientSex = "O"
    ds.PatientBirthDate = "19900101"
    ds.Modality = modality
    ds.StudyDate = _dt.date.today().strftime("%Y%m%d")
    ds.StudyDescription = study_desc
    ds.BodyPartExamined = body_part
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.Manufacturer = "MedVisionAI-Synthetic"

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    ds.WindowCenter = 128
    ds.WindowWidth = 256
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 1.0
    if num_frames > 1:
        ds.NumberOfFrames = num_frames
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    return ds


def _synthetic_chest_xray():
    rows, cols = 512, 512
    y, x = np.ogrid[:rows, :cols]
    cy, cx = rows / 2, cols / 2
    lung_l = ((x - cx + 90) ** 2 / 120**2 + (y - cy) ** 2 / 180**2) < 1
    lung_r = ((x - cx - 90) ** 2 / 120**2 + (y - cy) ** 2 / 180**2) < 1
    img = np.full((rows, cols), 180, dtype=np.int32)
    img[lung_l] = 60
    img[lung_r] = 60
    noise = (np.random.randn(rows, cols) * 8).astype(np.int32)
    img = np.clip(img + noise, 0, 255).astype(np.uint16)
    ds = _make_base_ds("CR", rows, cols, 1, "Demo^Chest^XRay", "Chest PA - SYNTHETIC DEMO", "CHEST")
    ds.PixelData = img.tobytes()
    return ds


def _synthetic_ct_series(num_frames=20):
    rows, cols = 256, 256
    y, x = np.ogrid[:rows, :cols]
    cy, cx = rows / 2, cols / 2
    frames = []
    for i in range(num_frames):
        r = 60 + 20 * np.sin(i / 3)
        mask = (x - cx) ** 2 + (y - cy) ** 2 < r**2
        img = np.full((rows, cols), 20, dtype=np.uint16)
        img[mask] = 150 + i * 2
        frames.append(img)
    arr = np.stack(frames)
    ds = _make_base_ds("CT", rows, cols, num_frames, "Abdomen^CT", "Abdomen CT - SYNTHETIC DEMO SERIES", "ABDOMEN")
    ds.PixelData = arr.tobytes()
    return ds


def _synthetic_mri_series(num_frames=15):
    rows, cols = 256, 256
    y, x = np.ogrid[:rows, :cols]
    cy, cx = rows / 2, cols / 2
    frames = []
    for i in range(num_frames):
        mask = (x - cx) ** 2 + (y - cy) ** 2 < 70**2
        img = np.full((rows, cols), 30, dtype=np.uint16)
        img[mask] = 100 + int(30 * np.sin(i / 2))
        blob = ((x - cx - 20) ** 2 / 15**2 + (y - cy + 10 - i) ** 2 / 15**2) < 1
        img[blob] = 220
        frames.append(img)
    arr = np.stack(frames)
    ds = _make_base_ds("MR", rows, cols, num_frames, "Brain^MRI", "Brain MRI - SYNTHETIC DEMO SERIES", "BRAIN")
    ds.PixelData = arr.tobytes()
    return ds


def _save_ds(ds, path):
    # pydicom 3.x uses enforce_file_format; 2.x uses write_like_original=False.
    try:
        ds.save_as(path, enforce_file_format=True)
    except TypeError:
        ds.save_as(path, write_like_original=False)


def ensure_samples_exist() -> bool:
    """Generate the 3 synthetic .dcm files if missing. Never crashes the app."""
    try:
        os.makedirs(SAMPLE_DIR, exist_ok=True)
        makers = {
            "Synthetic Chest X-Ray": _synthetic_chest_xray,
            "Synthetic CT Series": _synthetic_ct_series,
            "Synthetic MRI Series": _synthetic_mri_series,
        }
        for label, fname in SAMPLES.items():
            path = os.path.join(SAMPLE_DIR, fname)
            if not os.path.exists(path):
                _save_ds(makers[label](), path)
        return True
    except Exception as e:
        st.session_state["_sample_error"] = f"{type(e).__name__}: {e}"
        return False


# =====================================================================================
# 2. DICOM READ / WINDOW UTILS
# =====================================================================================
def load_dicom(file_like):
    return pydicom.dcmread(file_like, force=True)


def decode_pixel_array(ds) -> np.ndarray:
    """Decode once at load time and cache -- ds.pixel_array is expensive to recompute."""
    return ds.pixel_array


def count_frames(ds, arr: np.ndarray) -> int:
    """Derive the frame count from the decoded array (more reliable than the header tag)."""
    spp = int(getattr(ds, "SamplesPerPixel", 1))
    if spp == 1:
        return int(arr.shape[0]) if arr.ndim == 3 else 1
    return int(arr.shape[0]) if arr.ndim == 4 else 1


def get_raw_frame(ds, frame_idx: int, arr: np.ndarray, n_frames: int) -> np.ndarray:
    frame = arr[frame_idx] if n_frames > 1 else arr
    spp = int(getattr(ds, "SamplesPerPixel", 1))
    if spp == 3 and frame.ndim == 3:  # RGB -> grayscale
        frame = frame.mean(axis=-1)
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    frame = frame.astype(np.float64) * slope + intercept
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        frame = (frame.max() + frame.min()) - frame  # invert, keep the value range
    return frame


def pixel_range(ds, arr: np.ndarray):
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    a = float(arr.min()) * slope + intercept
    b = float(arr.max()) * slope + intercept
    return min(a, b), max(a, b)


def _first_number(value):
    if value is None:
        return None
    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        value = value[0]
    return float(value)


def default_window(ds, pmin: float, pmax: float):
    try:
        wc = _first_number(getattr(ds, "WindowCenter", None))
        ww = _first_number(getattr(ds, "WindowWidth", None))
    except Exception:
        wc = ww = None
    if wc is None or ww is None or ww <= 0:
        return (pmin + pmax) / 2, max(pmax - pmin, 1.0)
    return wc, ww


def apply_window(frame: np.ndarray, center: float, width: float) -> np.ndarray:
    width = max(width, 1.0)
    lower = center - width / 2
    upper = center + width / 2
    windowed = np.clip(frame, lower, upper)
    windowed = (windowed - lower) / (upper - lower) * 255.0
    return windowed.astype(np.uint8)


def extract_metadata(ds) -> dict:
    fields = [
        "PatientName", "PatientID", "PatientSex", "StudyDate",
        "Modality", "BodyPartExamined", "Rows", "Columns", "PixelSpacing", "Manufacturer",
    ]
    return {f: str(getattr(ds, f, "N/A")) for f in fields}


def to_pil(frame_uint8: np.ndarray, zoom_pct: int = 100) -> Image.Image:
    img = Image.fromarray(frame_uint8)
    if zoom_pct != 100:
        w, h = img.size
        img = img.resize((max(int(w * zoom_pct / 100), 1), max(int(h * zoom_pct / 100), 1)), RESAMPLE)
    return img


# =====================================================================================
# 3. AI INFERENCE: pretrained TorchXRayVision DenseNet121 (+ Grad-CAM)
# =====================================================================================
@st.cache_resource(show_spinner=False)
def get_model():
    """Loaded once per process and shared across reruns/sessions."""
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def get_model_lock():
    # The model (and Grad-CAM hooks on it) is shared across sessions -> serialize access.
    return threading.Lock()


def preprocess_for_model(frame_raw: np.ndarray) -> torch.Tensor:
    """frame_raw = raw (post Rescale) array BEFORE display windowing, so the model sees the real
    signal and not the operator's window/level."""
    img = frame_raw.astype(np.float32)
    img = img - img.min()
    max_val = img.max()
    if max_val > 0:
        img = img / max_val
    img = (img * 2048.0) - 1024.0  # TorchXRayVision expects ~[-1024, 1024]
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return torch.nn.functional.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)


def run_inference(tensor: torch.Tensor) -> dict:
    """Returns {pathology: probability 0-1}, sorted highest first."""
    model = get_model()
    with get_model_lock(), torch.no_grad():
        output = model(tensor)[0]
    probs = output.detach().cpu().numpy()
    findings = {
        name: float(p)
        for name, p in zip(model.pathologies, probs)
        if name and not np.isnan(p)
    }
    return dict(sorted(findings.items(), key=lambda kv: kv[1], reverse=True))


def generate_gradcam_overlay(tensor: torch.Tensor, target_idx: int) -> np.ndarray:
    """RGB uint8 image: input + Grad-CAM heatmap for model.pathologies[target_idx]."""
    model = get_model()
    inp = tensor.clone()
    with get_model_lock():
        cam = GradCAM(model=model, target_layers=[model.features.denseblock4])
        try:
            grayscale_cam = cam(
                input_tensor=inp, targets=[ClassifierOutputTarget(target_idx)], eigen_smooth=False
            )[0]
        finally:  # always release the hooks, otherwise they pile up on the shared model
            try:
                cam.activations_and_grads.release()
            except Exception:
                pass

    base = tensor[0, 0].detach().cpu().numpy()
    base = (base - base.min()) / (base.max() - base.min() + 1e-8)
    base_rgb = np.stack([base] * 3, axis=-1).astype(np.float32)
    return show_cam_on_image(base_rgb, grayscale_cam, use_rgb=True)


# =====================================================================================
# 4. GEMINI LAYER -- one hardened caller (retry/backoff + model fallback) used by all features
# =====================================================================================
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest").strip() or "gemini-flash-latest"
_fallbacks = os.getenv("GEMINI_FALLBACK_MODELS", "gemini-2.5-flash,gemini-2.5-flash-lite")
GEMINI_MODELS = [GEMINI_MODEL] + [
    m.strip() for m in _fallbacks.split(",") if m.strip() and m.strip() != GEMINI_MODEL
]
RETRYABLE_CODES = {429, 500, 502, 503, 504}


class LLMError(Exception):
    """Raised with a user-friendly message when the Gemini call ultimately fails."""


def get_gemini_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
    if not key or key == "your_api_key_here":
        try:
            key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
        except Exception:
            key = ""
    return "" if key == "your_api_key_here" else key


def llm_available() -> bool:
    return GENAI_AVAILABLE and bool(get_gemini_api_key())


@st.cache_resource(show_spinner=False)
def get_client(api_key: str):
    return genai.Client(api_key=api_key)


def _friendly_llm_error(e: Exception) -> str:
    code = getattr(e, "code", None)
    if code in (429,):
        return "Gemini quota/rate limit hit. Wait a minute and try again."
    if code in (500, 502, 503, 504):
        return "Gemini is busy right now (server overloaded). Try again in a bit."
    if code in (400, 401, 403):
        return f"Gemini rejected the request ({code}). Check GEMINI_API_KEY and that the model is allowed for it."
    if code == 404:
        return "None of the configured Gemini models were found. Check GEMINI_MODEL in your .env."
    return f"Gemini request failed ({type(e).__name__}: {str(e)[:200]})"


def call_gemini(contents, system_instruction: str, json_mode: bool = False,
                max_tokens: int = 4096, temperature: float = 0.2, retries: int = 3) -> str:
    """Call Gemini with retry/backoff on transient errors and fallback to the next model on 404.
    Returns the response text, or raises LLMError."""
    if not GENAI_AVAILABLE:
        raise LLMError("The 'google-genai' package is not installed. Run: pip install google-genai")
    api_key = get_gemini_api_key()
    if not api_key:
        raise LLMError("GEMINI_API_KEY is not set.")

    client = get_client(api_key)
    cfg_kwargs = dict(
        system_instruction=system_instruction,
        max_output_tokens=max_tokens,
        temperature=temperature,
    )
    if json_mode:
        cfg_kwargs["response_mime_type"] = "application/json"
    config = gtypes.GenerateContentConfig(**cfg_kwargs)

    last_err: Exception | None = None
    for model in GEMINI_MODELS:
        for attempt in range(retries):
            try:
                response = client.models.generate_content(model=model, contents=contents, config=config)
                text = getattr(response, "text", None)
                if text and text.strip():
                    return text
                last_err = LLMError("Gemini returned an empty response (it may have been blocked or truncated).")
            except gerrors.APIError as e:
                last_err = e
                code = getattr(e, "code", None)
                if code == 404:  # model unavailable for this key -> try next model
                    break
                if code not in RETRYABLE_CODES:  # bad key / bad request -> no point retrying
                    raise LLMError(_friendly_llm_error(e)) from e
            except Exception as e:  # network hiccups etc.
                last_err = e
            if attempt < retries - 1:
                time.sleep(min(2**attempt + random.random(), 8))
    if isinstance(last_err, LLMError):
        raise last_err
    raise LLMError(_friendly_llm_error(last_err or Exception("unknown error")))


def parse_json(text: str):
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        s, e = t.find("{"), t.rfind("}")
        if s != -1 and e > s:
            return json.loads(t[s : e + 1])
        raise


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if str(x).strip()]
    return [str(v)] if str(v).strip() else []


# ---- 4a. Structured report summary (grounded ONLY in DenseNet output) -----------------
SYSTEM_PROMPT = (
    "You are assisting a radiologist by summarizing AI model output. "
    "You are not diagnosing. Never invent a finding, symptom, or "
    "recommendation that is not directly supported by the structured "
    "findings you are given. If confidence is low across the board, say so "
    "plainly instead of manufacturing a confident-sounding summary. "
    "If a modality warning is present, state that the scores are not meaningful. "
    "Respond ONLY with valid JSON, no markdown fences, no preamble, in "
    "exactly this shape: "
    '{"clinical_summary": str, "possible_findings": [str], '
    '"severity": str, "confidence_note": str, "recommended_next_steps": [str]}'
)


def _modality_note(metadata: dict) -> str:
    modality = metadata.get("Modality", "").upper()
    if modality in ("CR", "DX", "DR"):
        return ""
    return (f"The screening model is trained on chest X-rays only; this study is "
            f"{modality or 'of unknown modality'}, so the scores are not clinically meaningful.")


def _build_user_message(findings: dict, metadata: dict, top_n: int = 6) -> str:
    payload = {
        "modality": metadata.get("Modality", "N/A"),
        "body_part": metadata.get("BodyPartExamined", "N/A"),
        "patient_sex": metadata.get("PatientSex", "N/A"),
        "modality_warning": _modality_note(metadata),
        "findings_with_confidence": {k: round(v, 3) for k, v in list(findings.items())[:top_n]},
    }
    return ("Structured model output (this is the ONLY source of truth -- do not "
            "add anything beyond it):\n" + json.dumps(payload, indent=2))


def _fallback_summary(findings: dict, metadata: dict, top_n: int = 6) -> dict:
    top = list(findings.items())[:top_n]
    top_str = ", ".join(f"{name} ({p * 100:.0f}%)" for name, p in top)
    highest_name, highest_p = top[0] if top else ("N/A", 0.0)
    note = _modality_note(metadata)
    return {
        "clinical_summary": (
            f"Automated screen of this {metadata.get('Modality', 'N/A')} study. The highest-scoring model "
            f"output was {highest_name} at {highest_p * 100:.0f}% model confidence. This is a screening "
            f"score, not a diagnosis. {note}"
        ).strip(),
        "possible_findings": [name for name, _ in top],
        "severity": "Not assessed (AI summary unavailable)",
        "confidence_note": f"Top signals: {top_str}" if top else "No findings above threshold",
        "recommended_next_steps": [
            "Clinical correlation with patient history required",
            "Radiologist review of the flagged region before any action",
        ],
    }


def _normalize_summary(parsed, findings: dict) -> dict:
    if not isinstance(parsed, dict):
        raise ValueError("summary JSON was not an object")
    allowed = {k.lower(): k for k in findings}  # hallucination guard
    possible = [allowed[f.lower()] for f in _as_list(parsed.get("possible_findings")) if f.lower() in allowed]
    return {
        "clinical_summary": str(parsed.get("clinical_summary", "")).strip() or "N/A",
        "possible_findings": possible,
        "severity": str(parsed.get("severity", "")).strip() or "N/A",
        "confidence_note": str(parsed.get("confidence_note", "")).strip() or "N/A",
        "recommended_next_steps": _as_list(parsed.get("recommended_next_steps")),
    }


def generate_report(findings: dict, metadata: dict):
    """Returns (summary_dict, error_message_or_None). Always returns a usable summary."""
    if not llm_available():
        return _fallback_summary(findings, metadata), None
    try:
        text = call_gemini(_build_user_message(findings, metadata), SYSTEM_PROMPT,
                           json_mode=True, max_tokens=4096)
        return _normalize_summary(parse_json(text), findings), None
    except LLMError as e:
        return _fallback_summary(findings, metadata), str(e)
    except Exception as e:
        return _fallback_summary(findings, metadata), f"Could not parse Gemini's summary ({type(e).__name__}: {e})"


# ---- 4b. Image-direct visual analysis (separate from the guarded DenseNet findings) ----
VISION_SYSTEM_PROMPT = (
    "You are describing the visual appearance of a medical image for a "
    "clinician's quick reference. You are not diagnosing and you are not "
    "the validated detection model -- your output is a separate, exploratory "
    "visual impression only. Describe only what is visually observable "
    "(positioning/orientation, symmetry, density, notable shadows or opacities, "
    "visible devices/lines if any, image quality) in plain, hedged language. "
    "Never state or imply a diagnosis, never assign a probability or confidence "
    "percentage, and never claim a finding is present with certainty. If a "
    "specific question is asked, answer it only in terms of what is visible, in "
    "hedged language, and say plainly if it cannot be determined from the image. "
    "If image quality, positioning, or modality limits what can be said, say so. "
    "If the image does not look like the stated modality, say so in limitations. "
    "Respond ONLY with valid JSON, no markdown fences, no preamble, in exactly "
    'this shape: {"visual_observations": [str], "answer": str, "limitations": str}. '
    'Use an empty string for "answer" when no question was asked.'
)


def _fallback_vision_description(reason: str) -> dict:
    return {
        "visual_observations": [reason],
        "answer": "",
        "limitations": "No image-direct analysis was run for this image.",
    }


def describe_image_with_llm(pil_img: Image.Image, modality: str = "", body_part: str = "",
                            user_prompt: str = ""):
    """Returns (description_dict, error_message_or_None). Cached per (image, question)."""
    if not llm_available():
        reason = ("google-genai is not installed." if not GENAI_AVAILABLE
                  else "AI visual analysis unavailable -- set GEMINI_API_KEY to enable it.")
        return _fallback_vision_description(reason), None

    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=92)
    image_bytes = buf.getvalue()

    cache = st.session_state.setdefault("llm_cache", {})
    cache_key = hashlib.sha256(image_bytes + b"|" + user_prompt.strip().encode()).hexdigest()
    if cache_key in cache:
        return cache[cache_key], None

    question = user_prompt.strip()
    task = (f"Image modality: {modality or 'unknown'}. Body part: {body_part or 'unspecified'}.\n"
            + (f"Question from the user: {question}" if question
               else "Describe the visual appearance of this image."))
    try:
        text = call_gemini(
            [gtypes.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), task],
            VISION_SYSTEM_PROMPT, json_mode=True, max_tokens=4096,
        )
        parsed = parse_json(text)
        if not isinstance(parsed, dict):
            raise ValueError("response JSON was not an object")
        result = {
            "visual_observations": _as_list(parsed.get("visual_observations")),
            "answer": str(parsed.get("answer", "") or "").strip(),
            "limitations": str(parsed.get("limitations", "") or "").strip(),
        }
        cache[cache_key] = result
        return result, None
    except LLMError as e:
        return _fallback_vision_description("Gemini request failed."), str(e)
    except Exception as e:
        return _fallback_vision_description("Gemini returned an unreadable response."), \
            f"Could not parse Gemini's response ({type(e).__name__}: {e})"


# ---- 4c. Follow-up chat -- scoped strictly to what's already on screen -------------------
CHAT_SYSTEM_PROMPT = (
    "You are answering follow-up questions about a medical imaging screening report "
    "that has already been generated. You may only discuss, clarify, or explain "
    "the structured findings, AI-generated summary, and visual impression given "
    "to you below -- treat them as your only source of truth. Do not introduce "
    "any new finding, diagnosis, probability, or recommendation that isn't "
    "already present in that material. If the user asks something the given "
    "report material does not cover (a new symptom, a treatment question, "
    "anything requiring information you were not given), say plainly that it's "
    "outside what this report covers and suggest they raise it with the "
    "reviewing clinician instead of answering from general knowledge. Keep "
    "answers short and in plain language."
)


def _build_chat_context(findings, metadata, llm_report, vision_description) -> str:
    top = dict(list(findings.items())[:8]) if findings else {}
    payload = {
        "modality": metadata.get("Modality", "N/A"),
        "modality_warning": _modality_note(metadata),
        "findings_with_confidence": {k: round(v, 3) for k, v in top.items()},
        "ai_summary": llm_report or {},
        "ai_visual_impression": vision_description or {},
    }
    return "Report material (the only source of truth for this conversation):\n" + json.dumps(payload, indent=2)


def answer_followup_question(question, findings, metadata, llm_report, vision_description, chat_history):
    """Returns (answer_or_None, error_message_or_None). On failure the caller must NOT store an answer."""
    if not llm_available():
        return None, ("google-genai is not installed." if not GENAI_AVAILABLE
                      else "Follow-up chat is unavailable -- set GEMINI_API_KEY.")
    try:
        context = _build_chat_context(findings, metadata, llm_report, vision_description)
        contents = [gtypes.Content(role="user", parts=[gtypes.Part(text=context)]),
                    gtypes.Content(role="model", parts=[gtypes.Part(text="Understood. I will only use this report material.")])]
        for turn in chat_history:
            role = "user" if turn["role"] == "user" else "model"
            contents.append(gtypes.Content(role=role, parts=[gtypes.Part(text=turn["content"])]))
        contents.append(gtypes.Content(role="user", parts=[gtypes.Part(text=question)]))
        text = call_gemini(contents, CHAT_SYSTEM_PROMPT, json_mode=False, max_tokens=2048, temperature=0.3)
        return text.strip(), None
    except LLMError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Chat failed ({type(e).__name__}: {e})"


# =====================================================================================
# 5. PDF REPORT BUILDER (writes to any file-like object -> safe for multiple users)
# =====================================================================================
def _pil_to_flowable(pil_img: Image.Image, max_width_in: float = 3.2) -> RLImage:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    w, h = pil_img.size
    scale = (max_width_in * inch) / w
    return RLImage(buf, width=w * scale, height=h * scale)


def _para(text, style) -> Paragraph:
    """Paragraph with LLM/user text escaped so '<' or '&' can't break reportlab's mini-markup."""
    return Paragraph(xml_escape(str(text)), style)


def build_report_pdf(output, original_image, gradcam_image, findings, metadata,
                     llm_report, vision_description=None, top_n: int = 8):
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], fontSize=18, spaceAfter=4)
    sub_style = ParagraphStyle("SubX", parent=styles["Normal"], textColor=colors.grey, fontSize=9)
    h2_style = ParagraphStyle("H2X", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6)
    h2_exp_style = ParagraphStyle("H2ExpX", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6,
                                  textColor=colors.HexColor("#b45309"))
    body_style = ParagraphStyle("BodyX", parent=styles["Normal"], fontSize=10, leading=14)
    disclaimer_style = ParagraphStyle("DisclaimerX", parent=styles["Normal"], fontSize=8,
                                      textColor=colors.grey, leading=11)

    doc = SimpleDocTemplate(output, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                            leftMargin=0.6 * inch, rightMargin=0.6 * inch)
    story = [Paragraph("MedVisionAI Report", title_style), Spacer(1, 10),
             HRFlowable(width="100%", color=colors.lightgrey)]

    story.append(Paragraph("Study Information", h2_style))
    meta_rows = [
        ["Modality", metadata.get("Modality", "N/A"), "Body Part", metadata.get("BodyPartExamined", "N/A")],
        ["Patient Sex", metadata.get("PatientSex", "N/A"), "Study Date", metadata.get("StudyDate", "N/A")],
        ["Rows x Cols", f"{metadata.get('Rows', 'N/A')} x {metadata.get('Columns', 'N/A')}",
         "Manufacturer", metadata.get("Manufacturer", "N/A")],
    ]
    meta_table = Table(meta_rows, colWidths=[1.1 * inch, 2.1 * inch, 1.1 * inch, 2.1 * inch])
    meta_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.grey),
        ("TEXTCOLOR", (2, 0), (2, -1), colors.grey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.whitesmoke),
    ]))
    story.append(meta_table)

    story.append(Paragraph("Original Slide & Model Attention (Grad-CAM)", h2_style))
    img_table = Table(
        [[_pil_to_flowable(original_image.convert("RGB")), _pil_to_flowable(gradcam_image)],
         [Paragraph("Original", sub_style), Paragraph("Grad-CAM overlay", sub_style)]],
        colWidths=[3.3 * inch, 3.3 * inch],
    )
    img_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story.append(img_table)

    story.append(Paragraph("Detected Findings (Model Confidence)", h2_style))
    rows = [["Finding", "Confidence"]] + [[name, f"{p * 100:.1f}%"] for name, p in list(findings.items())[:top_n]]
    find_table = Table(rows, colWidths=[4.4 * inch, 1.6 * inch])
    find_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f7f9")]),
    ]))
    story.append(find_table)

    story.append(Paragraph("AI-Generated Summary", h2_style))
    story.append(_para(llm_report.get("clinical_summary", "N/A"), body_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Severity:</b> {xml_escape(str(llm_report.get('severity', 'N/A')))}", body_style))
    story.append(Paragraph(f"<b>Confidence note:</b> {xml_escape(str(llm_report.get('confidence_note', 'N/A')))}", body_style))
    steps = llm_report.get("recommended_next_steps", [])
    if steps:
        story.append(Paragraph("<b>Recommended next steps:</b>", body_style))
        for s in steps:
            story.append(Paragraph(f"&bull; {xml_escape(str(s))}", body_style))

    if vision_description and (vision_description.get("visual_observations") or vision_description.get("answer")):
        story.append(Paragraph("AI Visual Impression", h2_exp_style))
        story.append(Paragraph(
            " ", disclaimer_style))
        story.append(Spacer(1, 4))
        for obs in vision_description.get("visual_observations", []):
            story.append(Paragraph(f"&bull; {xml_escape(str(obs))}", body_style))
        if vision_description.get("answer"):
            story.append(Spacer(1, 4))
            story.append(Paragraph(f"<b>Answer to question:</b> {xml_escape(vision_description['answer'])}", body_style))
        if vision_description.get("limitations"):
            story.append(Spacer(1, 4))
            story.append(Paragraph(f"<b>Limitations:</b> {xml_escape(vision_description['limitations'])}", body_style))

    story.append(Paragraph("Doctor's Notes", h2_style))
    story.append(Table([[""]], colWidths=[6.8 * inch], rowHeights=[70],
                       style=TableStyle([("BOX", (0, 0), (-1, -1), 0.6, colors.grey)])))
    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", color=colors.lightgrey))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Generated by an AI-assisted screening prototype. Not a medical device. Not a diagnosis. "
        "All outputs require review by a qualified clinician.", disclaimer_style))

    doc.build(story)
    return output


# =====================================================================================
# 6. STREAMLIT APP
# =====================================================================================
def inject_css():
    st.markdown("""
    <style>
    .stApp { background-color: #0B1F2A; color: #E4EDEF; }
    section[data-testid="stSidebar"] { background-color: #0A1922; border-right: 1px solid #17323F; }
    .brand { font-weight:700; font-size:1.1rem; color:#E4EDEF; }
    .brand span { color:#02C39A; }
    .badge { color:#8FA3AA; font-size:0.75rem; letter-spacing:0.05em; }
    .hero-label { color:#02C39A; letter-spacing:0.12em; font-size:0.75rem; font-weight:600; }
    .hero-title { font-size:2.6rem; font-weight:800; line-height:1.15; margin:10px 0 16px 0; color:#F4F8F8; }
    .hero-sub { color:#9FB4BA; font-size:1.05rem; max-width:640px; line-height:1.6; }
    .stat-box { border:1px solid #17323F; border-radius:10px; padding:14px 18px; background:#0F2530; }
    .stat-label { color:#8FA3AA; font-size:0.72rem; letter-spacing:0.08em; }
    .stat-value { color:#02C39A; font-weight:700; font-size:1rem; margin-top:4px; }
    .stage-card { border:1px solid #17323F; border-radius:12px; padding:20px; background:#0F2530; height:100%; }
    .stage-num { color:#02C39A; font-size:0.75rem; font-weight:700; }
    .stage-title { font-size:1.15rem; font-weight:700; margin:8px 0; color:#F4F8F8; }
    .stage-body { color:#9FB4BA; font-size:0.9rem; line-height:1.5; }
    .dropbox { border:1.5px dashed #1F4552; border-radius:12px; padding:28px; background:#0F2530; text-align:center; }
    .meta-row { display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid #14262F; font-size:0.85rem; }
    .meta-key { color:#8FA3AA; }
    .meta-val { color:#E4EDEF; }
    div.stButton > button { background-color:#028090; color:white; border:none; border-radius:8px; padding:0.5rem 1.1rem; font-weight:600; }
    div.stButton > button:hover { background-color:#02C39A; color:#0B1F2A; }
    </style>
    """, unsafe_allow_html=True)


inject_css()

# ---- session state ----
_STATE_DEFAULTS = {
    "page": "home", "ds": None, "source_name": None, "pixel_array": None, "n_frames": 1,
    "pix_range": (0.0, 1.0), "default_win": (0.5, 1.0), "study_id": 0,
    "findings": None, "gradcam_img": None, "analyzed_img": None, "analyzed_frame": None,
    "llm_summary": None, "pdf_bytes": None, "vision_description": None, "chat_history": [],
    "llm_errors": {}, "llm_cache": {}, "load_error": None, "_upload_id": None,
}
for _k, _v in _STATE_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v.copy() if isinstance(_v, (dict, list)) else _v


def go(page):
    st.session_state.page = page


def set_llm_error(key, message):
    errs = dict(st.session_state.llm_errors)
    if message:
        errs[key] = message
    else:
        errs.pop(key, None)
    st.session_state.llm_errors = errs


def load_into_state(file_like, name) -> bool:
    try:
        ds = load_dicom(file_like)
        arr = decode_pixel_array(ds)
    except Exception as e:
        st.session_state.load_error = (
            f"Could not read '{name}' as a DICOM image ({type(e).__name__}: {e}). "
            "If it is a compressed DICOM, install: pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg"
        )
        return False

    pmin, pmax = pixel_range(ds, arr)
    st.session_state.load_error = None
    st.session_state.ds = ds
    st.session_state.source_name = name
    st.session_state.pixel_array = arr
    st.session_state.n_frames = count_frames(ds, arr)
    st.session_state.pix_range = (pmin, pmax)
    st.session_state.default_win = default_window(ds, pmin, pmax)
    st.session_state.study_id += 1
    st.session_state.findings = None
    st.session_state.gradcam_img = None
    st.session_state.analyzed_img = None
    st.session_state.analyzed_frame = None
    st.session_state.llm_summary = None
    st.session_state.pdf_bytes = None
    st.session_state.vision_description = None
    st.session_state.chat_history = []
    st.session_state.llm_errors = {}
    st.session_state.page = "viewer"
    return True


# ---- top bar ----
top_l, top_m1, top_m2, top_r = st.columns([5, 1, 1, 2])
with top_l:
    st.markdown('<div class="brand">Med<span>Vision</span>AI</div>', unsafe_allow_html=True)
with top_m1:
    st.button("Home", on_click=go, args=("home",))
with top_m2:
    st.button("Viewer", on_click=go, args=("viewer",))
with top_r:
    st.markdown('<div class="badge" style="text-align:right;"></div>', unsafe_allow_html=True)

st.markdown("---")

ensure_samples_exist()


def render_home():
    st.markdown('<div class="hero-label">RADIOLOGY WORKFLOW</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">AI-assisted triage for<br>medical imaging.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">MedVisionAI reads a DICOM study, gives you full window/level '
        'and slice navigation, scores it with a pretrained chest X-ray model, explains the '
        'prediction with Grad-CAM, adds an optional Gemini visual analysis, and packages it into a '
        'PDF for a doctor to review. DICOM reading and the screening model run locally; the optional '
        'Gemini features send the rendered image to Google\'s API.</div>', unsafe_allow_html=True)
    st.write("")
    c1, c2, _ = st.columns([1.3, 1.3, 4])
    with c1:
        st.button("Open the viewer →", on_click=go, args=("viewer",))
    with c2:
        st.markdown('<div style="border:1px solid #1F4552;border-radius:8px;padding:0.5rem 1.1rem;'
                    'text-align:center;color:#9FB4BA;">How it works</div>', unsafe_allow_html=True)

    st.write("")
    st.write("")
    s1, s2, s3 = st.columns(3)
    for col, label, val in zip(
        [s1, s2, s3],
        ["MODALITIES", "PROCESSING", "EXPLAINABILITY"],
        ["CR · CT · MR", "Local model + optional Gemini", "Grad-CAM"],
    ):
        with col:
            st.markdown(f'<div class="stat-box"><div class="stat-label">{label}</div>'
                        f'<div class="stat-value">{val}</div></div>', unsafe_allow_html=True)

    st.write("")
    st.write("")
    st.markdown("### Three stages, one pass")
    st.caption("Each stage is independently inspectable: stop after the viewer, or run the full pipeline.")
    a, b, c = st.columns(3)
    stages = [
        ("01", "Upload", "Drop a DICOM Part 10 file. Headers and pixel data are read locally."),
        ("02", "Analyze", "A pretrained chest X-ray model scores the image, explained with Grad-CAM. "
                          "Optionally, Gemini describes or answers questions about the image."),
        ("03", "Review", "Findings + key slide are packaged into a shareable PDF for a doctor to review."),
    ]
    for col, (num, title, body) in zip([a, b, c], stages):
        with col:
            st.markdown(f'<div class="stage-card"><div class="stage-num">{num}</div>'
                        f'<div class="stage-title">{title}</div>'
                        f'<div class="stage-body">{body}</div></div>', unsafe_allow_html=True)

    st.write("")
    st.write("")
    st.markdown(
        '<div style="border-top:1px solid #17323F; padding-top:18px;">'
        '<b>Load a study and see the pipeline run</b><br>'
        '<span style="color:#9FB4BA;font-size:0.85rem;">3 synthetic sample studies are bundled '
        'in the viewer if you don\'t have a .dcm file to hand. Use synthetic or public data only.</span></div>',
        unsafe_allow_html=True)
    st.button("Start →", on_click=go, args=("viewer",))


def render_viewer():
    left, right = st.columns([1, 3.2])

    # ------------------------------- left column: import + metadata -------------------------------
    with left:
        st.markdown('<div class="badge">LOCAL IMPORT</div>', unsafe_allow_html=True)
        st.markdown("#### Open a DICOM study")
        st.markdown('<div class="dropbox">', unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop a .dcm file or click to browse", type=None,
                                    accept_multiple_files=False, label_visibility="collapsed")
        st.markdown('<div style="color:#8FA3AA;font-size:0.8rem;margin-top:8px;">'
                    'Works with .dcm and extensionless files.</div></div>', unsafe_allow_html=True)

        # Load only when the uploaded file actually changes -- otherwise every rerun
        # (any button click) would wipe findings, LLM output and chat history.
        if uploaded is not None:
            upload_id = (uploaded.name, uploaded.size)
            if st.session_state._upload_id != upload_id:
                st.session_state._upload_id = upload_id
                load_into_state(io.BytesIO(uploaded.getvalue()), uploaded.name)

        st.write("")
        st.markdown('<div class="badge">OR TRY A SAMPLE</div>', unsafe_allow_html=True)
        for label, fname in SAMPLES.items():
            if st.button(label, key=f"sample_{fname}"):
                path = os.path.join(SAMPLE_DIR, fname)
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        load_into_state(io.BytesIO(f.read()), label)
                else:
                    st.session_state.load_error = (
                        f"Sample file could not be created ({st.session_state.get('_sample_error', 'unknown error')})."
                    )

        if st.session_state.load_error:
            st.error(st.session_state.load_error)

        if st.session_state.ds is not None:
            st.write("")
            st.markdown('<div class="badge">METADATA</div>', unsafe_allow_html=True)
            for k, v in extract_metadata(st.session_state.ds).items():
                st.markdown(f'<div class="meta-row"><span class="meta-key">{xml_escape(k)}</span>'
                            f'<span class="meta-val">{xml_escape(v)}</span></div>', unsafe_allow_html=True)

    # ------------------------------- right column: viewer + AI -------------------------------
    with right:
        ds = st.session_state.ds
        if ds is None:
            st.markdown('<div style="display:flex;align-items:center;justify-content:center;height:400px;'
                        'border:1px solid #17323F;border-radius:12px;color:#8FA3AA;">'
                        'No study loaded yet — upload a file or pick a sample on the left.</div>',
                        unsafe_allow_html=True)
            return

        arr = st.session_state.pixel_array
        n_frames = st.session_state.n_frames
        sid = st.session_state.study_id
        pmin, pmax = st.session_state.pix_range
        dwc, dww = st.session_state.default_win
        meta = extract_metadata(ds)
        modality = meta.get("Modality", "").upper()

        st.markdown(f"##### {st.session_state.source_name}")

        lo = int(np.floor(min(pmin, dwc)))
        hi = int(np.ceil(max(pmax, dwc)))
        if hi <= lo:
            hi = lo + 1
        ww_max = max(int(np.ceil((pmax - pmin) * 2)), int(np.ceil(dww)) + 1, 2)

        c1, c2, c3 = st.columns(3)
        with c1:
            wc = st.slider("Window Center", lo, hi, int(np.clip(round(dwc), lo, hi)), key=f"wc_{sid}")
        with c2:
            ww = st.slider("Window Width", 1, ww_max, int(np.clip(round(dww), 1, ww_max)), key=f"ww_{sid}")
        with c3:
            zoom = st.slider("Zoom %", 50, 300, 100, step=10, key=f"zoom_{sid}")

        frame_idx = 0
        if n_frames > 1:
            frame_idx = st.slider(f"Slice (0 – {n_frames - 1})", 0, n_frames - 1, n_frames // 2, key=f"frame_{sid}")

        raw = get_raw_frame(ds, frame_idx, arr, n_frames)
        windowed = apply_window(raw, wc, ww)
        img_clean = Image.fromarray(windowed)  # 100% zoom -- what we send to the LLM / PDF
        img_view = to_pil(windowed, zoom)

        st.image(img_view)
        st.caption(f"{modality or 'N/A'} · {ds.Rows}x{ds.Columns} · frame {frame_idx + 1}/{n_frames} · "
                   f"WW/WL {ww}/{wc} · zoom {zoom}%")

        if modality not in ("CR", "DX", "DR"):
            st.warning(f"The screening model is trained on chest X-rays only — this study is "
                       f"{modality or 'unknown modality'}. Analysis will still run but the numbers are "
                       f"not meaningful for this modality.", icon="⚠️")

        st.markdown("---")
        st.markdown("##### AI Analysis")

        if not GENAI_AVAILABLE:
            st.warning("The 'google-genai' package is not installed. Run: pip install google-genai")
        elif not get_gemini_api_key():
            st.warning("Gemini is not configured. Set GEMINI_API_KEY in your local .env file or in the "
                       "deployed app's secrets / environment variables.")
        for err_key, label in [("summary", "AI summary"), ("vision", "AI image analysis"), ("chat", "Follow-up chat")]:
            if st.session_state.llm_errors.get(err_key):
                st.error(f"{label}: {st.session_state.llm_errors[err_key]}")

        # ---------- 1. disease screening (DenseNet121 + Grad-CAM + LLM summary) ----------
        st.markdown("**1 · Disease screening (DenseNet121 + Grad-CAM)**")
        if st.button("Run Analysis"):
            analysis_ok = False
            try:
                with st.spinner("Loading model (first run downloads pretrained weights) and scoring..."):
                    model = get_model()
                    tensor = preprocess_for_model(raw)
                    new_findings = run_inference(tensor)
                    if not new_findings:
                        st.error("The model returned no findings, so Grad-CAM and the summary cannot continue.")
                    else:
                        top_idx = model.pathologies.index(next(iter(new_findings)))
                        overlay = Image.fromarray(generate_gradcam_overlay(tensor, top_idx)).resize((448, 448), RESAMPLE)
                        st.session_state.findings = new_findings
                        st.session_state.gradcam_img = overlay
                        st.session_state.analyzed_img = img_clean.copy()
                        st.session_state.analyzed_frame = frame_idx
                        st.session_state.pdf_bytes = None
                        st.session_state.chat_history = []
                        set_llm_error("chat", None)
                        analysis_ok = True
                if analysis_ok:
                    with st.spinner("Writing AI summary..."):
                        report, err = generate_report(new_findings, meta)
                        st.session_state.llm_summary = report
                        set_llm_error("summary", err)
            except Exception as e:
                analysis_ok = False
                st.error(f"Analysis failed: {type(e).__name__}: {e}")
            if analysis_ok:
                st.rerun()

        findings = st.session_state.findings
        if findings:
            if st.session_state.analyzed_frame != frame_idx:
                st.info(f"The findings below were computed for frame {st.session_state.analyzed_frame + 1}. "
                        "Click Run Analysis to score the current frame.")
            fc1, fc2 = st.columns([1.3, 1])
            with fc1:
                st.markdown("**Findings (model confidence)**")
                for name, p in list(findings.items())[:8]:
                    bar_col, val_col = st.columns([4, 1])
                    with bar_col:
                        st.progress(min(max(p, 0.0), 1.0), text=name)
                    with val_col:
                        st.write(f"{p * 100:.1f}%")
            with fc2:
                st.markdown("**Grad-CAM**")
                if st.session_state.gradcam_img is not None:
                    st.image(st.session_state.gradcam_img)
                st.caption("Highlighted region drove the top prediction.")

            sm = st.session_state.llm_summary
            st.markdown("**AI Summary**")
            if sm:
                st.write(sm.get("clinical_summary", ""))
                st.markdown(f"**Severity:** {sm.get('severity', 'N/A')}  \n"
                            f"**Confidence note:** {sm.get('confidence_note', 'N/A')}")
                for s in sm.get("recommended_next_steps", []):
                    st.markdown(f"- {s}")
            if st.button("Generate AI Summary"):
                with st.spinner("Writing AI summary..."):
                    report, err = generate_report(findings, meta)
                    st.session_state.llm_summary = report
                    st.session_state.pdf_bytes = None
                    set_llm_error("summary", err)
                st.rerun()
        else:
            st.caption("Click **Run Analysis** to score this slide.")

        # ---------- 2. Gemini image analysis (independent of DenseNet) ----------
        st.markdown("---")
        st.markdown("**2 · AI image analysis (Gemini)**")
        with st.expander("Describe / ask about this image ", expanded=True):
            st.caption(
                "Sends the image itself to a vision-capable LLM (Gemini) for a plain-language visual "
                "description, or an answer to your question. Unlike the findings above, this is **not** "
                "produced by the DenseNet121 model and is **not** checked by the hallucination guard "
                "-- treat it as an exploratory second opinion only, not a finding."
            )
            custom_q = st.text_input("Ask something about this image",
                                     placeholder="e.g. Does the cardiac silhouette look enlarged?",
                                     key=f"vision_q_{sid}")
            if st.button("Analyze Image with AI", key="describe_image_btn"):
                with st.spinner("Sending image to Gemini..."):
                    desc, err = describe_image_with_llm(
                        img_clean, modality=modality,
                        body_part=meta.get("BodyPartExamined", ""), user_prompt=custom_q,
                    )
                    set_llm_error("vision", err)
                    if err is None:
                        st.session_state.vision_description = desc
                        st.session_state.pdf_bytes = None
                st.rerun()

            vd = st.session_state.vision_description
            if vd:
                for obs in vd.get("visual_observations", []):
                    st.markdown(f"- {obs}")
                if vd.get("answer"):
                    st.markdown(f"**Answer:** {vd['answer']}")
                if vd.get("limitations"):
                    st.caption(f"Limitations: {vd['limitations']}")

        # ---------- 3. PDF report ----------
        st.markdown("---")
        st.markdown("##### Report")
        if findings:
            if st.button("Generate PDF Report"):
                try:
                    with st.spinner("Building PDF..."):
                        report = st.session_state.llm_summary
                        if report is None:
                            report, err = generate_report(findings, meta)
                            st.session_state.llm_summary = report
                            set_llm_error("summary", err)
                        buf = io.BytesIO()
                        build_report_pdf(
                            buf,
                            original_image=st.session_state.analyzed_img or img_clean,
                            gradcam_image=st.session_state.gradcam_img,
                            findings=findings, metadata=meta, llm_report=report,
                            vision_description=st.session_state.vision_description,
                        )
                        st.session_state.pdf_bytes = buf.getvalue()
                except Exception as e:
                    st.error(f"PDF generation failed: {type(e).__name__}: {e}")

            if st.session_state.pdf_bytes:
                st.success("Report ready")
                st.download_button("Download PDF Report", data=st.session_state.pdf_bytes,
                                   file_name="medvisionai_report.pdf", mime="application/pdf")
        else:
            st.caption("Run the analysis first to unlock the PDF report.")

        # ---------- 4. follow-up chat ----------
        if findings or st.session_state.vision_description:
            st.markdown("---")
            st.markdown("##### Ask About This Report")
            st.caption("Answers are scoped to the findings, summary, and visual impression.")
            for turn in st.session_state.chat_history:
                with st.chat_message(turn["role"]):
                    st.write(turn["content"])

            user_question = st.chat_input("Ask a question about this report...")
            if user_question:
                answer, err = answer_followup_question(
                    user_question, findings or {}, meta, st.session_state.llm_summary,
                    st.session_state.vision_description, st.session_state.chat_history,
                )
                if answer is not None:  # only successful turns go into the history
                    st.session_state.chat_history.append({"role": "user", "content": user_question})
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    set_llm_error("chat", None)
                else:
                    set_llm_error("chat", err)
                st.rerun()


if st.session_state.page == "home":
    render_home()
else:
    render_viewer()
