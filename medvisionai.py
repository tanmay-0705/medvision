import os
from click import prompt
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


import io
import json
import tempfile
import datetime as _dt

import numpy as np
import cv2
import streamlit as st
from PIL import Image


import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

import torch
import torchvision
import torchxrayvision as xrv
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, HRFlowable,
)



# SAMPLE DICOM GENERATOR  
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
    lung_l = ((x - cx + 90) ** 2 / 120 ** 2 + (y - cy) ** 2 / 180 ** 2) < 1
    lung_r = ((x - cx - 90) ** 2 / 120 ** 2 + (y - cy) ** 2 / 180 ** 2) < 1
    img = np.full((rows, cols), 180, dtype=np.int32)
    img[lung_l] = 60
    img[lung_r] = 60
    noise = (np.random.randn(rows, cols) * 8).astype(np.int32)
    img = np.clip(img + noise, 0, 255).astype(np.uint16)
    ds = _make_base_ds("CR", rows, cols, 1, "Demo^Chest^XRay",
                        "Chest PA ", "CHEST")
    ds.PixelData = img.tobytes()
    return ds


def _synthetic_ct_series(num_frames=20):
    rows, cols = 256, 256
    y, x = np.ogrid[:rows, :cols]
    cy, cx = rows / 2, cols / 2
    frames = []
    for i in range(num_frames):
        r = 60 + 20 * np.sin(i / 3)
        mask = (x - cx) ** 2 + (y - cy) ** 2 < r ** 2
        img = np.full((rows, cols), 20, dtype=np.uint16)
        img[mask] = 150 + i * 2
        frames.append(img)
    arr = np.stack(frames)
    ds = _make_base_ds("CT", rows, cols, num_frames, "Abdomen^CT",
                        "Abdomen CT - SYNTHETIC DEMO SERIES ", "ABDOMEN")
    ds.PixelData = arr.tobytes()
    return ds


def _synthetic_mri_series(num_frames=15):
    rows, cols = 256, 256
    y, x = np.ogrid[:rows, :cols]
    cy, cx = rows / 2, cols / 2
    frames = []
    for i in range(num_frames):
        mask = (x - cx) ** 2 + (y - cy) ** 2 < 70 ** 2
        img = np.full((rows, cols), 30, dtype=np.uint16)
        img[mask] = 100 + int(30 * np.sin(i / 2))
        blob = ((x - cx - 20) ** 2 / 15 ** 2 + (y - cy + 10 - i) ** 2 / 15 ** 2) < 1
        img[blob] = 220
        frames.append(img)
    arr = np.stack(frames)
    ds = _make_base_ds("MR", rows, cols, num_frames, "Brain^MRI",
                        "Brain MRI - SYNTHETIC  SERIES (", "BRAIN")
    ds.PixelData = arr.tobytes()
    return ds


def ensure_samples_exist():
    """Generate the 3 sample .dcm files on first run only (skips if already there)."""
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    paths = {name: os.path.join(SAMPLE_DIR, fname) for name, fname in SAMPLES.items()}
    if not os.path.exists(paths["Chest X-Ray Sample1"]):
        _synthetic_chest_xray().save_as(paths["Chest X-Ray Sample1"], write_like_original=False)
    if not os.path.exists(paths["Chest X-Ray Sample2"]):
        _synthetic_ct_series().save_as(paths["Chest X-Ray Sample2"], write_like_original=False)
    if not os.path.exists(paths["Chest X-Ray Sample3"]):
        _synthetic_mri_series().save_as(paths["Chest X-Ray Sample3"], write_like_original=False)



#DICOM READ / WINDOW UTILS  

def load_dicom(file_like):
    return pydicom.dcmread(file_like, force=True)


def get_num_frames(ds) -> int:
    return int(getattr(ds, "NumberOfFrames", 1))


def decode_pixel_array(ds) -> np.ndarray:
    """Decode the full pixel array ONCE (call at load time, cache the result --
    ds.pixel_array can be expensive to recompute on every slider rerun)."""
    return ds.pixel_array


def get_raw_frame(ds, frame_idx: int = 0, cached_array: np.ndarray = None) -> np.ndarray:
    arr = cached_array if cached_array is not None else ds.pixel_array
    if arr.ndim == 3 and get_num_frames(ds) > 1:
        frame = arr[frame_idx]
    else:
        frame = arr
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    return frame.astype(np.float64) * slope + intercept


def default_window(ds):
    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)
    if wc is None or ww is None:
        arr = ds.pixel_array.astype(np.float64)
        lo, hi = float(arr.min()), float(arr.max())
        return (lo + hi) / 2, max(hi - lo, 1.0)
    wc = float(wc[0]) if isinstance(wc, pydicom.multival.MultiValue) else float(wc)
    ww = float(ww[0]) if isinstance(ww, pydicom.multival.MultiValue) else float(ww)
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
        "PatientName", "PatientID", "PatientSex",
        "Modality", "BodyPartExamined",
        "Rows", "Columns", "PixelSpacing",
        "Manufacturer", 
    ]
    return {f: str(getattr(ds, f, "N/A")) for f in fields}


def to_pil(frame_uint8: np.ndarray, zoom_pct: int = 100) -> Image.Image:
    img = Image.fromarray(frame_uint8)
    if zoom_pct != 100:
        w, h = img.size
        new_w, new_h = max(int(w * zoom_pct / 100), 1), max(int(h * zoom_pct / 100), 1)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


# AI INFERENCE:Pretrained TorchXRayVision DenseNet121 
_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = xrv.models.DenseNet(weights="densenet121-res224-all")
        _MODEL.eval()
    return _MODEL


def preprocess_for_model(frame_raw: np.ndarray) -> torch.Tensor:
    """frame_raw = RAW (post RescaleSlope/Intercept) array, BEFORE display
    windowing -- the model should see the real signal, not an operator's
    chosen window/level."""
    img = frame_raw.astype(np.float32)
    img = img - img.min()
    max_val = img.max()
    if max_val > 0:
        img = img / max_val
    img = (img * 2048.0) - 1024.0   # TorchXRayVision's expected ~[-1024, 1024] range

    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    tensor = torch.nn.functional.interpolate(
        tensor, size=(224, 224), mode="bilinear", align_corners=False
    )
    return tensor


def run_inference(frame_raw: np.ndarray) -> dict:
    """Returns {pathology_name: probability 0-1}, sorted highest first."""
    model = get_model()
    tensor = preprocess_for_model(frame_raw)
    with torch.no_grad():
        output = model(tensor)[0]
    probs = output.detach().numpy()
    findings = {name: float(p) for name, p in zip(model.pathologies, probs) if name}
    return dict(sorted(findings.items(), key=lambda kv: kv[1], reverse=True))

# 4. GRAD-CAM EXPLAINABILITY 

def generate_gradcam_overlay(frame_raw: np.ndarray, target_pathology_idx: int) -> np.ndarray:
    """Returns an RGB uint8 224x224x3 image: input frame + Grad-CAM heatmap,
    showing what drove the prediction for pathology model.pathologies[idx]."""
    model = get_model()
    tensor = preprocess_for_model(frame_raw)
    tensor.requires_grad_(True)

    target_layers = [model.features.denseblock4]   # last conv block of DenseNet121
    targets = [ClassifierOutputTarget(target_pathology_idx)]

    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=tensor, targets=targets, eigen_smooth=False)[0]

    base = tensor[0, 0].detach().numpy()
    base = (base - base.min()) / (base.max() - base.min() + 1e-8)
    base_rgb = np.stack([base] * 3, axis=-1)

    return show_cam_on_image(base_rgb, grayscale_cam, use_rgb=True)

# 5. LLM REPORT SUMMARY  (Day 3)

SYSTEM_PROMPT = (
    "You are assisting a radiologist by summarizing AI model output. "
    "You are not diagnosing. Never invent a finding, symptom, or "
    "recommendation that is not directly supported by the structured "
    "findings you are given. If confidence is low across the board, say so "
    "plainly instead of manufacturing a confident-sounding summary. "
    "Respond ONLY with valid JSON, no markdown fences, no preamble, in "
    "exactly this shape: "
    '{"clinical_summary": str, "possible_findings": [str], '
    '"severity": str, "confidence_note": str, "recommended_next_steps": [str]}'
)


def _build_user_message(findings: dict, metadata: dict, top_n: int = 6) -> str:
    top_findings = dict(list(findings.items())[:top_n])
    payload = {
        "modality": metadata.get("Modality", "N/A"),
        "body_part": metadata.get("BodyPartExamined", "N/A"),
        "patient_sex": metadata.get("PatientSex", "N/A"),
        "findings_with_confidence": {k: round(v, 3) for k, v in top_findings.items()},
    }
    return ("Structured model output (this is the ONLY source of truth -- do not "
            "add anything beyond it):\n" + json.dumps(payload, indent=2))


def _fallback_summary(findings: dict, metadata: dict, top_n: int = 6) -> dict:
    top = list(findings.items())[:top_n]
    top_str = ", ".join(f"{name} ({p*100:.0f}%)" for name, p in top)
    highest_name, highest_p = top[0] if top else ("N/A", 0.0)
    return {
        "clinical_summary": (
            f"Automated screen of this {metadata.get('Modality','N/A')} study flagged "
            f"{len(top)} findings above baseline, the highest being {highest_name} "
            f"at {highest_p*100:.0f}% model confidence." 
        ),
        "possible_findings": [name for name, _ in top],
        '''"severity": "Not assessed -- set GEMINI_API_KEY for a proper severity read",'''
        "confidence_note": f"Top signal: {top_str}" if top else "No findings above threshold",
        "recommended_next_steps": [
            "Clinical correlation with patient history required",
            "Radiologist review of the flagged region before any action",
        ],
    }


def _guard_against_hallucination(parsed: dict, findings: dict) -> dict:
    allowed = set(findings.keys())
    parsed["possible_findings"] = [f for f in parsed.get("possible_findings", []) if f in allowed]
    return parsed


def generate_report(findings: dict, metadata: dict) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return _fallback_summary(findings, metadata)

    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=_build_user_message(findings, metadata),
            config={
                "system_instruction": SYSTEM_PROMPT,
                "response_mime_type": "application/json",
                "max_output_tokens": 2048,
            },
        )

        raw_text = response.text

        parsed = json.loads(raw_text)

        return _guard_against_hallucination(parsed, findings)

    except Exception as e:
        st.session_state["_gemini_error"] = str(e)
        st.session_state["_gemini_raw"] = locals().get("raw_text", "N/A")
        return _fallback_summary(findings, metadata)

    '''except Exception as e:
        print(f"Gemini API error: {e}")
        return _fallback_summary(findings, metadata)'''



# PDF REPORT BUILDER 
def _pil_to_flowable(pil_img: Image.Image, max_width_in: float = 3.2) -> RLImage:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    w, h = pil_img.size
    scale = (max_width_in * inch) / w
    return RLImage(buf, width=w * scale, height=h * scale)


def build_report_pdf(output_path, original_image, gradcam_image, findings, metadata,
                      llm_report, top_n: int = 8) -> str:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], fontSize=18, spaceAfter=4)
    sub_style = ParagraphStyle("SubX", parent=styles["Normal"], textColor=colors.grey, fontSize=9)
    h2_style = ParagraphStyle("H2X", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6)
    body_style = ParagraphStyle("BodyX", parent=styles["Normal"], fontSize=10, leading=14)
    disclaimer_style = ParagraphStyle("DisclaimerX", parent=styles["Normal"], fontSize=8,
                                       textColor=colors.grey, leading=11)

    doc = SimpleDocTemplate(output_path, pagesize=letter, topMargin=0.6 * inch,
                             bottomMargin=0.6 * inch, leftMargin=0.6 * inch, rightMargin=0.6 * inch)
    story = []

    story.append(Paragraph("MedVisionAI Report", title_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", color=colors.lightgrey))

    story.append(Paragraph("Study Information", h2_style))
    meta_rows = [
        ["Modality", metadata.get("Modality", "N/A"), "Body Part", metadata.get("BodyPartExamined", "N/A")],
        ["Patient Sex", metadata.get("PatientSex", "N/A"), "Study Date", metadata.get("StudyDate", "N/A")],
        ["Rows x Cols", f"{metadata.get('Rows','N/A')} x {metadata.get('Columns','N/A')}",
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
        [[_pil_to_flowable(original_image), _pil_to_flowable(gradcam_image)],
         [Paragraph("Original", sub_style),
          Paragraph("Grad-CAM overlay", sub_style)]],
        colWidths=[3.3 * inch, 3.3 * inch],
    )
    img_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story.append(img_table)

    story.append(Paragraph("Detected Findings (Model Confidence)", h2_style))
    top_findings = list(findings.items())[:top_n]
    rows = [["Finding", "Confidence"]] + [[name, f"{p*100:.1f}%"] for name, p in top_findings]
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
    story.append(Paragraph(llm_report.get("clinical_summary", "N/A"), body_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Severity:</b> {llm_report.get('severity', 'N/A')}", body_style))
    story.append(Paragraph(f"<b>Confidence note:</b> {llm_report.get('confidence_note', 'N/A')}", body_style))
    steps = llm_report.get("recommended_next_steps", [])
    if steps:
        story.append(Paragraph("<b>Recommended next steps:</b>", body_style))
        for s in steps:
            story.append(Paragraph(f"&bull; {s}", body_style))

    story.append(Paragraph("Doctor's Notes", h2_style))
    story.append(Table([[""]], colWidths=[6.8 * inch], rowHeights=[70],
                        style=TableStyle([("BOX", (0, 0), (-1, -1), 0.6, colors.grey)])))

    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", color=colors.lightgrey))
    story.append(Spacer(1, 6))
    

    doc.build(story)
    return output_path



#STREAMLIT APP

st.set_page_config(page_title="MedVisionAI", layout="wide", page_icon="//")
ensure_samples_exist()


def inject_css():
    st.markdown("""
    <style>
    .stApp { background-color: #0a0e14; color: #e6edf3; }
    section[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #1c2333; }
    .brand { font-weight:700; font-size:1.1rem; color:#e6edf3; }
    .brand span { color:#3b82f6; }
    .badge { color:#7d8590; font-size:0.75rem; letter-spacing:0.05em; }
    .hero-label { color:#3b82f6; letter-spacing:0.12em; font-size:0.75rem; font-weight:600; }
    .hero-title { font-size:2.6rem; font-weight:800; line-height:1.15; margin:10px 0 16px 0; }
    .hero-sub { color:#9aa4b2; font-size:1.05rem; max-width:640px; line-height:1.6; }
    .stat-box { border:1px solid #1c2333; border-radius:10px; padding:14px 18px; background:#0d1117; }
    .stat-label { color:#7d8590; font-size:0.72rem; letter-spacing:0.08em; }
    .stat-value { color:#e6edf3; font-weight:700; font-size:1rem; margin-top:4px; }
    .stage-card { border:1px solid #1c2333; border-radius:12px; padding:20px; background:#0d1117; height:100%; }
    .stage-num { color:#3b82f6; font-size:0.75rem; font-weight:700; }
    .stage-title { font-size:1.15rem; font-weight:700; margin:8px 0; }
    .stage-body { color:#9aa4b2; font-size:0.9rem; line-height:1.5; }
    .dropbox { border:1.5px dashed #2a3347; border-radius:12px; padding:28px; background:#0d1117; text-align:center; }
    .meta-row { display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid #161c28; font-size:0.85rem; }
    .meta-key { color:#7d8590; }
    .meta-val { color:#e6edf3; }
    div.stButton > button { background-color:#3b82f6; color:white; border:none; border-radius:8px; padding:0.5rem 1.1rem; font-weight:600; }
    div.stButton > button:hover { background-color:#2563eb; color:white; }
    </style>
    """, unsafe_allow_html=True)


inject_css()

# ---- session state ----
for key, default in [
    ("page", "home"), ("ds", None), ("source_name", None), ("pixel_array", None),
    ("findings", None), ("gradcam_img", None), ("llm_summary", None), ("_pdf_bytes", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def go(page):
    st.session_state.page = page


def load_into_state(file_like, name):
    ds = load_dicom(file_like)
    st.session_state.ds = ds
    st.session_state.source_name = name
    st.session_state.pixel_array = decode_pixel_array(ds)
    st.session_state.findings = None
    st.session_state.gradcam_img = None
    st.session_state.llm_summary = None
    st.session_state._pdf_bytes = None
    st.session_state.page = "viewer"


@st.cache_resource(show_spinner=False)
def cached_model():
    # Survives Streamlit reruns and is shared across sessions in this process
    # -- the model only loads once no matter how many times the script re-runs.
    return get_model()


# ---- top bar ----
top_l, top_m1, top_m2, top_r = st.columns([5, 1, 1, 2])
with top_l:
    st.markdown('<div class="brand">Med<span>Vision</span>AI</div>', unsafe_allow_html=True)
with top_m1:
    if st.button("Home", use_container_width=True):
        go("home")
with top_m2:
    if st.button("Viewer", use_container_width=True):
        go("viewer")
with top_r:
    st.markdown('<div class="badge" style="text-align:right;"></div>', unsafe_allow_html=True)

st.markdown("---")


def render_home():
    st.markdown('<div class="hero-label">RADIOLOGY WORKFLOW </div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">AI-assisted triage for<br>medical imaging.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">MedVisionAI reads a DICOM study, gives you full window/level '
        'and slice navigation, scores it with a pretrained chest X-ray model, explains the '
        'prediction with Grad-CAM, and packages it into a PDF for a doctor to review. '
        'Everything runs locally.</div>', unsafe_allow_html=True)
    st.write("")
    c1, c2, _ = st.columns([1.3, 1.3, 4])
    with c1:
        if st.button("Open the viewer →", use_container_width=True):
            go("viewer")
    with c2:
        st.markdown('<div style="border:1px solid #2a3347;border-radius:8px;padding:0.5rem 1.1rem;'
                    'text-align:center;color:#9aa4b2;">How it works</div>', unsafe_allow_html=True)

    st.write("")
    st.write("")
    s1, s2, s3 = st.columns(3)
    for col, label, val in zip(
        [s1, s2, s3],
        ["MODALITIES", "PROCESSING", "EXPLAINABILITY"],
        ["CR · CT · MR", "Local / Python", "Grad-CAM"],
    ):
        with col:
            st.markdown(f'<div class="stat-box"><div class="stat-label">{label}</div>'
                        f'<div class="stat-value">{val}</div></div>', unsafe_allow_html=True)

    st.write("")
    st.write("")
    st.markdown("### Three stages, one pass")
    st.caption("Each stage is independently inspectable, stop after the viewer, or run the full pipeline.")
    a, b, c = st.columns(3)
    stages = [
        ("01", "Upload", "Drop a DICOM Part 10 file. Headers and pixel data are read locally; your study never leaves this machine."),
        ("02", "Analyze", "A pretrained chest X-ray model scores the image for common abnormalities, explained with Grad-CAM."),
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
        '<div style="border-top:1px solid #1c2333; padding-top:18px;">'
        '<b>Load a study and see the pipeline run</b><br>'
        '<span style="color:#9aa4b2;font-size:0.85rem;">3 synthetic sample studies are bundled '
        'below if you don\'t have a .dcm file to hand.</span></div>', unsafe_allow_html=True)
    if st.button("Start →"):
        go("viewer")


def render_viewer():
    left, right = st.columns([1, 3.2])

    with left:
        st.markdown('<div class="badge">LOCAL IMPORT</div>', unsafe_allow_html=True)
        st.markdown("#### Open a DICOM study")
        st.markdown('<div class="dropbox">', unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop a .dcm file or click to browse", type=None,
                                     accept_multiple_files=False, label_visibility="collapsed")
        st.markdown('<div style="color:#7d8590;font-size:0.8rem;margin-top:8px;">'
                    'Works with .dcm and extensionless files. Nothing leaves this machine.</div></div>',
                    unsafe_allow_html=True)

        if uploaded is not None:
            load_into_state(io.BytesIO(uploaded.getvalue()), uploaded.name)

        st.write("")
        st.markdown('<div class="badge">OR TRY A SAMPLE</div>', unsafe_allow_html=True)
        for label, fname in SAMPLES.items():
            if st.button(label, use_container_width=True, key=f"sample_{fname}"):
                with open(os.path.join(SAMPLE_DIR, fname), "rb") as f:
                    load_into_state(io.BytesIO(f.read()), label)

        if st.session_state.ds is not None:
            st.write("")
            st.markdown('<div class="badge">METADATA</div>', unsafe_allow_html=True)
            for k, v in extract_metadata(st.session_state.ds).items():
                st.markdown(f'<div class="meta-row"><span class="meta-key">{k}</span>'
                            f'<span class="meta-val">{v}</span></div>', unsafe_allow_html=True)

    with right:
        ds = st.session_state.ds
        if ds is None:
            st.markdown('<div style="display:flex;align-items:center;justify-content:center;height:400px;'
                        'border:1px solid #1c2333;border-radius:12px;color:#7d8590;">'
                        'No study loaded yet — upload a file or pick a sample on the left.</div>',
                        unsafe_allow_html=True)
            return

        st.markdown(f"##### {st.session_state.source_name}")
        n_frames = get_num_frames(ds)

        c1, c2, c3 = st.columns(3)
        default_wc, default_ww = default_window(ds)
        with c1:
            wc = st.slider("Window Center", -1000, 3000, int(default_wc))
        with c2:
            ww = st.slider("Window Width", 1, 4000, int(default_ww))
        with c3:
            zoom = st.slider("Zoom %", 50, 300, 100, step=10)

        frame_idx = st.slider(f"Slice (1 / {n_frames})", 0, n_frames - 1, n_frames // 2) if n_frames > 1 else 0

        raw = get_raw_frame(ds, frame_idx, cached_array=st.session_state.pixel_array)
        windowed = apply_window(raw, wc, ww)
        img = to_pil(windowed, zoom)

        st.image(img, use_container_width=False)
        st.caption(f"{getattr(ds, 'Modality', 'N/A')} · {ds.Rows}x{ds.Columns} · "
                  f"frame {frame_idx + 1}/{n_frames} · WW/WL {ww}/{wc} · zoom {zoom}%")

        modality = str(getattr(ds, "Modality", "")).upper()
        if modality not in ("CR", "DX", "DR"):
            st.warning(f"Model is trained on chest X-rays only — this study is "
                      f"{modality or 'unknown modality'}. Analysis will still run but the "
                      f"numbers are not meaningful for this modality.", icon="⚠️")

        st.markdown("---")
        st.markdown("##### AI Analysis")
        run_col, _ = st.columns([1, 3])
        with run_col:
            run_clicked = st.button("Run Analysis", use_container_width=True)

        if run_clicked:
            with st.spinner("Loading model (first run downloads pretrained weights) and scoring..."):
                cached_model()
                findings = run_inference(raw)
                st.session_state.findings = findings
                top_idx = get_model().pathologies.index(list(findings.keys())[0])
                overlay = generate_gradcam_overlay(raw, top_idx)
                st.session_state.gradcam_img = Image.fromarray(overlay)
                st.session_state.llm_summary = None

        if st.session_state.findings:
            fc1, fc2 = st.columns([1.3, 1])
            with fc1:
                st.markdown("**Findings (model confidence)**")
                for name, p in list(st.session_state.findings.items())[:8]:
                    bar_col, val_col = st.columns([4, 1])
                    with bar_col:
                        st.progress(min(max(p, 0.0), 1.0), text=name)
                    with val_col:
                        st.write(f"{p*100:.1f}%")
            with fc2:
                st.markdown("**Grad-CAM**")
                if st.session_state.gradcam_img is not None:
                    st.image(st.session_state.gradcam_img, use_container_width=True)
                st.caption("Highlighted region drove the top prediction above.")

            st.markdown("---")
            st.markdown("##### Report")
            
            pdf_col, _ = st.columns([1, 3])
            with pdf_col:
                pdf_clicked = st.button("Generate PDF Report", use_container_width=True)

            if pdf_clicked:
                with st.spinner("Writing summary and building PDF..."):
                    meta = extract_metadata(ds)
                    report = generate_report(st.session_state.findings, meta)
                    st.session_state.llm_summary = report
                    tmp_path = os.path.join(tempfile.gettempdir(), f"medvisionai_report_{frame_idx}.pdf")
                    build_report_pdf(tmp_path, original_image=img, gradcam_image=st.session_state.gradcam_img,
                                    findings=st.session_state.findings, metadata=meta, llm_report=report)
                    with open(tmp_path, "rb") as f:
                        st.session_state._pdf_bytes = f.read()

            if st.session_state._pdf_bytes:
                st.success("Report ready")
                st.download_button("Download PDF Report", data=st.session_state._pdf_bytes,
                                    file_name="medvisionai_report.pdf", mime="application/pdf")
        else:
            st.caption("Click **Run Analysis** to score this slide and unlock the PDF report step.")


if st.session_state.page == "home":
    render_home()
else:
    render_viewer()
