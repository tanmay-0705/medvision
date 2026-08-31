# MedVisionAI

**AI-Assisted Medical Imaging Analysis & Explainability**

MedVisionAI is a **Streamlit-based medical imaging prototype** that loads DICOM studies, provides basic image visualization, runs a pretrained AI model for chest X-ray analysis, explains predictions using Grad-CAM, and generates an AI-assisted PDF report.

## Workflow

* **DICOM Viewer** : Loads `.dcm` files and displays image data, metadata, slices/frames, Window Level and Window Width controls, and zoom.
* **AI Analysis** : Uses a pretrained **TorchXRayVision DenseNet121** model to generate pathology confidence scores.
* **Explainable AI** : Uses **Grad-CAM** to highlight regions that influenced the top model prediction.
* **LLM Report Generation** : Uses **Google Gemini** to convert structured AI findings into a readable summary.
* **PDF Report** : Combines study metadata, original image, Grad-CAM, model findings, AI summary and doctor's notes into a downloadable report.
* **Synthetic DICOM Data** : Automatically creates sample X-Ray, CT and MRI DICOM files for testing without requiring a real dataset.

The application is organized into DICOM processing, AI inference, Grad-CAM, LLM reporting, PDF generation and Streamlit UI components.

## How It Works

```text
DICOM Upload
     ↓
DICOM Metadata + Pixel Processing
     ↓
Image Viewer
(Window/Level + Slice + Zoom)
     ↓
Pretrained DenseNet121
     ↓
Pathology Confidence Scores
     ↓
Grad-CAM Explainability
     ↓
Gemini Structured Summary
     ↓
PDF Report
```

The AI model processes the raw image data separately from the display windowing, while Gemini receives **structured model findings and metadata**, not the raw image.

## Tech Stack

**Python · Streamlit · PyTorch · TorchXRayVision · pydicom · OpenCV · NumPy · Grad-CAM · Google Gemini · ReportLab**

## Setup

```bash
pip install -r requirements.txt
```

Create `.env`:

```env
GEMINI_API_KEY=your_api_key_here
```

Run:

```bash
streamlit run medvisionai_all.py
```


