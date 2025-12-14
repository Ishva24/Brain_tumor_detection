# Brain Tumor Classification & Severity Prediction System - Design Specification & Implementation Plan

## 1. Executive Summary
This document outlines the architecture, design, and implementation features for a Senior Medical AI UX Architect-grade web interface. The system aims to provide accessible, explainable, and actionable insights for clinicians and patients/researchers based on Brain MRI scans.

## 2. Technical Architecture

### 2.1 Technology Stack
- **Frontend**: Next.js 14 (App Router)
  - **Styling**: Tailwind CSS + ShadCN UI (for medical-grade cleanliness)
  - **State Management**: React Context / Zustand
  - **Visualizations**: Recharts (Charts), D3.js (Complex interaction), Canvas API (Heatmaps)
  - **Maps**: Leaflet / Mapbox (for hospital search)
- **Backend**: FastAPI (Python)
  - High-performance async API
  - Integration with TensorFlow/Keras for model inference
  - Integration with LLM (Gemini/OpenAI) via LangChain or direct API for explanations
- **AI Core**:
  - Existing Models: Baseline CNN, VGG19, ResNet50 (loaded from `brain_tumor_classifier.py`)
  - Integration Wrapper: `inference.py` extended for API usage
  - XAI: Grad-CAM implementation for heatmap generation

### 2.2 System Diagram
[Client (Next.js)] <--> [FastAPI Gateway]
                            |
           -----------------------------------------
           |                 |                     |
    [Inference Engine]   [LLM Service]    [Location Service]
           |                 |
    [TF Models]        [Explanation Prompts]

## 3. User Interface Design & Wireframes

### 3.1 Global Elements
- **Navigation Bar**: Clean, sticky. Logo, "Upload", "History", "Resources", "Settings".
- **Theme**: "Medical Calm" palette.
  - Primary: Deep Teal (#0F766E) - Trust, Health
  - Secondary: Soft Blue (#E0F2FE) - Backgrounds
  - Accent (Severity):
    - Low: Emerald Green (#10B981)
    - Moderate: Amber (#F59E0B)
    - High: Rose Red (#F43F5E)
- **Accessibility**: ARIA labels, specific contrast ratios, keyboard navigation support.

### 3.2 Landing / Overview Dashboard
- **Hero**: "Advanced AI for Neurological Health".
- **Disclaimer**: Prominent banner: "This tool is for educational/research support only. Not a definitive medical diagnosis."
- **Stats Widget**: Live indicators of "Models Active: 4", "System Status: Online".
- **Visual**: Rotating 3D Brain wireframe or clean medical illustration.

### 3.3 MRI Upload & Analysis Panel
- **Upload Zone**: Large drop zone supporting `.jpg`, `.png`, `.dcm`.
  - *Interaction*: Drag file -> Visual validation check -> Preview.
- **Metadata Card**: Extract and show: "Resolution: 224x224", "Modality: T1-Weighted" (simulated if JPG).
- **CTA**: "Analyze Scan" button with a sleek circular progress indicator.

### 3.4 Prediction Results Panel
- **Split View Layout**:
  - **Left**: Original Scan with "Toggle Heatmap" switch.
  - **Right**: Results Card.
    - **Top**: Tumor Type (e.g., "Meningioma") with Confidence Badge (e.g., "98%").
    - **Severity**: "Moderate Risk" gauge meter.
    - **Distribution**: Horizontal Bar chart showing prob. of all 4 classes.
- **Dynamic Content**:
  - *If Severity=High*: "Urgent Attention Advised" banner appears.
  - *If Severity=Low*: "Routine Monitoring" text.

### 3.5 Explainable AI (XAI) Section
- **Visual**: Overlay Grad-CAM heatmap on the MRI.
  - Controls: Opacity slider, "Isolate Tumor Region" toggle.
- **Textual**: "Why this prediction?"
  - **LLM Generated**: "The model focused on the hyperintense region in the left frontal lobe, which is characteristic of Meningioma structures..."

### 3.6 Medical Knowledge Hub (Context-Aware)
- **Tabs**: "Symptoms", "Causes", "Treatments".
- Content dynamically loads based on prediction (e.g., if "Pituitary", show Pituitary-specific info).
- **Source Citations**: Small footnotes linking to medical journals.

### 3.7 Personalized Real-Time Recommendations
- **Map View**: Integrated map centered on user's IP location.
  - Pins: "City Neuro Hospital", "State Cancer Center".
- **List View**: "Nearest Specialists", sorted by distance.
- **Emergency**: "Helpline: 102" (localized).

### 3.8 Visual Analytics
- **Patient Journey**: Timeline chart (Mock data) showing "Typical Recovery Time".
- **Risk Analysis**: Radar chart comparing "Genetic Factors" vs "Environmental" (if history provided).

## 4. Component Breakdown (Frontend)

1.  `Layout.tsx`: Main wrapper, sidebar/navbar.
2.  `HeroSection.tsx`: Landing page visuals.
3.  `FileUploader.tsx`: Drag-and-drop logic.
4.  `AnalysisView.tsx`: Orchestrator for results.
5.  `MRIViewer.tsx`: Canvas-based image viewer with heatmap overlay.
6.  `ResultCard.tsx`: Displays prediction & confidence.
7.  `SeverityGauge.tsx`: D3/Recharts semi-circle gauge.
8.  `ExplainabilityPanel.tsx`: Textual & Visual XAI controls.
9.  `HospitalMap.tsx`: Map integration.
10. `ReportGenerator.tsx`: PDF generation logic.

## 5. API Design (Sample JSON Responses)

**POST /api/predict**
```json
{
  "prediction": {
    "class": "Meningioma",
    "confidence": 0.94,
    "severity": "Moderate",
    "probabilities": {
      "glioma": 0.02,
      "meningioma": 0.94,
      "notumor": 0.01,
      "pituitary": 0.03
    }
  },
  "explanation": {
    "heatmap_b64": "data:image/png;base64...",
    "text_reasoning": "The model identified a well-defined extra-axial mass..."
  }
}
```

**GET /api/recommendations?lat=...&long=...**
```json
{
  "hospitals": [
    {
      "name": "Central Neuro Institute",
      "distance": "2.4 km",
      "type": "Specialty"
    }
  ]
}
```

## 6. LLM Medical Prompts
*To be used by the Backend LLM Service*

**Prompt for Patient Explanation:**
"You are a compassionate medical expert. The AI has detected a [PREDICTED_CLASS] with [CONFIDENCE]% confidence. Explain what this means in simple terms for a patient. Do not give a diagnosis, but explain the typical characteristics of this finding. Keep it under 100 words."

**Prompt for Clinical Explainer:**
"Analyze the attached Grad-CAM heatmap regions. Explain the radiological features typical of [PREDICTED_CLASS] that might reside in these high-activation areas. Use medical terminology suitable for a junior doctor."

## 7. Future Roadmap
1.  **DICOM Integration**: Full native DICOM parsing (windowing/leveling) in browser.
2.  **Longitudinal Tracking**: User accounts to track tumor growth over time.
3.  **Federated Learning**: Allow hospitals to train on local data without privacy breach.
