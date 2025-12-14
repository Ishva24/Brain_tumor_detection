# Deployment Plan - Brain Tumor Detection

This plan outlines the steps to deploy the project for free using **Render** (Backend) and **Vercel** (Frontend).

## User Review Required
> [!WARNING]
> **Memory Usage on Free Tier**: You are deploying a TensorFlow application. Free hosting tiers (like Render's free tier) typically limit memory to **512MB**. TensorFlow and large models (300MB+) often exceed this, causing the app to crash (OOM - Out Of Memory). 
> **Recommendation**: If deployment fails due to memory, we may need to:
> 1. Use the smallest model (`VGG19_Transfer_best.keras` is ~85MB).
> 2. Upgrade to a paid instance (starting ~$7/mo).
> 3. Optimize the model (convert to TFLite or ONNX).

> [!IMPORTANT]
> **Large File Storage (Git LFS)**: Your model files are large (>100MB). Standard GitHub uploads fail for files >100MB.
> You MUST install Git LFS and track these files before pushing to GitHub.

## Proposed Changes

### 1. Backend Preparation (Render)
We need to ensure the backend allows direct model loading and dependencies are set for a cloud environment.

#### [MODIFY] [backend/main.py](file:///e:/ishva.works/Brain_tumor_detection/backend/main.py)
- Ensure CORS config allows the Vercel app URL (or `*` for testing).
- Add a root health check (already exists).

#### [MODIFY] [backend/requirements.txt](file:///e:/ishva.works/Brain_tumor_detection/backend/requirements.txt)
- Ensure all dependencies are pinned.
- `tensorflow-cpu` is often recommended for deployment to save slug size, but `tensorflow` is fine if space permits.

#### [NEW] [render.yaml](file:///e:/ishva.works/Brain_tumor_detection/render.yaml)
- Create a Blueprint file for Render to define the service.
- **Command**: `uvicorn backend.main:app --host 0.0.0.0 --port 10000` (Render sets `PORT`).

### 2. Frontend Preparation (Vercel)

#### [MODIFY] [frontend/.env.production](file:///e:/ishva.works/Brain_tumor_detection/frontend/.env.production)
- We will need to set `NEXT_PUBLIC_API_URL` to the Render Backend URL once deployed.

### 3. Git LFS Setup (Terminal Actions)
- I will run commands to initialize Git LFS and track `.keras` files.

## Verification Plan

### Automated Tests
- Run `test_chat.py` locally to ensure logic holds.
- **Render**: Monitor build logs for success.
- **Vercel**: Monitor build logs.

### Manual Verification
1. **Push to GitHub**: Confirm LFS files uploaded correctly.
2. **Deploy Backend**: Check Render logs for "Uvicorn running...". Use the `/docs` endpoint to verify it's up.
3. **Deploy Frontend**: Check Vercel URL. Upload a sample MRI image and verify the prediction works (this is the ultimate test of memory limits).
