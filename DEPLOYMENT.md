# Deployment Guide - Brain Tumor Detection

This guide helps you deploy the project for free.

## Prerequisites
- GitHub Account
- [Render Account](https://render.com) (for Backend)
- [Vercel Account](https://vercel.com) (for Frontend)

## 1. Push to GitHub (Critical)
I have already set up Git LFS for you. content
**Run these commands in your terminal to push:**
```bash
git add .
git commit -m "Prepare for deployment"
git push origin main
```
> **Note**: This may take a while as it uploads the large model files (~500MB).

## 2. Deploy Backend (Render)
1. Go to your [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repository.
4. Render should detect the `render.yaml` blueprint (if not, choose "Build from source").
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port 10000`
   - **Plan**: Free
5. Click **Create Web Service**.
6. **Wait for deployment**. Once live, copy the URL (e.g., `https://brain-tumor-api-xxxx.onrender.com`).

> **Warning**: If the deploy fails with "Out of Memory", retry. If it persists, we may need to use a smaller model.

## 3. Deploy Frontend (Vercel)
1. Go to your [Vercel Dashboard](https://vercel.com/dashboard).
2. Click **Add New...** -> **Project**.
3. Import your GitHub repository.
4. **Configure Project**:
   - **Framework Preset**: Next.js (Detected automatically)
   - **Root Directory**: `frontend` (Important! Click "Edit" and select the `frontend` folder).
   - **Environment Variables**:
     - Key: `NEXT_PUBLIC_API_URL`
     - Value: `YOUR_RENDER_BACKEND_URL/api` (Paste the URL from Step 2, append `/api`).
5. Click **Deploy**.

## 4. Final Verification
- Open your Vercel URL.
- Upload an MRI image.
- If you get a result, congratulations! You have deployed a deep learning app for free.
