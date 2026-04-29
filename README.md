# Brain Tumor Classification System

A multimodal deep learning system for brain tumor classification and analysis, featuring a React frontend and a Flask/Python backend.

## Project Structure

- **BrainTumorClass/**: Backend logic, models, and API.
  - `api.py`: Flask API for model serving.
  - `predict.py`: Core prediction and inference logic.
  - `src/`: Source modules for data loading, classification, and XAI (Grad-CAM).
- **frontend/**: React + Vite application for the user interface.
  - `src/`: UI components, pages, and API services.

## Setup

### Backend
1. Navigate to `BrainTumorClass/`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the API: `python api.py`.
4. Local API URL: `http://localhost:5001`.

### Frontend
1. Navigate to `frontend/`.
2. Install dependencies: `npm install`.
3. Run the dev server: `npm run dev`.

## Deployment

### Backend on Render

Deploy `BrainTumorClass/` as a Render Web Service.

Recommended settings:

- Root Directory: `BrainTumorClass`
- Runtime: `Python 3`
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn api:app --bind 0.0.0.0:$PORT --workers 1 --timeout 180`
- Health Check Path: `/api/health`

Environment variables:

- `PYTHON_VERSION=3.10.13`
- `FRONTEND_ORIGIN=https://your-vercel-app.vercel.app`

This repo also includes `render.yaml`, so Render Blueprint deploys can use the same settings automatically.

### Frontend on Vercel

Deploy the `frontend/` folder as the Vercel project.

Recommended settings:

- Framework Preset: `Vite`
- Root Directory: `frontend`
- Build Command: `npm run build`
- Output Directory: `dist`

Environment variables:

- `VITE_API_BASE_URL=https://your-render-service.onrender.com`

The included `frontend/vercel.json` rewrites all routes to `index.html`, so direct links like `/dashboard` work after deployment.

## Features
- Real-time brain tumor classification.
- Probability distribution visualization.
- XAI (Grad-CAM) for model interpretability.
- Diagnostic plotting and pipeline visualization.
