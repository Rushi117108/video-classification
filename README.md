---
title: Video Action Classifier
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Video Action Classifier

Classifies human actions in videos using a hybrid **EfficientNet + BiLSTM** architecture.
- Trained on **UCF101** benchmark — 101 action categories
- Achieves **89.9% accuracy** on UCF101 test set
- Built with **TensorFlow**, served via **FastAPI**, containerized with **Docker**

## API Usage

### Health Check
GET /health

### Predict
POST /predict
- Body: form-data, key = `file`, value = your video file (.mp4 / .avi)

### Response
{
  "predicted_class": "Basketball",
  "confidence": 0.94,
  "top_3": [...]
}

## Setup

### 1. Clone the repo
git clone https://github.com/Rushi117108/video-classification.git

### 2. Download the model
Download `efficient_net_model_full.h5` from HuggingFace and place it in the `model/` folder:
👉 https://huggingface.co/spaces/rushi-117/video-action-classifier

### 3. Install dependencies
pip install -r requirements.txt

### 4. Run locally
uvicorn app.main:app --reload --port 8000
