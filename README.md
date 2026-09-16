# ANGIKA — Bharatanatyam Pose Recognition

**ANGIKA** is a machine learning project that automatically recognizes and classifies Bharatanatyam dance poses from images. It uses **Google MediaPipe** to extract 3D body pose landmarks and an **SVM classifier** to identify poses, achieving **83% accuracy** across 9 pose classes.

## Pose Classes

| Class | Class | Class |
|-------|-------|-------|
| Ardhamandalam | Bramha | Garuda |
| Muzhumandi | Nagabandham | Nataraj |
| Prenkhana | Samapadam | Swastika |

## Project Structure

```plaintext
ANGIKA/
├── data/
│   └── raw/            # Input images organized by pose class (e.g. data/raw/Nataraj/)
├── models/             # Trained models + MediaPipe asset (not tracked in git — see models/README.md)
├── src/
│   ├── preprocess.py          # Extracts frames from videos into data/raw/
│   ├── feature_extraction.py  # MediaPipe landmark extraction -> data/mp_features.npy
│   ├── train.py               # Trains SVM -> models/mp_svm_model.pkl
│   ├── evaluate.py            # Prints accuracy + per-class report
│   └── predict.py             # CLI + GUI pose predictor with skeleton overlay
├── requirements.txt
└── README.md
```

## How It Works

1. **Pose Landmark Extraction** — Google MediaPipe detects 33 body keypoints (x, y, z) per image, giving a 99-dimensional feature vector that captures the geometry of the dancer's pose.
2. **Classification** — An SVM trained on these landmark vectors predicts the pose class.
3. **Prediction** — Given a new image, the pipeline detects landmarks, predicts the pose, and overlays the skeleton on the image using OpenCV.

## Setup

### Prerequisites
- Python 3.9+
- Windows / Linux / macOS

### 1. Clone the repository

```bash
git clone https://github.com/PranoyM12/ANGIKA.git
cd ANGIKA
```

### 2. Create and activate a virtual environment

```powershell
# PowerShell (Windows)
python -m venv angika-env
.\angika-env\Scripts\Activate.ps1
```

```bash
# Bash (Linux / macOS)
python -m venv angika-env
source angika-env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the MediaPipe Pose Landmarker model

```powershell
# PowerShell (Windows)
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task" -OutFile "models/pose_landmarker.task"
```

```bash
# Bash (Linux / macOS)
curl -o models/pose_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
```

## Running the Pipeline

> All commands must be run from the project root with the virtual environment activated.

### Step 1 — Preprocess (optional, if starting from raw videos)

```bash
python src/preprocess.py
```

Extracts video frames as RGB images into `data/raw/<pose_class>/`.

### Step 2 — Extract Features

```bash
python src/feature_extraction.py
```

Runs MediaPipe on every image in `data/raw/` and saves landmark features to `data/mp_features.npy` and `data/mp_labels.npy`.

### Step 3 — Train the Model

```bash
python src/train.py
```

Trains an SVM classifier and saves it to `models/mp_svm_model.pkl`.

### Step 4 — Evaluate

```bash
python src/evaluate.py
```

Prints overall accuracy and per-class precision, recall, and F1-score.

### Step 5 — Make Predictions

**CLI mode** — run on a single image and see the skeleton overlay:

```bash
python src/predict.py --image_path path/to/image.jpg
```

**GUI mode** — open an interactive file picker:

```bash
python src/predict.py
```

## Results

| Metric | Score |
|--------|-------|
| Overall Accuracy | **83%** |
| Best class (Nataraj) | 93% F1 |
| Worst class (Ardhamandalam) | 61% F1 |

## Next Steps

- Collect more training data per class to improve accuracy on harder poses (Ardhamandalam, Samapadam).
- Experiment with deep learning classifiers (MLP, CNN on skeleton heatmaps).
- Add real-time webcam prediction support.
- Extend to video-level pose sequence recognition for subtitle generation.

## Tech Stack

- [Google MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) — Pose landmark detection
- [scikit-learn](https://scikit-learn.org/) — SVM classifier
- [OpenCV](https://opencv.org/) — Image processing & skeleton overlay
- Python 3.9+
