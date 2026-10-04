# ANGIKA — AI-Based Bharatanatyam Pose Recognition

**ANGIKA** is a computer-vision and neural-network project that recognizes Bharatanatyam dance poses from images. Google MediaPipe extracts 3D body landmarks and a neural-network (MLP) classifier learns the relationship between those landmarks and the nine pose classes.

The previous version used a **linear SVM**. The current version replaces that classifier with a **feed-forward neural network (MLP)** and feature standardization. The SVM can be retained separately as a baseline for research comparison, but it is no longer the main ANGIKA model.

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
│   └── raw/                  # Input images organized by pose class
├── models/                   # Trained neural-network model + MediaPipe asset
├── src/
│   ├── preprocess.py        # Extracts frames from videos into data/raw/
│   ├── feature_extraction.py# MediaPipe landmark extraction -> 99 features
│   ├── train.py             # Trains StandardScaler + MLP -> mp_mlp_model.pkl
│   ├── evaluate.py          # Held-out test metrics + confusion matrix
│   └── predict.py           # CLI + GUI pose predictor with skeleton overlay
├── requirements.txt
└── README.md
```

## How It Works

1. **Pose Landmark Extraction** — MediaPipe detects 33 body landmarks (x, y, z) per image, producing a 99-dimensional feature vector.
2. **Feature Standardization** — `StandardScaler` puts the landmark features on comparable scales before neural-network training.
3. **Neural Classification** — an MLP with two hidden layers (128 and 64 neurons, ReLU activation, Adam optimizer and early stopping) predicts the pose class.
4. **Prediction** — a new image is passed through the same MediaPipe + scaling + MLP pipeline and the predicted class is displayed with the detected skeleton.

## Why MLP Instead of the Previous Linear SVM?

The previous classifier used a linear decision boundary. The MLP can learn **non-linear relationships** among the 99 landmark coordinates, which is useful because a dance pose is defined by the configuration of many joints together rather than by independent coordinates.

This is an incremental upgrade: the MediaPipe landmark extractor remains unchanged, so the project can directly compare the new neural model against the old SVM baseline if required for the report.

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

> Run all commands from the project root with the virtual environment activated.

### Step 1 — Preprocess (optional)

```bash
python src/preprocess.py
```

### Step 2 — Extract Features

```bash
python src/feature_extraction.py
```

This creates:
- `data/mp_features.npy`
- `data/mp_labels.npy`

### Step 3 — Train the Neural Network

```bash
python src/train.py
```

This trains the MLP and saves:

```text
models/mp_mlp_model.pkl
```

### Step 4 — Evaluate

```bash
python src/evaluate.py
```

Evaluation uses the same **stratified 80/20 held-out split** as training and reports accuracy, precision, recall, F1-score and a confusion matrix.

### Step 5 — Make Predictions

**CLI:**

```bash
python src/predict.py --image_path path/to/image.jpg
```

**GUI:**

```bash
python src/predict.py
```

## Current Limitations and Planned AI Extensions

The current upgraded model is still a **single-image pose classifier**. The following are planned research extensions rather than already implemented features:

- MediaPipe Hands for detailed mudra recognition.
- Temporal models such as LSTM/Transformer for movement sequences.
- Reference-vs-learner landmark comparison.
- Joint-level deviation and similarity scoring.
- Real-time webcam feedback.
- Performer-independent evaluation when suitable performer labels/data are available.

## Tech Stack

- **Google MediaPipe** — 3D body landmark detection
- **scikit-learn MLPClassifier** — neural-network pose classifier
- **StandardScaler** — feature normalization
- **OpenCV** — image processing and skeleton overlay
- **NumPy** — numerical feature storage
- **joblib** — model serialization
