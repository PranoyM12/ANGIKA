# This directory holds trained model files and the MediaPipe pose landmarker asset.
# These files are NOT tracked in git because they are large binaries.
#
# To set up:
# 1. Download the MediaPipe pose landmarker model:
#    Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task" -OutFile "models/pose_landmarker.task"
#
# 2. Run the pipeline to generate the trained SVM:
#    python src/feature_extraction.py
#    python src/train.py
