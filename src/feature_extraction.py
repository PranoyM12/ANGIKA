import cv2
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def extract_mediapipe_features(image_dir):
    features = []
    labels = []
    
    model_path = os.path.abspath('models/pose_landmarker.task')
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False)
    detector = vision.PoseLandmarker.create_from_options(options)
    
    for pose in os.listdir(image_dir):
        pose_dir = os.path.join(image_dir, pose)
        if not os.path.isdir(pose_dir):
            continue
        for img_name in os.listdir(pose_dir):
            img_path = os.path.join(pose_dir, img_name)
            
            try:
                mp_image = mp.Image.create_from_file(img_path)
            except Exception as e:
                continue
                
            detection_result = detector.detect(mp_image)
            
            if len(detection_result.pose_landmarks) > 0:
                landmarks_list = detection_result.pose_landmarks[0]
                landmarks = []
                for landmark in landmarks_list:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                features.append(landmarks)
                labels.append(pose)
            else:
                print(f"No pose detected in {img_path}")
                
    return np.array(features), np.array(labels)

if __name__ == "__main__":
    features, labels = extract_mediapipe_features('data/raw')
    print(f"Successfully extracted features for {len(features)} images.")
    np.save('data/mp_features.npy', features)
    np.save('data/mp_labels.npy', labels)
