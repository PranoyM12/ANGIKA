import cv2
import os
import joblib
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import argparse

# MediaPipe Pose connections (pairs of landmark indices)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]


def draw_skeleton(img, landmarks_list):
    """Draw pose landmarks and connections on an image using OpenCV."""
    h, w = img.shape[:2]
    points = []
    for lm in landmarks_list:
        cx, cy = int(lm.x * w), int(lm.y * h)
        points.append((cx, cy))

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(img, points[start_idx], points[end_idx], (0, 255, 0), 2)

    for pt in points:
        cv2.circle(img, pt, 4, (0, 0, 255), -1)

    return img


def get_detector():
    model_path = os.path.abspath('models/pose_landmarker.task')
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False)
    return vision.PoseLandmarker.create_from_options(options)


def extract_feature(img_path, detector):
    mp_image = mp.Image.create_from_file(img_path)
    detection_result = detector.detect(mp_image)

    if len(detection_result.pose_landmarks) == 0:
        raise ValueError("No human pose detected in the image.")

    landmarks_list = detection_result.pose_landmarks[0]
    landmarks = []
    for lm in landmarks_list:
        landmarks.extend([lm.x, lm.y, lm.z])

    # Draw skeleton using OpenCV (no solutions module needed)
    img = mp_image.numpy_view().copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = draw_skeleton(img, landmarks_list)

    return np.array(landmarks).reshape(1, -1), img


def predict_pose(model_path, img_path, detector):
    model = joblib.load(model_path)
    feature, annotated_img = extract_feature(img_path, detector)
    prediction = model.predict(feature)
    return prediction[0], annotated_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Bharatanatyam pose with MediaPipe")
    parser.add_argument('--image_path', type=str, help="Path to the image")
    args = parser.parse_args()

    model_file = 'models/mp_svm_model.pkl'
    detector = get_detector()

    if args.image_path:
        try:
            pose, annotated_img = predict_pose(model_file, args.image_path, detector)
            print(f"Predicted Pose: {pose}")
            cv2.imshow("Detected Pose", annotated_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        # GUI mode
        import tkinter as tk
        from tkinter import filedialog, messagebox

        root = tk.Tk()
        root.title("Bharatanatyam Pose Prediction")

        def select_image():
            img_path = filedialog.askopenfilename(
                title="Select an image",
                filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
            )
            if img_path:
                try:
                    pose, annotated_img = predict_pose(model_file, img_path, detector)
                    result_label.config(text=f"Predicted Pose: {pose}")
                    cv2.imshow("Detected Pose", annotated_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                except Exception as e:
                    messagebox.showerror("Error", f"An error occurred: {e}")

        select_button = tk.Button(root, text="Select Image", command=select_image, font=("Helvetica", 12))
        select_button.pack(pady=20)
        result_label = tk.Label(root, text="Predicted Pose: None", font=("Helvetica", 14))
        result_label.pack(pady=20)
        root.mainloop()
