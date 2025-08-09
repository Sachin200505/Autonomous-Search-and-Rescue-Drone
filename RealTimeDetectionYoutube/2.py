#! /usr/bin/env python3
import numpy as np
import cv2
from yt_dlp import YoutubeDL
import time
import os

# Set the working directory explicitly (optional, for debugging purposes)
os.chdir(r"C:\Users\Ajay B\Desktop\Hackathons-Jan\Drone")

# Params
RELATIVE_PATH = os.getcwd()
#PROTOTXT = os.path.join(RELATIVE_PATH, "utilities", "SSD_deploy.prototxt")
PROTOTXT = "/utilities/SSD"
MODEL = os.path.join(RELATIVE_PATH, "utilities", "SSD.caffemodel")
URL = "https://www.youtube.com/watch?v=mLEi9PebUNQ"  # YouTube URL

CONF_THRES = 0.4  # Confidence threshold for making predictions

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
           "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

ENABLE_GPU = 0

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def load_model():
    """Load the SSD model from the given paths."""
    # Debug paths
    print(f"[DEBUG] Current Working Directory: {RELATIVE_PATH}")
    print(f"[DEBUG] PROTOTXT Path: {PROTOTXT}")
    print(f"[DEBUG] MODEL Path: {MODEL}")

    # Ensure the prototxt and model files exist
    if not os.path.exists(PROTOTXT):
        raise FileNotFoundError(f"File not found: {PROTOTXT}")
    if not os.path.exists(MODEL):
        raise FileNotFoundError(f"File not found: {MODEL}")

    # Load the model
    model = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    if ENABLE_GPU:
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    print("[INFO] Model loaded successfully.")
    return model

def get_stream_url():
    """Fetch the direct video stream URL using yt-dlp."""
    ydl_opts = {
        'quiet': True,
        'format': 'best[ext=mp4]'
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(URL, download=False)
        stream_url = info['url']
        print("[INFO] Stream URL fetched successfully.")
        return stream_url

def subscribe_stream():
    """Subscribe to the YouTube video stream."""
    stream_url = get_stream_url()
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise ValueError("[ERROR] Failed to open the video stream.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
    print("[INFO] Subscribed to stream.")
    return cap

if __name__ == "__main__":
    try:
        model = load_model()
        video = subscribe_stream()

        cv2.namedWindow('output', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('output', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                print("[INFO] Stream ended or error occurred.")
                break

            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

            model.setInput(blob)
            detections = model.forward()

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > CONF_THRES:
                    idx = int(detections[0, 0, i, 1])

                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            cv2.imshow('output', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
