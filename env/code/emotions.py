import cv2
import numpy as np
from deepface import DeepFace

# Load video
video_path = "data/c.mp4"
cap = cv2.VideoCapture(video_path)

frame_rate = int(cap.get(cv2.CAP_PROP_FPS))  # Get frame rate
frame_count = 0

emotion_results = []  # Store results as (time, emotion)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop when the video ends

    # Process every nth frame to optimize
    if frame_count % frame_rate == 0:  # Process one frame per second
        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion']
            timestamp = frame_count / frame_rate
            emotion_results.append((timestamp, emotion))
            print(f"Time: {timestamp:.2f}s, Emotion: {emotion}")
        except Exception as e:
            print(f"Error at frame {frame_count}: {e}")

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

# Save results to a file
with open("emotion_results.txt", "w") as f:
    for timestamp, emotion in emotion_results:
        f.write(f"{timestamp:.2f}s: {emotion}\n")
