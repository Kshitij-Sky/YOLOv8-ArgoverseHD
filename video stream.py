import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('runs/detect/train/weights/best.pt')

# Open the video file
video_path = 'https://www.youtube.com/watch?v=wtKSwWmHaaI'      # Enter video path here
results = model.track(source=video_path, show=True)
print(results)

