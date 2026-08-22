import cv2
import numpy as np
from inference.pose import extract_landmarks_from_video

video = "checkpoints/debug_upload.mp4"
seq = extract_landmarks_from_video(video, 30)

print(seq.shape)
print("Done")
