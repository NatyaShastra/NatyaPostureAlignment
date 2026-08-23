import cv2
from inference.pose import extract_landmarks_from_video, extract_mid_frame_rgb
from inference.overlay import draw_skeleton_overlay

video = "checkpoints/debug_upload.mp4"
seq = extract_landmarks_from_video(video, 30)
frame_rgb, mid = extract_mid_frame_rgb(video)

canvas = draw_skeleton_overlay(frame_rgb, seq[len(seq)//2], set(), adavu_label="Test")
cv2.imwrite("test_canvas.jpg", canvas)
print("Done")
