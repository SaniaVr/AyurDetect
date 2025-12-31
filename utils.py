import os
import cv2
from datetime import datetime

def create_session_folder(base_dir="captures"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(base_dir, f"session_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    return session_dir

def crop_region(frame, box):
    h, w, _ = frame.shape
    x1, y1, x2, y2 = box

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]
