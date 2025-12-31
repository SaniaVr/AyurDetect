import cv2
import numpy as np
from ultralytics import YOLO

import os
from datetime import datetime

# model = YOLO("yolov8n-face.pt")
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

model_path = hf_hub_download(
    repo_id="arnabdhar/YOLOv8-Face-Detection",
    filename="model.pt"
)

model = YOLO(model_path)

def detect_face(frame):
    # results = model(frame, conf=0.5, verbose=False)
    results = model(frame, conf=0.6, iou=0.5, verbose=False)
    if len(results) == 0 or results[0].boxes is None:
        return None

    boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:
        return None

    x1, y1, x2, y2 = boxes[0].astype(int)
    return x1, y1, x2, y2

def ayurvedic_face_regions(x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1

    regions = {
        # Forehead: unchanged
        "Forehead": (
            x1 + int(0.10 * w),
            y1 + int(0.02 * h),
            x2 - int(0.10 * w),
            y1 + int(0.28 * h)
        ),

        # Eyes: WIDER horizontally
        "Eyes": (
            x1 + int(0.08 * w),   # was 0.15
            y1 + int(0.25 * h),
            x2 - int(0.08 * w),   # was 0.15
            y1 + int(0.45 * h)
        ),

        # Left Cheek: unchanged
        "Left Cheek": (
            x1 + int(0.02 * w),
            y1 + int(0.42 * h),
            x1 + int(0.38 * w),
            y2 - int(0.28 * h)
        ),

        # Right Cheek: unchanged
        "Right Cheek": (
            x2 - int(0.38 * w),
            y1 + int(0.42 * h),
            x2 - int(0.02 * w),
            y2 - int(0.28 * h)
        ),

        # Nose: unchanged
        "Nose": (
            x1 + int(0.38 * w),
            y1 + int(0.30 * h),
            x2 - int(0.38 * w),
            y1 + int(0.60 * h)
        ),

        # Chin: STARTS BELOW LIPS (IMPORTANT FIX)
        "Chin": (
            x1 + int(0.30 * w),
            y1 + int(0.80 * h),   # ⬅️ moved down (was 0.60)
            x2 - int(0.30 * w),
            y2 + int(0.05 * h)
        )
    }

    return regions


def detect_teeth(frame, x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1

    # Mouth region (between nose & chin)
    mouth = frame[
        y1 + int(0.58 * h): y1 + int(0.68 * h),
        x1 + int(0.22 * w): x2 - int(0.22 * w)
    ]

    if mouth.size == 0:
        return None

    gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)

    # Teeth indicators
    mean_intensity = gray.mean()
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean()

    # Smile / teeth heuristic (tuned)
    if mean_intensity > 155 and edge_density > 8:
        return (
            x1 + int(0.22 * w),
            y1 + int(0.58 * h),
            x2 - int(0.22 * w),
            y1 + int(0.68 * h)
        )

    return None

def draw_regions(frame, regions):
    colors = {
        "Forehead": (255, 255, 0),
        "Eyes": (0, 255, 255),
        "Left Cheek": (0, 255, 0),
        "Right Cheek": (0, 255, 0),
        "Nose": (255, 0, 0),
        # "Lips": (255, 0, 255),
        "Chin": (0, 165, 255)
    }

    for name, (x1, y1, x2, y2) in regions.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), colors.get(name, (255,255,255)), 2)
        cv2.putText(
            frame, name, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.get(name, (255,255,255)), 1
        )

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


def save_region(frame, box, save_path):
    region = crop_region(frame, box)
    if region is not None:
        cv2.imwrite(save_path, region)

def capture_regions(frame, regions, face_box, teeth_box=None, base_dir="captures"):
    """
    Saves clean (annotation-free) crops for:
    - Full face
    - Each Ayurvedic facial region
    - Teeth (if present)
    """
    session_dir = create_session_folder(base_dir)

    # 1️⃣ Save full face
    save_region(
        frame,
        face_box,
        os.path.join(session_dir, "full_face.jpg")
    )

    # 2️⃣ Save each facial region
    for name, box in regions.items():
        filename = name.lower().replace(" ", "_") + ".jpg"
        save_region(
            frame,
            box,
            os.path.join(session_dir, filename)
        )

    # 3️⃣ Save teeth if detected
    if teeth_box:
        save_region(
            frame,
            teeth_box,
            os.path.join(session_dir, "teeth.jpg")
        )

    return session_dir


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Cannot access webcam")
        return

    print("✅ Webcam running")
    print("👉 Press 'c' to CAPTURE images")
    print("👉 Press 'q' to QUIT")

    prev_face = None
    ALPHA = 0.7

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))

        clean_frame = frame.copy()
        display_frame = frame.copy()

        face = detect_face(frame)
        regions = None
        teeth_box = None

        if face:
            if prev_face is None:
                prev_face = face
            else:
                prev_face = tuple(
                    int(ALPHA * p + (1 - ALPHA) * f)
                    for p, f in zip(prev_face, face)
                )

            regions = ayurvedic_face_regions(*prev_face)
            draw_regions(frame, regions)

            teeth_box = detect_teeth(frame, *prev_face)
            if teeth_box:
                cv2.rectangle(frame, (teeth_box[0], teeth_box[1]),
                              (teeth_box[2], teeth_box[3]), (255, 255, 255), 2)
                cv2.putText(frame, "Teeth",
                            (teeth_box[0], teeth_box[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)

        else:
            prev_face = None
            cv2.putText(frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Ayurvedic Facial Region Detection", frame)

        key = cv2.waitKey(1) & 0xFF

        # 📸 CAPTURE BUTTON
        #from region_capture import capture_regions
        
        if key == ord('c') and prev_face and regions:
            session_dir = capture_regions(
            frame=clean_frame,
            regions=regions,
            face_box=prev_face,
            teeth_box=teeth_box
            )
            print(f"✅ Regions captured at {session_dir}")
        
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()