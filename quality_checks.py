import cv2
import numpy as np

def is_blurry(image, threshold=80):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    focus_measure = cv2.Laplacian(gray, cv2.CV_64F).var()
    return focus_measure < threshold

def is_too_dark_or_bright(image, low=40, high=210):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = gray.mean()
    return mean_intensity < low or mean_intensity > high

def is_face_too_small(face_box, frame_shape, min_ratio=0.15):
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = face_box

    face_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_w * frame_h

    return (face_area / frame_area) < min_ratio

def quality_check(image, face_box=None, frame_shape=None):
    if is_blurry(image):
        return False, "Image too blurry"

    if is_too_dark_or_bright(image):
        return False, "Poor lighting"

    if face_box and frame_shape:
        if is_face_too_small(face_box, frame_shape):
            return False, "Face too far from camera"

    return True, "OK"
