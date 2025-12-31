import cv2

REGION_SIZES = {
    "full_face": (224, 224),
    "forehead": (128, 128),
    "eyes": (128, 64),
    "left_cheek": (128, 128),
    "right_cheek": (128, 128),
    "nose": (96, 128),
    "chin": (128, 128),
    "teeth": (128, 64)
}

def standardize_region(image, region_name):
    if image is None:
        return None

    if region_name not in REGION_SIZES:
        return image

    width, height = REGION_SIZES[region_name]
    return cv2.resize(image, (width, height),
                      interpolation=cv2.INTER_AREA)
