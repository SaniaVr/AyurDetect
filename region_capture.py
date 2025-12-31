import os
import cv2
from utils import crop_region, create_session_folder
from region_standardizer import standardize_region
from quality_checks import quality_check

def capture_regions(clean_frame, regions, face_box,
                    teeth_box=None, base_dir="captures"):

    # 🔹 Global quality check on full face
    ok, msg = quality_check(clean_frame, face_box, clean_frame.shape)
    if not ok:
        print(f"❌ Capture rejected: {msg}")
        return None

    session_dir = create_session_folder(base_dir)

    # 1️⃣ Full face
    face_img = crop_region(clean_frame, face_box)
    face_img = standardize_region(face_img, "full_face")
    cv2.imwrite(os.path.join(session_dir, "full_face.jpg"), face_img)

    # 2️⃣ Facial regions
    for name, box in regions.items():
        region_img = crop_region(clean_frame, box)
        region_img = standardize_region(
            region_img, name.lower().replace(" ", "_")
        )
        cv2.imwrite(
            os.path.join(
                session_dir,
                name.lower().replace(" ", "_") + ".jpg"
            ),
            region_img
        )

    # 3️⃣ Teeth (optional)
    if teeth_box:
        teeth_img = crop_region(clean_frame, teeth_box)
        teeth_img = standardize_region(teeth_img, "teeth")
        cv2.imwrite(os.path.join(session_dir, "teeth.jpg"), teeth_img)

    print("✅ Capture passed quality checks")
    return session_dir
