import cv2
import numpy as np
from docaligner import DocAligner

def warp_perspective(image, corners):
    pts1 = np.float32(corners)

    # คำนวณความกว้างและความสูงเป้าหมาย
    top_w = np.linalg.norm(pts1[0] - pts1[1])
    bot_w = np.linalg.norm(pts1[3] - pts1[2])
    max_w = int(max(top_w, bot_w))

    left_h = np.linalg.norm(pts1[0] - pts1[3])
    right_h = np.linalg.norm(pts1[1] - pts1[2])
    max_h = int(max(left_h, right_h))

    # พิกัดปลายทาง
    pts2 = np.float32([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1]
    ])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(image, matrix, (max_w, max_h))
    return warped

def process_document(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"ไม่พบไฟล์: {image_path}")

    aligner = DocAligner()
    pred = aligner(image)

    # เช็คว่าผลลัพธ์ส่งกลับมาเป็น dict หรือ array โดยตรง
    if isinstance(pred, dict) and 'corners' in pred:
        corners = pred['corners']
    else:
        corners = pred

    # แปลงให้เป็น numpy array
    corners = np.array(corners)

    print(f"ขนาดพิกัดมุมที่ตรวจพบ (Shape): {corners.shape}")

    # ตรวจสอบว่าได้ครบ 4 จุดหรือไม่ก่อนทำ Perspective Transform
    if corners is None or len(corners) < 4:
        print("⚠️ ไม่สามารถตรวจพบพิกัดครบ 4 มุมได้ (หาเจอไม่ครบ 4 จุด)")
        # คืนค่าภาพเดิมกลับไปป้องกันโปรแกรมพัง
        return image

    warped_image = warp_perspective(image, corners)
    return warped_image

if __name__ == "__main__":
    result = process_document("meter_info.jpg")
    cv2.imwrite("warped_result.jpg", result)
    
    
