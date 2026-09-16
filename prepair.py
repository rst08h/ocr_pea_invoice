import cv2
import numpy as np

def order_points(pts):
    """จัดเรียงพิกัด 4 จุด: [ซ้ายบน, ขวาบน, ขวาล่าง, ซ้ายล่าง]"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    return rect

def four_point_transform(image, pts):
    """ทำ Perspective Transform เพื่อดัดภาพเอกสารให้ตั้งตรง"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # คำนวณความกว้างและความสูงใหม่ของเอกสาร
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # พิกัดเป้าหมายแบบสี่เหลี่ยมผืนผ้า
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # คำนวณ Matrix และ Warp ภาพ
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def remove_shadows(gray_img):
    """ลบเงาและปรับความสว่างพื้นหลังให้สม่ำเสมอด้วย Morphological Operations"""
    dilated = cv2.dilate(gray_img, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated, 21)
    diff_img = 255 - cv2.absdiff(gray_img, bg_img)
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return norm_img

def preprocess_document(image_path):
    # 1. Load Image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"ไม่พบไฟล์ภาพ: {image_path}")
    
    orig = image.copy()

    # 2. Document Detection (หาขอบเอกสาร)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # หา Contours ที่ใหญ่ที่สุด 5 อันแรก
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    doc_cnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # ถ้าพิกัดมี 4 จุด สันนิษฐานว่าเป็นขอบกระดาษ
        if len(approx) == 4:
            doc_cnt = approx
            break

    # 3. Perspective Transform (ดัดภาพตรง)
    if doc_cnt is not None:
        warped = four_point_transform(orig, doc_cnt.reshape(4, 2))
    else:
        print("ไม่พบขอบกระดาษ 4 มุมที่ชัดเจน ใช้ภาพเดิมดำเนินการต่อ")
        warped = orig

    # 4. Grayscale Conversion
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # 5. Shadow Removal & Denoising
    no_shadow = remove_shadows(warped_gray)
    denoised = cv2.fastNlMeansDenoising(no_shadow, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # 6. Adaptive Thresholding (Binarization เหมาะสำหรับ OCR)
    binary = cv2.adaptiveThreshold(
        denoised, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        blockSize=15, 
        C=10
    )

    return warped, binary

# --- ตัวอย่างการใช้งาน ---
if __name__ == "__main__":
    # เปลี่ยนเป็นเส้นทางไฟล์ของคุณ
    input_file = "./invoice/13491061_384169_07082026140306.jpg" 
    
    try:
        # warped_img: ภาพสีที่ดัดตรงแล้ว (สำหรับอ่านด้วยตา)
        # binary_img: ภาพขาวดำความละเอียดสูง (สำหรับส่งเข้า OCR)
        warped_img, binary_img = preprocess_document(input_file)

        # บันทึกผลลัพธ์
        cv2.imwrite("output_warped.jpg", warped_img)
        cv2.imwrite("output_binary.jpg", binary_img)
        print("ประมวลผลสำเร็จ บันทึกไฟล์ output_warped.jpg และ output_binary.jpg เรียบร้อยแล้ว")
        
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")