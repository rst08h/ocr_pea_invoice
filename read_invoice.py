import os
import cv2
from ultralytics import YOLO
import onnxruntime as ort

def crop_document_elements(image_path, model_path, output_dir="./cropped_results"):
    # 1. โหลดโมเดล YOLO (ใช้ไฟล์ .pt หรือ .onnx ก็ได้)
    model = YOLO(model_path)

    # 2. อ่านภาพต้นฉบับด้วย OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ ไม่สามารถเปิดไฟล์ภาพได้: {image_path}")
        return

    # 3. รัน Inference ตรวจจับองค์ประกอบ
    results = model(image_path, conf=0.3)[0]

    # สร้างโฟลเดอร์หลักสำหรับเก็บภาพที่ Crop
    os.makedirs(output_dir, exist_ok=True)

    cropped_data = [] # สำหรับเก็บพิกัดและภาพที่ Crop ไว้ใช้งานต่อในความจำ (In-memory)

    # 4. วนลูปอ่าน Bounding Box ทุกตัวที่ตรวจพบ
    for idx, box in enumerate(results.boxes):
        # ดึงค่าพิกัด [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # ดึง Class ID และ Class Name
        class_id = int(box.cls[0].item())
        class_name = results.names[class_id]
        confidence = float(box.conf[0].item())

        # ป้องกันพิกัดทะลุขอบภาพ (Boundary Check)
        h_orig, w_orig = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_orig, x2), min(h_orig, y2)

        # 5. สไลซ์ Crop ภาพตามพิกัด OpenCV [y1:y2, x1:x2]
        cropped_img = image[y1:y2, x1:x2]

        # ข้ามถ้าภาพที่ Crop ขนาดเล็กเกินไปหรือว่างเปล่า
        if cropped_img.size == 0:
            continue

        # สร้างโฟลเดอร์ย่อยแยกตามชื่อ Class
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # ตั้งชื่อไฟล์และบันทึกลงดิสก์
        save_filename = f"{class_name}_{idx + 1}_conf{int(confidence*100)}.png"
        save_path = os.path.join(class_dir, save_filename)
        cv2.imwrite(save_path, cropped_img)

        # เก็บข้อมูลลง List สำหรับส่งเข้า OCR ทันทีโดยไม่ต้องอ่านไฟล์ใหม่
        cropped_data.append({
            "class_name": class_name,
            "confidence": confidence,
            "box": [x1, y1, x2, y2],
            "crop_image": cropped_img
        })

        print(f"✂️ Crop เรียบร้อย: {class_name} -> {save_path}")

    return cropped_data

# ==========================================
# ตัวอย่างการใช้งาน
# ==========================================
if __name__ == "__main__":
    IMAGE_PATH = "output_001.png"
    MODEL_PATH = "best.pt" # หรือ "best.onnx"

    crops = crop_document_elements(IMAGE_PATH, MODEL_PATH)

    # ตัวอย่าง: ดึงเฉพาะภาพโซน 'meter_info' ไปส่งต่อให้ OCR ในระบบ
    for item in crops:
        if "meter_info" in item["class_name"]:
            ocr_target_image = item["crop_image"]
            cv2.imshow("meter_info", ocr_target_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # คุณสามารถนำ ocr_target_image ส่งเข้า EasyOCR / PaddleOCR ได้ทันทีตรงนี้