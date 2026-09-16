import torch
from ultralytics import YOLO

def train_yolo():
    # 1. เช็กการใช้งาน GPU
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"🚀 ใช้การประมวลผลบน: {device}")

    # 2. โหลด Pre-trained Model มาเป็นโมเดลฐาน
    # เลือกขนาดได้: yolov8n.pt (เล็กสุด/เร็วสุด), yolov8s.pt, yolov8m.pt, yolov8l.pt (ใหญ่สุด/แม่นสุด)
    # หรือเปลี่ยนเป็น 'yolo11n.pt' หากต้องการใช้เวอร์ชัน YOLO11
    model = YOLO('yolo26m.pt')

    # 3. สั่งเริ่มการเทรน
    results = model.train(
        data='dataset/dataset.yaml',      # Path ชี้ไปยังไฟล์ dataset.yaml ของคุณ
        epochs=100,               # จำนวนรอบในการเทรน (เริ่มต้นแนะนำ 50-100)
        imgsz=640,                # ขนาดความกว้าง/สูงของภาพที่ส่งเข้าโมเดล
        batch=16,                 # จำนวนภาพต่อ 1 Batch (ปรับลดเป็น 8 หรือ 4 หาก GPU RAM ไม่พอ)
        device=device,            # ระบุ device (0 สำหรับ GPU, 'cpu' สำหรับ CPU)
        workers=4,                # จำนวน Thread สำหรับดึงข้อมูลภาพ
        project='runs/train',     # โฟลเดอร์สำหรับเซฟผลลัพธ์การเทรน
        name='pea_doc_layout',    # ชื่อโฟลเดอร์งานการเทรนนี้
        save=True,                # เซฟโมเดล Weights
        patience=20,              # Early stopping (ถ้าประเมินแล้วไม่ดีขึ้นติดกัน 20 epoch จะหยุดอัตโนมัติ)
        pretrained=True           # ใช้ Pre-trained Weights เพื่อช่วยให้เทรนไวขึ้น
    )

    print("\n✅ เทรนโมเดลเสร็จสิ้นเรียบร้อย!")
    print(f"📁 ไฟล์ Weights ถูกบันทึกไว้ที่: runs/train/pea_doc_layout/weights/best.pt")

if __name__ == '__main__':
    train_yolo()