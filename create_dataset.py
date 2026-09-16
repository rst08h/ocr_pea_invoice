import os
import random
import shutil
from pathlib import Path

# ==========================================
# 1. ตั้งค่า Path และ อัตราส่วน
# ==========================================
SOURCE_DIR = "./invoice"    # โฟลเดอร์ต้นทางที่มีทั้งไฟล์ภาพ และ .txt ปนกันอยู่
OUTPUT_DIR = "./dataset"    # โฟลเดอร์ปลายทางที่จะจัดโครงสร้างสำหรับ YOLO
TRAIN_RATIO = 0.8               # สัดส่วน Train 80% (ส่วน Val จะเป็น 20% อัตโนมัติ)
SEED = 42                       # ล็อกสุ่มเพื่อให้ผลลัพธ์เหมือนเดิมทุกครั้งที่รัน

# นามสกุลไฟล์รูปภาพที่รองรับ
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ==========================================
# 2. ฟังก์ชันหลักสำหรับดำเนินการ
# ==========================================
def split_dataset():
    random.seed(SEED)
    source_path = Path(SOURCE_DIR)
    output_path = Path(OUTPUT_DIR)

    # ค้นหาไฟล์รูปภาพทั้งหมดใน SOURCE_DIR
    image_files = [f for f in source_path.glob("*") if f.suffix.lower() in IMAGE_EXTENSIONS]
    
    # จับคู่ภาพกับไฟล์ .txt
    valid_pairs = []
    for img_path in image_files:
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            valid_pairs.append((img_path, txt_path))
        else:
            print(f"⚠️ คำเตือน: ไม่พบไฟล์ .txt สำหรับภาพ {img_path.name} (จะถูกข้าม)")

    if not valid_pairs:
        print("❌ ไม่พบคู่ไฟล์ภาพและ .txt เลย กรุณาเช็ก Path โฟลเดอร์ SOURCE_DIR อีกครั้ง")
        return

    print(f"พบคู่ไฟล์ที่สมบูรณ์ทั้งหมด: {len(valid_pairs)} คู่")

    # สุ่มสลับลำดับคู่ไฟล์
    random.shuffle(valid_pairs)

    # คำนวณจุดตัดสำหรับแบ่ง Train / Val
    train_count = int(len(valid_pairs) * TRAIN_RATIO)
    train_pairs = valid_pairs[:train_count]
    val_pairs = valid_pairs[train_count:]

    # สร้างโครงสร้างโฟลเดอร์ปลายทาง
    for split in ["train", "val"]:
        (output_path / split / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split / "labels").mkdir(parents=True, exist_ok=True)

    # คัดลอกไฟล์ไปยังโฟลเดอร์ปลายทาง
    def copy_files(pairs, split_name):
        for img_p, txt_p in pairs:
            dest_img = output_path / split_name / "images" / img_p.name
            dest_txt = output_path / split_name / "labels" / txt_p.name
            shutil.copy2(img_p, dest_img)
            shutil.copy2(txt_p, dest_txt)

    copy_files(train_pairs, "train")
    copy_files(val_pairs, "val")

    print("\n✅ จัดเตรียม Dataset เรียบร้อยแล้ว!")
    print(f"  - ชุด Train: {len(train_pairs)} คู่ -> {output_path}/train")
    print(f"  - ชุด Val  : {len(val_pairs)} คู่ -> {output_path}/val")

if __name__ == "__main__":
    split_dataset()