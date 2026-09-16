from ultralytics import YOLO
import time

# โหลดโมเดลสำเร็จรูปสำหรับ Document Layout จาก Hugging Face หรือ local weights
#model = YOLO("yolo26l-doclaynet.pt")
model = YOLO("best.pt")

start_time = time.perf_counter()

# ประมวลผลภาพเอกสาร

#results = model("invoice/13490857_382215_03082026161717.jpg")
results = model("invoice/13490859_380432_05082026161737.jpg")
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time:.4f} seconds")
# แสดงผลการตรวจจับตำแหน่งองค์ประกอบต่างๆ
results[0].show()