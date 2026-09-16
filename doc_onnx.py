import cv2
import numpy as np
import onnxruntime as ort
import time

# 1. รายชื่อ Class เรียงตาม Index (ตรงกับ dataset.yaml)
CLASSES = [
    "PEA_Logo", "Header", "meter_info", "meter_read_tou", "meter_read_srm",
    "meter_calc_detail", "meter_calc_baht", "invoice_summary", "Usage_History",
    "Sub-meter_deduction", "Header_type2", "meter_info_type2", "meter_read_tou_type2",
    "meter_read_srm_type2", "meter_calc_detail_type2", "meter_calc_baht_type2",
    "Power_Outage", "meter_read_type2"
]

#invoice_img="invoice/13490859_380432_05082026161737.jpg"
invoice_img="invoice/13490857_382215_03082026161717.jpg"


# 1. โหลดเซสชัน ONNX Runtime
session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])

# 2. อ่านภาพและเตรียม Image Pre-processing (640x640, BGR -> RGB, Normalized 0-1)
img = cv2.imread(invoice_img)
h_orig, w_orig = img.shape[:2]

input_img = cv2.resize(img, (640, 640))
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
input_img = input_img.transpose(2, 0, 1).astype(np.float32) / 255.0
input_img = np.expand_dims(input_img, axis=0)

start_time = time.perf_counter()
# 3. สั่งประมวลผลผ่าน ONNX
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: input_img})
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time:.4f} seconds")
# outputs[0] จะคืนค่า Bounding Boxes, Confidence scores, และ Class IDs

# 5. ถอดรหัสผลลัพธ์ (Post-processing)
predictions = np.squeeze(outputs).T # Shape: (8400, 22)

boxes, confidences, class_ids = [], [], []
x_factor = w_orig / 640.0
y_factor = h_orig / 640.0

for row in predictions:
    scores = row[4:]
    class_id = np.argmax(scores)
    confidence = scores[class_id]

    if confidence > 0.3: # Filter ตาม Confidence score threshold
        x, y, w, h = row[0], row[1], row[2], row[3]
        
        # แปลงพิกัดจาก Center (x, y, w, h) เป็น Top-Left (x1, y1, w, h) สเกลกลับเท่าภาพจริง
        left = int((x - 0.5 * w) * x_factor)
        top = int((y - 0.5 * h) * y_factor)
        width = int(w * x_factor)
        height = int(h * y_factor)

        boxes.append([left, top, width, height])
        confidences.append(float(confidence))
        class_ids.append(class_id)

# 6. กรองกรอบที่ซ้อนทับกันด้วย Non-Maximum Suppression (NMS)
indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.3, nms_threshold=0.45)

# 7. วาดกรอบสี่เหลี่ยมและชื่อ Class ลงบนภาพ original
for i in indices:
    box = boxes[i]
    left, top, width, height = box[0], box[1], box[2], box[3]
    class_name = CLASSES[class_ids[i]]
    score = confidences[i]

    # วาด กรอบสี่เหลี่ยม
    cv2.rectangle(img, (left, top), (left + width, top + height), (0, 255, 0), 2)
    
    # เขียนชื่อ Class และ % ความมั่นใจ
    label = f"{class_name}: {score:.2f}"
    cv2.putText(img, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# 8. แสดงผลภาพออกหน้าจอ
# cv2.imshow("PEA Layout Result", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# หรือจะบันทึกเป็นภาพใหม่
cv2.imwrite("result_onnx.jpg", img)

