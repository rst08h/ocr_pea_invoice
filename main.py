import doc_onnx 
import cv2



invoice_img = "invoice/13490857_382215_03082026161717.jpg"
# เรียกใช้ฟังก์ชันประมวลผล
detections = doc_onnx.find_box(invoice_img)

# โหลดภาพต้นฉบับมาเพื่อวาดกรอบ
img = cv2.imread(invoice_img)

# วาดกรอบสี่เหลี่ยมและข้อความจากข้อมูลที่ได้รับคืนมา
for item in detections:
    class_name = item["class_name"]
    score = item["confidence"]
    left, top, width, height = item["box"]
    #ส่งไป ocr
    
    match class_name:
        case 'PEA_Logo':
            print('is invoice')
    
    
    
    
    # วาดกรอบสี่เหลี่ยม
    cv2.rectangle(
        img, (left, top), (left + width, top + height), (0, 255, 0), 2
    )

    # เขียนชื่อ Class และ % ความมั่นใจ
    label = f"{class_name}: {score:.2f}"
    cv2.putText(
        img,
        label,
        (left, top - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

# แสดงผลภาพออกหน้าจอ
cv2.imshow("PEA Layout Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
