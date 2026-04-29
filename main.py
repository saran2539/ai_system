import cv2
from ultralytics import YOLO
import easyocr
import time

# 1. โหลดสมอง AI (ใช้ YOLOv8 และ EasyOCR)
model = YOLO('yolov8n.pt') 
reader = easyocr.Reader(['en'], gpu=False) 

url = "http://10.77.145.128:8080/video"
cap = cv2.VideoCapture(url)

# --- แก้ไขจุดนี้: เปลี่ยนจาก 2 เป็น 15 ตามคำขอครับ ---
PERSON_LIMIT = 15 
# ----------------------------------------------

last_ocr_time = 0 

print(f"--- ระบบเริ่มทำงาน (จะเตือนเมื่อคนเกิน {PERSON_LIMIT} คน) ---")

while True:
    success, frame = cap.read()
    if not success: break

    results = model(frame, stream=True)
    person_count = 0
    current_time = time.time()
    annotated_frame = frame.copy()

    for r in results:
        # วาดกรอบวัตถุทุกอย่าง (โน้ตบุ๊ก, แก้วน้ำ, รถ, คน)
        annotated_frame = r.plot() 

        for box in r.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])

            # --- นับจำนวนคน ---
            if class_id == 0: 
                person_count += 1
            
            # --- อ่านทะเบียนรถ (ทำงานทุก 0.5 วินาทีเพื่อความลื่นไหล) ---
            if class_id in [2, 3, 7] and conf > 0.5:
                if current_time - last_ocr_time > 0.5: 
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # เพิ่ม Margin รอบๆ รถอีกนิดเพื่อให้เห็นป้ายทะเบียนชัดขึ้น
                    car_roi = frame[max(0, y1-20):y2+20, max(0, x1-20):x2+20]
                    
                    if car_roi.size > 0:
                        # ลองเปลี่ยนเป็นอ่านทั้งไทยและอังกฤษ (ถ้าลง EasyOCR ครบจะอ่านไทยได้)
                        ocr_res = reader.readtext(car_roi)
                        for (bbox, text, prob) in ocr_res:
                            if prob > 0.3: # ลดเกณฑ์ความมั่นใจลงนิดนึงเพื่อให้แสดงผลบ่อยขึ้น
                                last_ocr_time = current_time
                                # วาดกรอบสีเหลืองและข้อความที่อ่านได้
                                cv2.putText(annotated_frame, f"Detected: {text}", (x1, y1 - 20), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                print(f"พบทะเบียน: {text} (ความชัด: {prob:.2f})")

    # --- ส่วนการแจ้งเตือน ---
    color = (0, 255, 0) # สีเขียว (ถ้าคนไม่เกิน 15)
    
    if person_count > PERSON_LIMIT:
        color = (0, 0, 255) # สีแดง (ถ้าคนเกิน 15)
        # ปรับขนาดตัวอักษรเตือนให้ใหญ่ขึ้น (ขนาด 1.5) เพื่อความเด่นชัด
        cv2.putText(annotated_frame, "!!! DANGER: OVER 15 PEOPLE !!!", (30, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 4)

    # แสดงจำนวนคนปัจจุบัน
    cv2.putText(annotated_frame, f"People in Area: {person_count}", (50, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("AI Security System (Limit: 15)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()