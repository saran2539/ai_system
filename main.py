import cv2
from ultralytics import YOLO

# 1. โหลดโมเดล
model = YOLO('yolov8n.pt') 

# 2. ตั้งค่ากล้องมือถือ
url = "http://10.77.145.128:8080/video"
cap = cv2.VideoCapture(url)

PERSON_LIMIT = 10 

while True:
    success, frame = cap.read()
    if not success: break

    # 3. ให้ AI ตรวจจับ (เราจะใช้ผลลัพธ์ 2 แบบ)
    results = model(frame, stream=True)
    
    person_count = 0 

    for r in results:
        # --- ส่วนที่ 1: วาดกรอบทุกอย่างที่ AI เห็น (คน, รถ, ของใช้) ---
        annotated_frame = r.plot() 

        # --- ส่วนที่ 2: วนลูปเพื่อหาจำนวนคนมานับเลข ---
        for box in r.boxes:
            class_id = int(box.cls[0])
            if class_id == 0:  # 0 คือ ID ของ 'person'
                person_count += 1

    # 4. แสดงจำนวนคนและแจ้งเตือนบน annotated_frame (ภาพที่มีกรอบทุกอย่างแล้ว)
    color = (0, 255, 0)
    if person_count > PERSON_LIMIT:
        color = (0, 0, 255)
        cv2.putText(annotated_frame, "!!! ALERT: TOO MANY PEOPLE !!!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.putText(annotated_frame, f"People Count: {person_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # แสดงผลภาพ annotated_frame
    cv2.imshow("AI Multi-Detection & Counter", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()