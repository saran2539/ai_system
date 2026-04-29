import cv2

# ใส่ IP มือถือของคุณตรงนี้ (อย่าลืมเติม /video ไว้ท้ายสุด)
# ถ้าเปลี่ยน Wi-Fi แล้วเลข IP เปลี่ยน อย่าลืมมาแก้ตรงนี้ด้วยนะครับ
url = "http://10.77.145.128:8080/video"

print("--- ระบบกำลังเชื่อมต่อกับกล้องมือถือ ---")

# คำสั่งเปิดการเชื่อมต่อ
cap = cv2.VideoCapture(url)

# เช็คว่าคอมเห็นกล้องมือถือไหม
if not cap.isOpened():
    print("!!! Error: ไม่สามารถเชื่อมต่อได้ !!!")
    print("1. เช็คว่ามือถือเปิด Start Server ในแอปหรือยัง")
    print("2. เช็คว่าคอมกับมือถือต่อ Wi-Fi เดียวกันไหม")
else:
    print("--- เชื่อมต่อสำเร็จ! กำลังแสดงภาพ ---")

while True:
    # อ่านภาพจากมือถือ
    success, frame = cap.read()
    
    if success:
        # แสดงภาพหน้าจอ (ย่อขนาดให้ดูง่าย)
        frame_small = cv2.resize(frame, (640, 480))
        cv2.imshow("Mobile Camera", frame_small)
    else:
        print("สัญญาณภาพจากมือถือขาดหาย...")
        break

    # วิธีปิด: คลิกที่หน้าต่างรูปภาพแล้วกดปุ่ม q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()