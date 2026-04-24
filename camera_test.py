import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not opening")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Failed to capture frame")
        break

    cv2.imshow("Camera Test", frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()