import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('traffic.mp4')

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break 

    # Вернули стандартный порог уверенности (30%)
    results = model(frame, stream=True, conf=0.3)

    car_count = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            
            # Теперь считаем только реальный транспорт: 2=машина, 5=автобус, 7=грузовик
            if cls in [2, 5, 7]:
                car_count += 1
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Логика светофора
    if car_count > 10:
        traffic_light = "GREEN (EXTENDED)"
        color = (0, 255, 0) 
    else:
        traffic_light = "RED (SOON)"
        color = (0, 0, 255) 

    # Текст на экране
    cv2.putText(frame, f'Cars detected: {car_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Smart Light: {traffic_light}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow('Smart Intersection MVP', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()