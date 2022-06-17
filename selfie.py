import cv2
from datetime import datetime as dt
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile_default.xml')

while True:
    _, frame = cap.read()
    original_img = face.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    for x, y, w, h in face:
        cv2.rectangle(face, (x, y), (x + w, y + h), (0, 255, 255), 2)
        face_roi = frame[y:y+h, x:x+w]
        gray_roi = gray[y:y+h, x:x+w]
        smile = smile_cascade.detectMultiScale(gray_roi, 1.3, 5)
        for x1, y1, w1, h1 in smile:
            cv2.rectangle(face_roi, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)
            cv2.imwrite(f'./pics/{str(dt.now().strftime("%Y-%m-%d-%H:%M:%S"))}.png', original_img)
    cv2.imshow('Image', frame)
    cv2.imshow('Gray', gray)
    if cv2.waitKey(10) == ord('q') or cv2.waitKey() == 27:
        break