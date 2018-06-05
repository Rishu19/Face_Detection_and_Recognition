import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(
    'C:\\Users\\HP\\PycharmProjects\\Rishabh\\haarcascades\\haarcascade_frontalface_default.xml')
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('C:\\Users\\HP\\PycharmProjects\\Rishabh\\haarcascades\\haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
id = input('Enter the id ')
sample = 0
while 1:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.7, 5)
    for (x, y, w, h) in faces:
        sample = sample + 1
        cv2.imwrite('C:\\Users\\HP\\PycharmProjects\\Rishabh\\Database(Custom)\\Training_faces' + str(id) + '.' + str(sample) + '.jpg',
                    gray[y:y + h, x:x + w])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.waitKey(600)
    cv2.imshow('Rishabh', frame)
    cv2.waitKey(1)
    if sample >= 20:
        break

cap.release()
cv2.destroyAllWindows()
