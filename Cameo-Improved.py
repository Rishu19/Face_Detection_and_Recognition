import cv2
import os
import numpy as np

mapping = ['', 'ELON MUSK', 'AAMIR KHAN', 'SCARLETT JOHANSSON', 'ED SHEERAN']


def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        'C:\\Users\\HP\\PycharmProjects\\Rishabh\\lbpcascades\\lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if (len(faces) == 0):
        return None, None
    for (x, y, w, h) in faces:
        a = x + w
        b = y + h
        (x, y, a, b) = faces[0]

    return gray[y:y + h, x:x + w], faces[0]


def prepare_training_data(folder_path):
    directory = os.listdir(folder_path)
    faces = []
    labels = []

    for dir_name in directory:

        if not dir_name.startswith('s'):
            continue

        label = int(dir_name.replace("s", ""))
        sub_dir_path = folder_path + "\\" + dir_name
        sub_img_names = os.listdir(sub_dir_path)

        for image_name in sub_img_names:
            if image_name.startswith("."):
                continue

            img_path = sub_dir_path + "\\" + image_name
            image = cv2.imread(img_path)
            cv2.imshow("Processing...", cv2.resize(image, (500, 500)))
            cv2.waitKey(1)
            face, rect = detect_face(image)
            if face is not None:
                faces.append(face)
                labels.append(label)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels


print("Processing Training Data...")
faces, labels = prepare_training_data("C:\\Users\\HP\\PycharmProjects\\Rishabh\\Database(Custom)\\training_faces")
print("Data Prepared")
print("Total Faces: " + str(len(faces)))
print("Total Labels: " + str(len(labels)))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

train = face_recognizer.train(faces, np.array(labels))


def draw_rect(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 5)


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (2, 2, 255), 2)


def predict(test):
    img = test.copy()
    face, rect = detect_face(img)
    label, conf = face_recognizer.predict(face)
    maps = mapping[label]
    conf = 100-float(conf)
    rk = str(conf)
    if (conf == 0.0):
        print('100% Match')

    print("CONFIDENCE SCORE: " + rk + '%')

    draw_rect(img, rect)
    draw_text(img, maps, rect[0], rect[1])
    draw_text(img, str(round(conf, 2)), rect[0] + 220, rect[3] + 115)
    return img


print("Predicting....")

sample1 = cv2.imread("C:\\Users\\HP\\PycharmProjects\\Rishabh\\Database(Custom)\\Testing_faces\\s2\\5.jpg")

predicting1 = predict(sample1)

cv2.imshow('Test 1', cv2.resize(predicting1, (600, 600)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
