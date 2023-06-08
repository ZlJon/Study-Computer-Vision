import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'img'
images = []
className = []
myList = os.listdir(path)
print(myList) # ['dill_gates.jpg', 'elon_musk.jpg', 'jack_ma.jpg']
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    className.append(os.path.splitext(cl)[0]) #확장자명 제거
print(className) # ['dill_gates', 'elon_musk', 'jack_ma']

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# 새로운 사진에 대해서 캠에 찍힌 시간과 이름을 기록함. 
def markAttendance(name):
    with open('OpenCVWokring/Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dtString}')

markAttendance('Elon')

encodeListKnown = findEncodings(images)
# print(len(encodeListKnown)) # 3
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = className[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)


    cv2.imshow('Webcam', img)
    cv2.waitKey(1)


# faceLoc = face_recognition.face_locations(imgElon)[0]
# encodeElon = face_recognition.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon,(faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)
# # print(faceLoc) # (170, 990, 491, 669) => 기본적으로 오른쪽 상단, 왼쪽 하단, 사각형으로 x1, y1, x2, y2 순임.

# faceLoc2 = face_recognition.face_locations(imgElon2)[0]
# encodeElon2 = face_recognition.face_encodings(imgElon2)[0]
# cv2.rectangle(imgElon2,(faceLoc2[3], faceLoc2[0]), (faceLoc2[1], faceLoc2[2]), (255, 0, 255), 2)

# # 서로의 얼굴을 비교하여 일치하는지에 대한 true or false 를 출력해줌 => 만약 다르다면 false
# results = face_recognition.compare_faces([encodeElon], encodeElon2)
# faceDis = face_recognition.face_distance([encodeElon], encodeElon2)