import cv2
import numpy as np
import face_recognition

# 사진 파일을 활용하여 얼굴 일치율 확인하기
imgElon = face_recognition.load_image_file('OpenCVWokring/elon_musk.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgElon2 = face_recognition.load_image_file('OpenCVWokring/dill_gates.jpg')
imgElon2 = cv2.cvtColor(imgElon2, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)
# print(faceLoc) # (170, 990, 491, 669) => 기본적으로 오른쪽 상단, 왼쪽 하단, 사각형으로 x1, y1, x2, y2 순임.

faceLoc2 = face_recognition.face_locations(imgElon2)[0]
encodeElon2 = face_recognition.face_encodings(imgElon2)[0]
cv2.rectangle(imgElon2,(faceLoc2[3], faceLoc2[0]), (faceLoc2[1], faceLoc2[2]), (255, 0, 255), 2)

# 서로의 얼굴을 비교하여 일치하는지에 대한 true or false 를 출력해줌 => 만약 다르다면 false
results = face_recognition.compare_faces([encodeElon], encodeElon2)
faceDis = face_recognition.face_distance([encodeElon], encodeElon2)
print(results, faceDis) # [True] elon_musk와 dill_gates를 비교하면 다르기 때문에 [False]
# face_distance => 일치 정도를 파악해줌 0 ~ 1 1에 가까울수록 신뢰성이 낮아진다고 보면 됨.
cv2.putText(imgElon2,f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('elon_musk', imgElon)
cv2.imshow('elon_musk2', imgElon2)
cv2.waitKey(0)
cv2.destroyAllWindows()