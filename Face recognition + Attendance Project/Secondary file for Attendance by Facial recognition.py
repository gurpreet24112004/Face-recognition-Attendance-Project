import cv2
import numpy as np
import face_recognition


imgkalam = face_recognition.load_image_file('Images/kalam.jpg')
imgkalam = cv2.cvtColor(imgkalam , cv2.COLOR_BGR2RGB)
imgkalamtest = face_recognition.load_image_file('Images/kalamtest.jpg')
imgkalamtest = cv2.cvtColor(imgkalamtest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgkalam)[0]
encodekalam = face_recognition.face_encodings(imgkalam)[0]
cv2.rectangle(imgkalam, (faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
faceLocTest = face_recognition.face_locations(imgkalamtest)[0]
encodekalamtest = face_recognition.face_encodings(imgkalamtest)[0]
cv2.rectangle(imgkalamtest, (faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

result = face_recognition.compare_faces([encodekalam],encodekalamtest)
facedis = face_recognition.face_distance([encodekalam],encodekalamtest)
print(result,facedis)
cv2.putText(imgkalamtest,f'{result}{round(facedis[0],2)}',(50,50),cv2.FONT_ITALIC,1,(0,0,255),2)


cv2.imshow('kalam test',imgkalamtest)
cv2.imshow('kalam',imgkalam)
cv2.waitKey(0)