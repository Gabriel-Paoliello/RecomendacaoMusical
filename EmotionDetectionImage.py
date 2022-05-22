import cv2
import numpy as np
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')

img = cv2.imread('img_database/{}'.format(str(input('Nome do arquivo de imagem: '))))
tela = np.array(img)

frame = tela

gray = cv2.cvtColor(tela, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray,1.3,5)

for (x,y,w,h) in faces:
        #cv2.putText(frame,'Estou feliz',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_color)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

# Ver imagem
#while True:
#
#    cv2.imshow('Capturando',frame)
#
#    key=cv2.waitKey(1)
#
#    if key == ord('q'):
#        break

try:
    predictions = DeepFace.analyze(frame)
    print(predictions['dominant_emotion'])
    print('\n', predictions)
except:
    print("Error") 

cv2.destroyAllWindows()
