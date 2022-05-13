import cv2, time
import numpy as np
from PIL import ImageGrab
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier('../haarcascade/haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('../haarcascade/haarcascade_eye.xml')

a=0

while True:
    a= a+1

    img = ImageGrab.grab(bbox=(0, 0, 1920, 1080))
    tela = np.array(img)

    frame = tela
    a=0
    #print(frame)

    gray = cv2.cvtColor(tela, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
            cv2.putText(frame,'Estou feliz',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_color)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('Capturando',frame)
    #try:
    #    predictions = DeepFace.analyze(frame)
    #    print("bia", predictions)
    #except:
    #    print("bia error")
    key=cv2.waitKey(1)

    if key == ord('q'):
        break


cv2.destroyAllWindows()
