import cv2, time
import numpy as np
from PIL import ImageGrab
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')


while True:

    img = ImageGrab.grab(bbox=(0, 0, 1920, 1080))
    tela = np.array(img)

    frame = tela
    
    gray = cv2.cvtColor(tela, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    try:
        predictions = DeepFace.analyze(frame)
        print("bia", predictions)
    except:
        print("bia error") 

    for (x,y,w,h) in faces:
            cv2.putText(frame,predictions['dominant_emotion'],(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_color)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('Capturando',frame)
    
    key=cv2.waitKey(1)

    if key == ord('q'):
        break


cv2.destroyAllWindows()
