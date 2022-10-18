import cv2
import pandas as pd
import numpy as np
from deepface import DeepFace
import os

def GetRosto():

    backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
    vid = cv2.VideoCapture(0)

    while True:

        ret,img = vid.read()
        tela = np.array(img)

        frame = tela
        
        cv2.imshow('Capturando',frame)
        
        key=cv2.waitKey(1)

        if key == ord('q'):
            fname = "Temp\\frame.png" 
            cv2.imwrite(fname,frame)
            img = DeepFace.detectFace(fname,detector_backend=backends[3])
            os.remove(fname)
            img = img[:, :, ::-1]
            cv2.imshow('Capturando',img)
            
            min_val,max_val=img.min(),img.max()
            img = 255.0*(img - min_val)/(max_val - min_val)
            img = img.astype(np.uint8)
            cv2.imwrite(fname,img)
            
            while True:
                key=cv2.waitKey(1)
                if key == ord('q'):
                    
                    break
            break

    
    cv2.destroyAllWindows()
    return fname

def get_face_param(img_param):
    df = pd.DataFrame(columns=list(img_param.keys()), data=[list(img_param.values())])
    for emotion in img_param['emotion']:
        df[emotion] = img_param['emotion'][emotion]
    df = df.drop(columns=['emotion','region', 'race'])
    return df

def get_face(image_path):
    fname = image_path

    try:
        predictions = DeepFace.analyze(fname, detector_backend='mtcnn')
    except:
        try:
            predictions = DeepFace.analyze(fname, detector_backend='opencv')
        except:
            print("Erro na leitura da imagem!")
            return 0
    return predictions

pred = get_face(GetRosto())
if pred != 0:
    new_data = get_face_param(pred)

df = new_data
print(df)
df.to_csv("rosto.csv")