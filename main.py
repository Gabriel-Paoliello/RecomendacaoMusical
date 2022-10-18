import cv2
import numpy as np
import os
from deepface import DeepFace
import pandas as pd
from asyncio.windows_events import NULL
from genericpath import exists
from pandas import DataFrame
from sklearn.neighbors import NearestNeighbors
import math

import webbrowser



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
            try:
                img = DeepFace.detectFace(fname,detector_backend=backends[3])
            except:
                continue
            
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

def get_face_image(image_path):
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

def training(database):

    df_train = database[['age', 'gender', 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']]
    df_train['gender'] = df_train['gender'].replace('Man',0).replace('Woman',1)

    X = df_train
    n_neighbors = 5
    knn = NearestNeighbors(n_neighbors=n_neighbors+1, metric= "cosine")
    knn.fit(X)
    return knn

def intervaloConfianca(df):

    df_int_conf = pd.DataFrame()

    for x in range(len(df.columns)):
        column=df[df.columns[x]]
        desvio=column.std(axis=None)
        err=desvio/math.sqrt(len(column))
        intervalo=err*1.96
        print(df.columns[x],"\nmin",column.mean()-intervalo,"max",column.mean()+intervalo)
        df_int_conf[df.columns[x]] = [column.mean()-intervalo, column.mean()+intervalo]
    return df_int_conf

def feedback(music_param, face_param):
    choice = input('Você gostou desta recomendação musical? [S/N]: ')
    choice = choice.lower()

    while(choice != 's' and choice != 'n'):
        print('Sua resposta tem que ser "S" ou "N" ')

        choice = input('Você gostou desta recomendação musical? [S/N]: ')
        choice = choice.lower()

    if(choice == 's'):
        # guarda dados
        df = face_param
        df = df.join(music_param)
        df.to_csv('database\databaseTest.csv', mode='a', header=False)
        return True
    else:
        #faz algo ainda
        return False

def play_music(musicId):
    
    f = open('Temp\index.html', 'w')
    
    html_template = """
    <html>
    <head></head>
    <body>
    <iframe style="border-radius:12px" src="https://open.spotify.com/embed/track/{id}?utm_source=generator" width="100%" height="352" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>
    
    </body>
    </html>
    """.format(id=musicId)
    
    f.write(html_template)
    
    f.close()
    
    filename = 'file:///'+os.getcwd()+'/' + 'Temp\index.html'
    webbrowser.open_new_tab(filename)

def main():

    database = pd.read_csv("database/database2.csv")
    print(database)

    #Treinamento
    knn = training(database)

    img1 = 'img_database_r\Homem\Pessoa 04\Bravo_r.png'

    #Imagem ou Câmera?
    #pred = get_face_image(GetRosto())
    pred = get_face_image(img1)

    if pred != 0:
        new_data = get_face_param(pred)

    rosto = new_data[['age', 'gender', 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']]
    rosto['gender'] = rosto['gender'].replace('Man',0).replace('Woman',1)
    print(rosto)

    item_selected = rosto.iloc[[0]]
    item_selected = item_selected[rosto.columns] 

    # Determine the neighbors
    d, neighbors = knn.kneighbors(item_selected.values.reshape(1, -1))
    neighbors = neighbors[0][1::] 
    d = d[0][1::] 

    print(neighbors)
    
    neighborsMusics = pd.DataFrame()

    for x in neighbors:
        print(database.iloc[[x]][['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo','duration_ms']])
        neighborsMusics = neighborsMusics.append(database.iloc[[x]][['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo','duration_ms']])

    intervalo_params = intervaloConfianca(neighborsMusics)

    music_names = []
    artist_names = []
    artist_genres = []
    for y in neighbors:
        music_names.append(database.iloc[y]['music_name'])
        artist_names.append(database.iloc[y]['artist_name'])
        res = (database.iloc[y]['artist_genres']).strip('][').split(', ')
        for item in res:
            if(not(item.replace("'",'') in artist_genres)):
                artist_genres.append(item.replace("'",''))
        #artist_genres = [*artist_genres , *(database.iloc[y]['artist_genres'])]
        #print(artist_genres)

    print(intervalo_params)
    print(music_names)
    print(artist_names)
    print(artist_genres)

    #Spotify WebAPI

    musicaRecomendada = neighborsMusics


    musicaRecomendada = neighborsMusics.mean()[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]

    musicaRecomendada[['key']] = musicaRecomendada[['key']].values.round()
    musicaRecomendada[['mode']] = musicaRecomendada[['mode']].values.round()
    musicaRecomendada['duration_ms'] = 210000

    musicaRecomendada = musicaRecomendada.to_frame().T

    print(musicaRecomendada)

    #Recebe musicId
    play_music("2hl6q70unbviGo3g1R7uFx")

    feedback(musicaRecomendada, new_data)


main()