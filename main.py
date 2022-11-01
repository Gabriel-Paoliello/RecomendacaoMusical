import cv2
import numpy as np
import os
from deepface import DeepFace
import pandas as pd
from genericpath import exists
from pandas import DataFrame
from sklearn.neighbors import NearestNeighbors
import math
import requests
import webbrowser
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json


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

def feedback(new_data_param):
    choice = input('Você gostou desta recomendação musical? [S/N]: ')
    choice = choice.lower()

    while(choice != 's' and choice != 'n'):
        print('Sua resposta tem que ser "S" ou "N" ')

        choice = input('Você gostou desta recomendação musical? [S/N]: ')
        choice = choice.lower()

    if(choice == 's'):
        # guarda dados
        new_data_param.to_csv('database\databaseTest.csv', mode='a', header=False)
        return True
    else:
        #faz algo ainda
        return False

def listToString(list):
    finalString=""
    for x in range(len(list)):
        if(x == len(list)-1):
            finalString=finalString+list[x]
            return finalString
        finalString=list[x]+","+finalString
    return finalString


def play_music(musicId):
    
    f = open('Temp\index.html', 'w')
    
    html_template = """
    <html>
    <head>
    <title>Recomendação Musical</title>
    </head>
    <body>
    <iframe style="border-radius:12px" src="https://open.spotify.com/embed/track/{id}?utm_source=generator" width="100%" height="352" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>
    
    </body>
    </html>
    """.format(id=musicId)
    
    f.write(html_template)
    
    f.close()
    
    filename = 'file:///'+os.getcwd()+'/' + 'Temp\index.html'
    webbrowser.open_new_tab(filename)

def post_music_request(intervalo_params, music_ids):

    int_min = intervalo_params.loc[0]
    int_max = intervalo_params.loc[1]

    url = 'http://localhost:5000/recommend/api/v1/get-recommendation'
    myobj = {
        "limit": "2",
        #"market":"ES",
        #"seed_artists": artist_ids,
        #"seed_genres": artist_genres,
        "seed_tracks": music_ids,
        #"min_danceability": str(int_min['danceability']),
        #"max_danceability": str(int_max['danceability']),
        #"min_energy": str(int_min['energy']),
        #"max_energy": str(int_max['energy']),
        #"min_acousticness": str(int_min['acousticness']),
        #"max_acousticness": str(int_max['acousticness']),
        #"min_instrumentalness": str(int_min['instrumentalness']),
        #"max_instrumentalness": str(int_max['instrumentalness']),
        #"min_loudness": str(int_min['loudness']),
        #"max_loudness": str(int_max['loudness']),
        #"min_speechiness": str(int_min['speechiness']),
        #"max_speechiness": str(int_max['speechiness']),
        #"min_liveness": str(int_min['liveness']),
        #"max_liveness": str(int_max['liveness']),
        #"max_valence": str(int_max['valence']),
        #"min_valence": str(int_min['valence']),
        #"max_tempo": str(int_max['tempo']),
        #"min_tempo": str(int_min['tempo'])
    }

    print(myobj)

    x = requests.post(url = url, json = myobj)
    return x

def main():

    audio_features = ["danceability","energy","key","loudness","mode","speechiness","acousticness","instrumentalness","liveness","valence","tempo","type","id","uri","track_href","analysis_url","duration_ms","time_signature"]

    #Pegar Audio_features
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="094d8890d57f4293b5e14500a347e8f0",client_secret="97f0921b9d2b4551b700ec347b56b301"))

    database = pd.read_csv("database/databaseTest.csv")

    #Treinamento
    knn = training(database)

    img1 = 'img_database_r\Homem\Pessoa 04\Bravo_r.png'

    #Imagem ou Câmera?
    pred = get_face_image(GetRosto())
    #pred = get_face_image(img1)

    if pred != 0:
        new_data = get_face_param(pred)

    rosto = new_data[['age', 'gender', 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']]
    rosto['gender'] = rosto['gender'].replace('Man',0).replace('Woman',1)

    item_selected = rosto.iloc[[0]]
    item_selected = item_selected[rosto.columns] 

    # Determine the neighbors
    d, neighbors = knn.kneighbors(item_selected.values.reshape(1, -1))
    neighbors = neighbors[0][1::] 
    d = d[0][1::] 

    print(neighbors)
    
    neighborsMusics = pd.DataFrame()

    for x in neighbors:
        print(database.iloc[[x]][['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']])
        neighborsMusics = neighborsMusics.append(database.iloc[[x]][['danceability', 'energy', 'loudness', 'speechiness','acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']])

    intervalo_params = intervaloConfianca(neighborsMusics)

    music_ids = []
    for y in neighbors:
        music_ids.append(database.iloc[y]['id'])
        
    str_music_ids = listToString(music_ids)

    #print(intervalo_params)
    #print(str_music_ids)
    #print(str_artist_ids)
    #print(str_artist_genres)

    #Spotify WebAPI
    musicaRecomendadas = post_music_request(intervalo_params, str_music_ids)
    
    idMusica = musicaRecomendadas.json()[0]["id"]

    #Recebe musicId
    play_music(idMusica)

    novo_dado = rosto

    for data in audio_features:
        novo_dado[data] = sp.audio_features("spotify:track:" + idMusica)[0][data]

    novo_dado['gender'] = novo_dado['gender'].replace(0,'Man').replace(1,'Woman')

    feedback(novo_dado)


while(True):
    if(input("Podemos iniciar? [S/N]: ").lower() == 's'):
        main()
    else:
        break