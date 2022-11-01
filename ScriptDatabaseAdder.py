from math import nan
from scipy.sparse import data
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="094d8890d57f4293b5e14500a347e8f0",
                                                           client_secret="97f0921b9d2b4551b700ec347b56b301"))


def main():

    database = pd.read_csv("database\database.csv")

    database['music_name'] = None
    database['artist_name'] = None
    database['artist_id'] = None
    database['artist_genres'] = None

    for x in range(len(database)) :
        trackId = database.iloc[[x]]['id'].values[0]
        track = sp.track(track_id=trackId)
        artist_info = sp.artist(track['album']['artists'][0]['uri'])
        
        database['music_name'].iloc[[x]] = track['name']
        database['artist_name'].iloc[[x]] = track['album']['artists'][0]['name']
        database['artist_id'].iloc[[x]] = track['album']['artists'][0]['id']
        database['artist_genres'].iloc[[x]] = [artist_info['genres']]
        
    print(database)
    database.to_csv('database\database3.csv', mode='w', header=False)


main()