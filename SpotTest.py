import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="094d8890d57f4293b5e14500a347e8f0",
                                                           client_secret="97f0921b9d2b4551b700ec347b56b301"))

track= 'Lonely World'

track_id = sp.search(q=track, type='track',limit=1)

print(track_id['tracks']['items'][0]['id'])
#print(track_id)

