import spotipy
from spotipy.oauth2 import SpotifyOAuth

SPOTIFY_SCOPE = 'user-library-read'
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=SPOTIFY_SCOPE))