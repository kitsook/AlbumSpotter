import json
import logging
import os.path

import spotipy
from spotipy.oauth2 import SpotifyOAuth

DUMP_TO = 'my_albums.json'
KEYS_TO_GET = { 'id', 'total_tracks', 'href', 'name', 'uri', 'artists', 'images' }
SPOTIFY_SCOPE = 'user-library-read'

def get_my_albums(sp, keys = KEYS_TO_GET, use_cache = True):
    result = []

    if use_cache:
        result = load_from_cache()
    if len(result) > 0:
        return result

    limit = 50
    offset = 0
    while True:
        the_albums = sp.current_user_saved_albums(limit, offset)['items']
        if len(the_albums) == 0:
            break

        result.extend([{key: album['album'][key] for key in keys} for album in the_albums])
        offset += len(the_albums)

    return result

def load_from_cache():
    if os.path.isfile(DUMP_TO):
        with open(DUMP_TO, encoding='utf-8') as f:
            return json.load(f)
    return []

def save_cache(my_albums):
    with open(DUMP_TO, 'w', encoding='utf-8') as f:
        json.dump(my_albums, f, indent=2)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=SPOTIFY_SCOPE))

    my_albums = get_my_albums(sp, use_cache=False)
    logging.info("Dumped %d albums", len(my_albums))
    save_cache(my_albums)
