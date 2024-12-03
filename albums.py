import json
import logging
import os.path

import spotipy
from spotipy.oauth2 import SpotifyOAuth

from config import config


_SPOTIFY_SCOPE = 'user-library-read'

def get_my_albums(keys = config['KEYS_TO_GET'], use_cache = True):
    result = []
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=_SPOTIFY_SCOPE))

    if use_cache:
        result = _load_from_cache()
    if len(result) > 0:
        return result

    limit = 50
    offset = 0
    while True:
        the_albums = sp.current_user_saved_albums(limit, offset)['items']
        if len(the_albums) == 0:
            break

        if keys and len(keys) > 0:
            result.extend([{key: album['album'][key] for key in keys} for album in the_albums])
        else:
            result.extend(the_albums)
        offset += len(the_albums)

    return result

def _load_from_cache():
    if os.path.isfile(config['CACHE_FILE']):
        with open(config['CACHE_FILE'], encoding='utf-8') as f:
            return json.load(f)
    return []

def _save_cache(my_albums):
    with open(config['CACHE_FILE'], 'w', encoding='utf-8') as f:
        json.dump(my_albums, f, indent=2)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    my_albums = get_my_albums(use_cache=False)
    logging.info("Dumped %d albums", len(my_albums))
    _save_cache(my_albums)
