import json
import logging
import os
from http import HTTPStatus

import requests
import spotipy
from PIL import Image
from spotipy.oauth2 import SpotifyOAuth

from albums import get_my_albums

SPOTIFY_SCOPE = 'user-library-read'
IMAGES_FOLDER = './images/'
LIMIT_IMAGES_FILE = 'my_vinyls.json'

def filter_images(image_dict):
    result = {}
    for id, images in image_dict.items():
        max_size = -1
        best_image = None
        for image in images:
            if image_size(image) > max_size:
                best_image = image
                max_size = image_size(image)
        if best_image:
            result[id] = best_image

    return result

def image_size(image):
    return image['height'] + image['width']

def fetch_image(image, filename, use_cache = True):
    full_path_name = IMAGES_FOLDER + filename
    if use_cache and os.path.isfile(full_path_name):
        try:
            with Image.open(full_path_name) as im:
                im.verify()
        except:
            pass
        else:
            return

    url = image['url']
    response = requests.get(url, timeout=10)
    if response.status_code == HTTPStatus.OK.value:
        with open(full_path_name, "wb") as f:
            f.write(response.content)

def get_limit_ids():
    result = []
    if LIMIT_IMAGES_FILE and os.path.isfile(LIMIT_IMAGES_FILE):
        with open(LIMIT_IMAGES_FILE, encoding='utf-8') as f:
            result = json.load(f)['id']
    return result

def image_filename(id):
    return id + ".jpg"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=SPOTIFY_SCOPE))

    my_albums = get_my_albums(sp)
    limit_ids = get_limit_ids()
    images = { album['id']: album['images'] for album in my_albums if
                    len(limit_ids) > 0 and album['id'] in limit_ids }

    logging.info("Going to download %d images", len(images))
    images = filter_images(images)
    for id, image in images.items():
        filename = image_filename(id)
        fetch_image(image, filename, False)
