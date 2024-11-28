import glob
import json
import logging
import os
from http import HTTPStatus

import requests
from PIL import Image

from albums import get_my_albums
from spotify import sp

SPOTIFY_SCOPE = 'user-library-read'
IMAGES_FOLDER = './images/'
LIMIT_IMAGES_FILE = 'my_vinyls.json'

def prepare_cover_arts(sp, size = (224, 224), use_cache = True):
    _get_cover_arts(sp, use_cache)
    _resize_image_files(size)

def _get_cover_arts(sp, use_cache = True):
    my_albums = get_my_albums(sp, use_cache = use_cache)
    limit_ids = _get_limit_ids()
    images = { album['id']: album['images'] for album in my_albums if
                    len(limit_ids) > 0 and album['id'] in limit_ids }

    images = _filter_images(images)
    logging.info("Preparing %d images", len(images))
    for album_id, image in images.items():
        filename = _image_filename(album_id)
        _fetch_image(image, filename, use_cache)

def _filter_images(image_dict):
    result = {}
    for id, images in image_dict.items():
        max_size = -1
        best_image = None
        for image in images:
            if _image_size(image) > max_size:
                best_image = image
                max_size = _image_size(image)
        if best_image:
            result[id] = best_image

    return result

def _image_size(image):
    return image['height'] + image['width']

def _fetch_image(image, filename, use_cache = True):
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

def _get_limit_ids():
    result = []
    if LIMIT_IMAGES_FILE and os.path.isfile(LIMIT_IMAGES_FILE):
        with open(LIMIT_IMAGES_FILE, encoding='utf-8') as f:
            result = json.load(f)['id']
    return result

def _image_filename(id):
    return id + ".jpg"

def _resize_image_files(size = (224, 224)):
    for file in glob.glob(IMAGES_FOLDER + "*"):
        try:
            with Image.open(file) as im:
                if im.width != size[0] or im.height != size[1]:
                    im.thumbnail(size, Image.Resampling.LANCZOS)
                    im.save(file, "JPEG")
        except IOError:
            logging.error("Failed to process file %s", file)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    prepare_cover_arts(sp, True)
