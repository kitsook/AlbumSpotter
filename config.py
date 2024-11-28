config = {
    # cache file to store album data downloaded from Spotify
    'CACHE_FILE': 'my_albums.json',
    # keys of the album data that we want
    'KEYS_TO_GET': { 'id', 'total_tracks', 'href', 'name', 'uri', 'artists', 'images' },
    # folder to store downloaded cover arts
    'COVER_ARTS_FOLDER': './images/cover_arts/',
    # folder to store training images. sub-folder will be created for each album
    'TRAINING_IMAGES_FOLDER': './images/training/',
    # an optional json file with an "id" list that limit the albums used for training
    'LIMIT_IMAGES_FILE': 'my_vinyls.json',
    # 'LIMIT_IMAGES_FILE': None,
    # file name for the resized cover art for training
    'ORIG_IMAGE_NAME': 'orig.jpg',
    # image size for training
    'TRAINING_IMG_SIZE': (224, 224),
    # output model path
    'OUTPUT_MODEL_FOLDER': './models/',
}

