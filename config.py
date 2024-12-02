import torch
import torchvision.transforms.v2 as transforms

_IMG_SIZE = (224, 224)

config = {
    # cache file to store album data downloaded from Spotify
    'CACHE_FILE': 'my_albums.json',
    # keys from the album data that we want
    'KEYS_TO_GET': { 'id', 'total_tracks', 'href', 'name', 'uri', 'artists', 'images', 'external_urls' },
    # folder to store downloaded cover arts
    'COVER_ARTS_FOLDER': './images/cover_arts/',
    # folder to store training images. sub-folder will be created for each album
    'TRAINING_IMAGES_FOLDER': './images/training/',
    # folder to store validation images
    'VALIDATION_IMAGES_FOLDER': './images/validation/',
    # an optional json file with an "id" list that limit the albums used for training
    # 'LIMIT_IMAGES_FILE': 'my_vinyls.json',
    'LIMIT_IMAGES_FILE': None,
    # file name of the cover art for training
    'ORIG_IMAGE_NAME': 'orig.jpg',
    # image size for training
    'TRAINING_IMG_SIZE': _IMG_SIZE,
    # output model path
    'OUTPUT_MODEL_FOLDER': './models/',
    'MODEL_TRANSFORMS': transforms.Compose([
        transforms.ToImage(),
        transforms.Resize(_IMG_SIZE),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}
