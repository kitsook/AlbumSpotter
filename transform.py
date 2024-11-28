import logging
import os
import random
import uuid

import torch
from PIL import Image
from torchvision.transforms import v2

from config import config
from images import prepare_images
from spotify import sp


def transform_training_images():
    torch.manual_seed(42)
    random.seed(1337)

    for subfolder in next(os.walk(config['TRAINING_IMAGES_FOLDER']))[1]:
        album_folder = config['TRAINING_IMAGES_FOLDER'] + subfolder + "/"
        with Image.open(album_folder + config['ORIG_IMAGE_NAME']) as orig_img:
            padded_imgs = [v2.Pad(padding=padding, fill=random.randint(0, 255))(orig_img) for padding in list(range(5, 50, 10))]
            _save_training_images(album_folder, padded_imgs)

            perspective_transformer = v2.RandomPerspective(distortion_scale=0.6, p=1.0, fill=random.randint(0, 255))
            perspective_imgs = [perspective_transformer(orig_img) for _ in range(10)]
            _save_training_images(album_folder, perspective_imgs)

            affine_transfomer = v2.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75), fill=random.randint(0, 255))
            affine_imgs = [affine_transfomer(orig_img) for _ in range(5)]
            _save_training_images(album_folder, affine_imgs)

            blurrer = v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))
            blurred_imgs = [blurrer(orig_img) for _ in range(5)]
            _save_training_images(album_folder, blurred_imgs)

            crop_transformer = v2.RandomResizedCrop(size=config['TRAINING_IMG_SIZE'])
            crop_imgs = [crop_transformer(orig_img) for _ in range(10)]
            _save_training_images(album_folder, crop_imgs)


def _save_training_images(album_folder, images):
    for img in images:
        new_img_name = str(uuid.uuid4()) + ".jpg"
        if img.width != config['TRAINING_IMG_SIZE'][0] or img.height != config['TRAINING_IMG_SIZE'][1]:
            img.thumbnail(config['TRAINING_IMG_SIZE'], Image.Resampling.LANCZOS)
        img.save(album_folder + new_img_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    prepare_images(use_cache = False)
    transform_training_images()

