import logging
import os
import random
import uuid

import torch
import torchvision.transforms.v2 as transforms
from PIL import Image

from config import config
from images import prepare_images
from spotify import sp


def transform_training_images():
    torch.manual_seed(42)
    random.seed(1337)

    logging.info("Transforming and augmenting images")
    for subfolder in next(os.walk(config['TRAINING_IMAGES_FOLDER']))[1]:
        album_folder = config['TRAINING_IMAGES_FOLDER'] + subfolder + "/"
        with Image.open(album_folder + config['ORIG_IMAGE_NAME']) as orig_img:

            perspective_transformer = transforms.RandomPerspective(distortion_scale=0.6, p=1.0, fill=random.randint(0, 255))
            perspective_imgs = [perspective_transformer(orig_img) for _ in range(5)]
            _save_training_images(album_folder, perspective_imgs)

            affine_transfomer = transforms.RandomAffine(degrees=20, translate=(0.1, 0.3), scale=(0.5, 0.75), fill=random.randint(0, 255))
            affine_imgs = [affine_transfomer(orig_img) for _ in range(8)]
            _save_training_images(album_folder, affine_imgs)

            blurrer = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))
            blurred_imgs = [blurrer(orig_img) for _ in range(3)]
            _save_training_images(album_folder, blurred_imgs)

            crop_transformer = transforms.RandomCrop(size=list(reversed([int(x * 0.8) for x in orig_img.size])))
            crop_imgs = [crop_transformer(orig_img) for _ in range(5)]
            _save_training_images(album_folder, crop_imgs)

            jitter = transforms.ColorJitter(brightness=.5, hue=.3)
            jittered_imgs = [jitter(orig_img) for _ in range(5)]
            _save_training_images(album_folder, jittered_imgs)

def _save_training_images(album_folder, images):
    for img in images:
        new_img_name = str(uuid.uuid4()) + ".jpg"
        if img.width != config['TRAINING_IMG_SIZE'][0] or img.height != config['TRAINING_IMG_SIZE'][1]:
            img.thumbnail(config['TRAINING_IMG_SIZE'], Image.Resampling.LANCZOS)
        img.save(album_folder + new_img_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    prepare_images(use_cache = True)
    transform_training_images()
