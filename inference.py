import json

import cv2
import torch
from PIL import Image

from albums import get_my_albums
from config import config

# model output level required to be considered as a valid prediction
PREDICTION_THRESHOLD = 0.
MODEL_TIMESTAMP = '20241201203107'

model_file = config['OUTPUT_MODEL_FOLDER'] + "model_" + MODEL_TIMESTAMP + ".pt"
mapping_file = config['OUTPUT_MODEL_FOLDER'] + "mapping_" + MODEL_TIMESTAMP + ".json"

mappings = {}
with open(mapping_file, encoding='utf-8') as f:
    mappings = json.load(f)

my_albums = get_my_albums(use_cache = True)
albums_dict = { album['id']: album for album in my_albums }

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torch.load(model_file, weights_only=False)
model.eval()

transform = config['MODEL_TRANSFORMS']

capture = cv2.VideoCapture(0)

while True:
    input("Press Enter to capture an image...")
    # discard first few frames. some webcams return dark images for first few reads
    for _ in range(10):
        retval, frame = capture.read()

    if retval:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        # save the input image for debugging
        image.save("./inference_image.jpg")

        output = model(transform(image).unsqueeze_(0)).to(device)
        index = output.data.numpy().argmax().item()
        confidences = output.data.numpy().squeeze()

        predicted_album_id = mappings[str(index)]
        predicted_album_name = albums_dict[predicted_album_id]['name']
        predicted_album_spotify = albums_dict[predicted_album_id]['external_urls']['spotify']
        confidence = confidences[index]
        if confidence > PREDICTION_THRESHOLD:
            print("I am guessing the album is '%s' with a confidence of %0.4f. You can play the album on Spotify at %s" %
                  (predicted_album_name, confidence, predicted_album_spotify))
        else:
            print("I don't recognize the album (confidence %0.4f)" % confidence)
    else:
        print("No frame captured")
        break

