import json

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

from albums import get_my_albums
from config import config

model_timestamp = '20241128185438'
model_file = config['OUTPUT_MODEL_FOLDER'] + "model_" + model_timestamp + ".pt"
mapping_file = config['OUTPUT_MODEL_FOLDER'] + "mapping_" + model_timestamp + ".json"

mappings = {}
with open(mapping_file, encoding='utf-8') as f:
    mappings = json.load(f)

my_albums = get_my_albums(use_cache = True)
albums_dict = { album['id']: album for album in my_albums }

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torch.load(model_file, weights_only=False)
model.eval()

transform = transforms.Compose([
        transforms.Resize(config['TRAINING_IMG_SIZE']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

capture = cv2.VideoCapture(0)

while True:
    input("Press Enter to capture an image...")
    # discard first few frames. some webcams give dark images for initial frames
    for _ in range(10):
        retval, frame = capture.read()

    if retval:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        image.save("./inference_image.jpg")

        output = model(transform(image).unsqueeze_(0))
        output = output.to(device)
        index = output.data.numpy().argmax()
        confidences = output.data.numpy().squeeze()

        predicted_album_id = mappings[str(index.item())]
        predicted_album_name = albums_dict[predicted_album_id]['name']
        confidence = confidences[index.item()]
        print("I am guessing the album is %s with confidence %0.4f" % (predicted_album_name, confidence))
    else:
        print("No frame captured")
        break
