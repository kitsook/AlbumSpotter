import json

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

from config import config

model_timestamp = '20241128014712'
model_file = config['OUTPUT_MODILE_FOLDER'] + "model_" + model_timestamp + ".zip"
mapping_file = config['OUTPUT_MODILE_FOLDER'] + "mapping_" + model_timestamp + ".json"

mappings = {}
with open(mapping_file, encoding='utf-8') as f:
    mappings = json.load(f)

my_albums = {}
with open(config['CACHE_FILE'], encoding='utf-8') as f:
    my_albums = json.load(f)

albums_dict = { album['id']: album for album in my_albums }

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torch.load(model_file, weights_only=False)
model.eval()

transform = transforms.Compose([
        transforms.Resize(config['TRAINING_IMG_SIZE']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# cv2.namedWindow("camera", 1)
capture = cv2.VideoCapture(0)

while True:
    input("Press Enter to capture an image...")
    # discard first few frames
    for _ in range(10):
        retval, frame = capture.read()

    if retval:
        # cv2.imshow("camera", img)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        image.save("./inference_image.jpg")

        x = transform(image)
        x.unsqueeze_(0)
        output = model(x)
        output = output.to(device)
        index = output.data.numpy().argmax()
        confidences = output.data.numpy().squeeze()
        print(confidences)
        print(index)

        predicted_album_id = mappings[str(index.item())]
        print("I am guessing the album is %s with confidence " % albums_dict[predicted_album_id]['name'])