import json
import os
import re

import torch
from PIL import Image

from config import config

def calc_accuracy(model_timestamp):
    model_file = config['OUTPUT_MODEL_FOLDER'] + "model_" + model_timestamp + ".pt"
    mapping_file = config['OUTPUT_MODEL_FOLDER'] + "mapping_" + model_timestamp + ".json"

    mappings = {}
    with open(mapping_file, encoding='utf-8') as f:
        mappings = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_file, weights_only=False)
    model.eval()

    transform = config['MODEL_TRANSFORMS']

    total = 0
    correct = 0

    subdirs = [ (f.name, f.path) for f in
                os.scandir(config['TESTING_IMAGES_FOLDER']) if f.is_dir() ]
    for album_id, subdir in subdirs:
        for file in os.listdir(subdir):
            image = Image.open(os.path.join(subdir, file))

            output = model(transform(image).unsqueeze_(0)).to(device)
            index = output.data.numpy().argmax().item()
            predicted_album_id = mappings[str(index)]

            if predicted_album_id == album_id:
                correct += 1
            total += 1

    accuracy = float(correct) / total
    print(f"[{model_file}] Accuracy: {correct}/{total} = {accuracy:.4f}")


def _find_model_timestamps():
    result = []
    for file in os.listdir(config['OUTPUT_MODEL_FOLDER']):
        if file.endswith(".pt"):
            tokens = re.split(r'[._]', file)
            result.append(tokens[1])
    return result

if __name__ == "__main__":
    for timestamp in _find_model_timestamps():
        calc_accuracy(timestamp)
