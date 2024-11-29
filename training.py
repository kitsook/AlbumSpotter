import datetime
import json

import torch
import torchvision
import torchvision.transforms.v2 as transforms
from early_stopping_pytorch import EarlyStopping
from torchvision.models import ResNet50_Weights

from config import config

# hyperparameters
NUM_EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToImage(),
    transforms.Resize(config['TRAINING_IMG_SIZE']),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_dataset = torchvision.datasets.ImageFolder(root=config['TRAINING_IMAGES_FOLDER'], transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# load the ResNet50 model
model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)

# parallelize training across multiple GPUs
model = torch.nn.DataParallel(model)

# modify the fully connected layer
in_features = model.module.fc.in_features
out_features = len(train_dataset.classes)
model.module.fc = torch.nn.Linear(in_features, out_features, bias=True)

# set the model to run on the device
model = model.to(device)

# define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# initialize early stopping object
early_stopping = EarlyStopping(patience=7, verbose=True)

# train the model...
for epoch in range(NUM_EPOCHS):
    for inputs, labels in train_loader:
        # move input and label tensors to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero out the optimizer
        optimizer.zero_grad()

        # forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # backward pass
        loss.backward()
        optimizer.step()

    # print the loss for every epoch
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.4f}')

    # check for early stopping
    early_stopping(loss.item(), model)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

# load the last checkpoint with the best model
model.load_state_dict(torch.load('checkpoint.pt', weights_only=True))

# save model
now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
model_filename = config['OUTPUT_MODEL_FOLDER'] + "model_" + now + ".pt"
torch.save(model, model_filename)

# save mapping of album id to index
mappings = { v: k for k,v in train_dataset.class_to_idx.items() }
with open(config['OUTPUT_MODEL_FOLDER'] + "mapping_" + now + ".json", 'w', encoding='utf-8') as f:
    json.dump(mappings, f, indent=2)
