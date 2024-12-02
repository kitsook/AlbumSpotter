import datetime
import json

import torch
import torchvision
from early_stopping_pytorch import EarlyStopping
from torchinfo import summary
from torchvision.models import ResNet50_Weights

from config import config

# hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
EARLY_STOPPING_PATIENCE = 4

# freezing first few layers in ResNet50 for fine tuning
FREEZING_LAYERS = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = config['MODEL_TRANSFORMS']

train_dataset = torchvision.datasets.ImageFolder(root=config['TRAINING_IMAGES_FOLDER'], transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# load the ResNet50 model
model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)

# modify the fully connected layer
in_features = model.fc.in_features
num_classes = len(train_dataset.classes)
model.fc = torch.nn.Linear(in_features, num_classes, bias=True)

# freeze layers
for name, layer in model.named_children():
    if name in FREEZING_LAYERS:
        for param in layer.parameters():
            param.requires_grad = False

# print(model)
# summary(model, input_size = (1, 3, 224, 224))

# set the model to run on the device
model = model.to(device)

# define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

# initialize early stopping object
early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=True)

# train the model...
for epoch in range(NUM_EPOCHS):
    print()
    print('Epoch {}/{}'.format(epoch+1, NUM_EPOCHS))
    print('-' * 11)

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        # move input and label tensors to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero out the optimizer
        optimizer.zero_grad()

        # forward pass
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # backward pass
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    # print the loss for every epoch
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

    # check for early stopping. using training loss instead of validation loss
    early_stopping(epoch_loss, model)
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
