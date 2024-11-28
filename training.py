import datetime
import json

import torch
import torchvision
import torchvision.transforms as transforms
from early_stopping_pytorch import EarlyStopping
from torchvision.models import ResNet50_Weights

from config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
num_epochs = 10
batch_size = 64
learning_rate = 0.001

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_dataset = torchvision.datasets.ImageFolder(root=config['TRAINING_IMAGES_FOLDER'], transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Load the ResNet50 model
model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)

# Parallelize training across multiple GPUs
model = torch.nn.DataParallel(model)

in_features = model.module.fc.in_features
out_features = len(train_dataset.classes)
model.module.fc = torch.nn.Linear(in_features, out_features, bias=True)

# Set the model to run on the device
model = model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Initialize early stopping object
early_stopping = EarlyStopping(patience=7, verbose=True)

# Train the model...
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        # Move input and label tensors to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero out the optimizer
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

    # Print the loss for every epoch
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

    # Early stopping call
    early_stopping(loss.item(), model)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

print(f'Finished Training, Loss: {loss.item():.4f}')
now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
model_filename = config['OUTPUT_MODILE_FOLDER'] + "model_" + now + ".zip"
torch.save(model, model_filename)

mappings = { v: k for k,v in train_dataset.class_to_idx.items() }
with open(config['OUTPUT_MODILE_FOLDER'] + "mapping_" + now + ".json", 'w', encoding='utf-8') as f:
    json.dump(mappings, f, indent=2)
