import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import accuracy_score

import time

#28x28x3

# Define a CNN model using PyTorch
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, padding=1)
        # print(self.conv1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(24, 49, kernel_size=3, padding=1)
        # print(self.conv2)
        
        self.conv3 = nn.Conv2d(49, 147, kernel_size=3, padding=1)

        self.pool2 = nn.MaxPool2d(kernel_size=1, stride=2)

        self.fc1 = nn.Linear(147 * 4 * 4, 1176)
        # print(self.fc1)
        self.fc2 = nn.Linear(1176, 196)
        # print(self.fc2)
        self.fc3 = nn.Linear(196, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool(x)

        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)

        x = nn.functional.relu(self.conv3(x))
        x = self.pool2(x)

        x = x.reshape(x.shape[0], -1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# Reshape data for PyTorch input
def preprocess_data(X):
    return torch.tensor(X.reshape(-1, 3, 28, 28), dtype=torch.float32)


# Function to learn a CNN model using PyTorch
def learn(X, y):
    X_processed = preprocess_data(X)
    y_tensor = torch.tensor(y, dtype=torch.long)

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dataset = TensorDataset(X_processed, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    #Set model to training mode
    model.train()
    start_time = time.time()
    for epoch in range(20):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if (epoch % 5) == 0:
            cur = time.time()
            elapsed = cur - start_time
            print("Completed epoch #" + str(epoch) + " | elapsed time: " + str(elapsed))
            

    return model

# Function to classify using the learned CNN model with PyTorch
def classify(Xtest, model):
    Xtest_processed = preprocess_data(Xtest)
    model.eval()
    with torch.no_grad():
        outputs = model(Xtest_processed)
    _, yhat = torch.max(outputs, 1)
    return yhat.numpy()

# Preprocess data for PyTorch (convert to PyTorch tensors)
def preprocess_data2(X, y):
    X_tensor = torch.tensor(X, dtype=torch.float32)  # Convert features to tensor
    y_tensor = torch.tensor(y, dtype=torch.long)    # Convert labels to tensor
    return X_tensor, y_tensor

# Define a simple model using PyTorch
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28 * 3, 128)  # Input size: 28*28*3, Output size: 128
        self.fc2 = nn.Linear(128, 64)           # Hidden layer size: 128, Output size: 64
        self.fc3 = nn.Linear(64, 10)            # Output size: 10 (classes)

    def forward(self, x):
        x = x.view(-1, 28 * 28 * 3)  # Flatten the input images
        x = torch.relu(self.fc1(x))  # Pass through first hidden layer with ReLU activation
        x = torch.relu(self.fc2(x))  # Pass through second hidden layer with ReLU activation
        x = self.fc3(x)              # Output layer (no activation for now)
        return x


# Function to learn a SimpleModel model using PyTorch
def learnSimple(X, y):
    X_train_tensor, y_train_tensor = preprocess_data2(X, y)

    model = SimpleModel()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 20
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model



# Function to classify using the learned CNN model with PyTorch
def classifySimple(Xtest, model):
    Xtest_processed = preprocess_data(Xtest)
    model.eval()
    with torch.no_grad():
        outputs = model(Xtest_processed)
    _, yhat = torch.max(outputs, 1)
    return yhat.numpy()


#Read file
train_data = pd.read_csv('A4train.csv', header=None)
val_data = pd.read_csv('A4val.csv', header=None)

#Numpy Array
X_train = train_data.iloc[:, 1:].values
# print(X_train.shape)
y_train = train_data.iloc[:, 0].values
X_val = val_data.iloc[:, 1:].values
y_val = val_data.iloc[:, 0].values

#Processed Data
X_train_tensor, y_train_tensor = preprocess_data2(X_train, y_train)
X_val_tensor, y_val_tensor = preprocess_data2(X_val, y_val)


def SimpleModelTest():
    model = learnSimple(X_train, y_train)
    yhat = classifySimple(X_val, model)

    #Get accuracy
    accuracy = accuracy_score(y_val_tensor.numpy(), yhat)
    print(yhat.shape)
    print(f"Validation Accuracy: {accuracy}")


def CNNTest():
    model = learn(X_train, y_train)
    yhat = classify(X_val, model)

    #Get accuracy
    accuracy = accuracy_score(y_val_tensor.numpy(), yhat)
    print(yhat.shape)
    print(f"Validation Accuracy: {accuracy}")


# Function to visualize an image
def plotImg(x):
    img = x.reshape((84, 28))
    plt.imshow(img, cmap='gray')
    plt.show()

# SimpleModelTest()
# print("smm finished")

# X_train[0].shape

CNNTest()
print("cnn finished")

sample_image = X_train[0]
# Visualize the sample image
plotImg(sample_image)