import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------
# 1. Load Dataset (Iris)
# ------------------------------
iris = load_iris()
X = iris.data   # features
y = iris.target # labels (0, 1, 2)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to torch tensors
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# ------------------------------
# 2. Define Deep Neural Network
# ------------------------------
class DeepNN(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, num_classes):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)  # input → hidden1
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)     # hidden1 → hidden2
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2, num_classes) # hidden2 → output

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

# ------------------------------
# 3. Initialize Model
# ------------------------------
input_size = X.shape[1]   # 4 features
hidden1, hidden2 = 16, 8
num_classes = 3           # iris has 3 classes
model = DeepNN(input_size, hidden1, hidden2, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ------------------------------
# 4. Training
# ------------------------------
epochs = 100
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# ------------------------------
# 5. Evaluation
# ------------------------------
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)

print("\nClassification Accuracy on Test Data: {:.2f}%".format(accuracy * 100))

/*
Epoch [10/100], Loss: 1.0312
Epoch [20/100], Loss: 0.8220
Epoch [30/100], Loss: 0.6012
Epoch [40/100], Loss: 0.4739
Epoch [50/100], Loss: 0.3485
Epoch [60/100], Loss: 0.2633
Epoch [70/100], Loss: 0.2148
Epoch [80/100], Loss: 0.1813
Epoch [90/100], Loss: 0.1578
Epoch [100/100], Loss: 0.1397

Classification Accuracy on Test Data: 100.00%
*/