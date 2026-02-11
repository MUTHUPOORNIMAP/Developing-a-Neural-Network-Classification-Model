# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="979" height="683" alt="image" src="https://github.com/user-attachments/assets/86c8bd0a-8e42-49d9-978f-6f81aec3426a" />


## DESIGN STEPS
### STEP 1: Load and Preprocess Data

Load the dataset, remove irrelevant columns (ID), handle missing values, encode categorical features using Label Encoding, and encode the target class (Segmentation).



### STEP 2: Feature Scaling and Data Split

Split the dataset into training and testing sets, then normalize the input features using StandardScaler for better neural network performance



### STEP 3: Convert Data to PyTorch Tensors

Convert the scaled training and testing data into PyTorch tensors and create DataLoader objects for batch-wise training and evaluation.



### STEP 4: Define the Neural Network Model

Design a feedforward neural network with multiple fully connected layers and ReLU activation functions, ending with an output layer for multi-class classification.



### STEP 5: Train the Model

Train the model using CrossEntropyLoss and Adam optimizer by performing forward propagation, loss calculation, backpropagation, and weight updates over multiple epochs.



### STEP 6: Evaluate and Predict

Evaluate the trained model on test data using accuracy, confusion matrix, and classification report, and perform prediction on a sample input.





## PROGRAM

### Name: Muthu Poornima P

### Register Number: 212224240099

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        super(PeopleClassifier, self).__init__()
        self.fc1=nn.Linear(input_size,32)
        self.fc2=nn.Linear(32,16)
        self.fc3=nn.Linear(16,8)
        self.fc4=nn.Linear(8,4)
        





    def forward(self, x):
      x=F.relu(self.fc1(x))
      x=F.relu(self.fc2(x))
      x=F.relu(self.fc3(x))
      x=self.fc4(x)
      return x


        
# Initialize the Model, Loss Function, and Optimizer

def train_model(model, train_loader, criterion, optimizer, epochs):
      model.train()
  for epoch in range(epochs):
    for inputs, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()




    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

```

### Dataset Information

<img width="1324" height="273" alt="image" src="https://github.com/user-attachments/assets/f69e7d72-7622-41df-8c27-3127f159622c" />


### OUTPUT


## Confusion Matrix

<img width="911" height="572" alt="image" src="https://github.com/user-attachments/assets/ef5a60be-34c4-418c-a525-9aed690dfe58" />


## Classification Report

<img width="724" height="445" alt="image" src="https://github.com/user-attachments/assets/42da4b30-35d1-4bdf-bd99-7f0f1483b465" />


### New Sample Data Prediction

<img width="473" height="94" alt="image" src="https://github.com/user-attachments/assets/5cf204c2-ab80-4dee-b133-c6e476ccaff9" />


## RESULT

This program has been executed successfully.
