from data import DigitsDataset
from model import ClassifierNN
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import torch
import pdb
import numpy as np
import matplotlib.pyplot as plt

df = DigitsDataset(file_name='ex3data1.mat')
X_train = df.X_train
y_train = df.y_train
"""
i=5
img = np.reshape(df.X_train[i], (20, 20))
plt.imshow(img)
plt.show()
"""
### Training the model ###

mymodel = ClassifierNN(input_features=400, hidden_layer1=25, hidden_layer2=30, output_features=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mymodel.parameters(), lr=0.01)

epochs = 100
losses = []

for i in range(epochs):
    y_pred = mymodel.forward(X_train)
    y_train = y_train.to(dtype=torch.long)
    loss = criterion(y_pred, y_train)
    losses.append(loss)
    print(f'epoch: {i:2}  loss: {loss.item():10.8f}')
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

### Validating and testing the model ###

X_test = df.X_test
y_test = df.y_test

preds = []
with torch.no_grad():
    for val in X_test:
        y_hat = mymodel.forward(val)
        preds.append(y_hat.argmax().item())

y_test = np.array(y_test)
preds = np.array(preds)
correct = (y_test == preds)
correct = correct.astype(int)

erreur = correct.sum()/len(correct)
print("The error is :", erreur)

pdb.set_trace()