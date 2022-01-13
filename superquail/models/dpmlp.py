# TODO: Cleanup and make class

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from load_data import load_data
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

from opacus import PrivacyEngine, utils, autograd_grad_sample


class MLP(nn.Module):
    def __init__(self, input_size, classes, hidden_layer_sizes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_layer_sizes[0])
        self.fc2 = nn.Linear(hidden_layer_sizes[0],hidden_layer_sizes[1])
        self.fc3 = nn.Linear(hidden_layer_sizes[1],classes)
        
    def forward(self, x):
        x = self.fc3(F.leaky_relu(self.fc2(F.leaky_relu(self.fc1(x), 0.2)), 0.2))
        return x

### Load dataset and split

loaded_datasets = load_data()
data = loaded_datasets['mushroom']["data"]
X = data.loc[:, data.columns != loaded_datasets['mushroom']["target"]]
y = data.loc[:, data.columns == loaded_datasets['mushroom']["target"]]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
x_train_numpy = scaler.fit_transform(x_train)
x_test_numpy = scaler.transform(x_test)
x_train = pd.DataFrame(x_train_numpy, columns = x_train.columns)
x_test = pd.DataFrame(x_test_numpy, columns = x_test.columns)

net = MLP(X.shape[1], len(np.unique(y)), (50,20))

sample_size=len(x_train)
batch_size=min(250, len(x_train))

import torch.optim as optim
optimizer = optim.Adam(net.parameters(), lr=.02, betas=(0.5, 0.9))
criterion = nn.CrossEntropyLoss()

privacy_engine = PrivacyEngine(
    net,
    batch_size,
    sample_size,
    alphas= [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
    noise_multiplier=3.0,
    max_grad_norm=1.0,
    clip_per_layer=True
)
privacy_engine.attach(optimizer)

target_delta = 1/x_train.shape[0]
print(target_delta)
for epoch in range(500):
    for i in range(int(len(x_train)/batch_size) + 1):
        data2 = x_train.iloc[i*batch_size:i*batch_size+batch_size, :]
        labels = y_train.iloc[i*batch_size:i*batch_size+batch_size, :]
        if len(labels) < batch_size:
            break
        X, Y = Variable(torch.FloatTensor([data2.to_numpy()]), requires_grad=True), Variable(torch.FloatTensor([labels.to_numpy()]), requires_grad=False)
        optimizer.zero_grad()
        y_pred = net(X)
        output = criterion(y_pred.squeeze(), Y.squeeze().long())
        output.backward()
        optimizer.step()
        
    if (epoch % 3 == 0.0):
        print("Epoch {} - loss: {}".format(epoch, output))
        epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(target_delta)
        print ('epsilon is {e}, alpha is {a}'.format(e=epsilon, a = best_alpha))
        if 3.0 < epsilon:
            break
        
predictions = torch.argmax(net(Variable(torch.FloatTensor([x_test.to_numpy()]), requires_grad=True))[0],1)
print(predictions)
from sklearn.metrics import accuracy_score
print('MLP Acc:' + str(accuracy_score(predictions.numpy(), y_test.to_numpy())))