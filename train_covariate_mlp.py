# %%
from sklearn.utils import compute_class_weight
import torch
from torch.nn import Linear
import torch.nn.functional as F

import logging
import time
import sys
import random
import argparse
import pickle
from pathlib import Path

import torch
import numpy as np

from utils.data_processing import get_data_node_classification
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.functional import one_hot

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from imblearn.under_sampling import RandomUnderSampler

# %%

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('MLP supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=100, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_epoch', type=int, default=10, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--use_validation', action='store_true',
                    help='Whether to use a validation set')

# try:
#   args = parser.parse_args()
# except:
#   parser.print_help()
#   sys.exit(0)
# 
# BATCH_SIZE = args.bs
# NUM_EPOCH = args.n_epoch
# DROP_OUT = args.drop_out
# GPU = args.gpu
# DATA = args.data
# NUM_LAYER = args.n_layer
# LEARNING_RATE = args.lr

# Path("./saved_models/").mkdir(parents=True, exist_ok=True)
# Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
# MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}' + '\
#   node-classification.pth'
# get_checkpoint_path = lambda \
#     epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}' + '\
#   mlp-node-classification.pth'
args = 0

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

# %%
features_df = pd.read_csv("./data/vre_mlp_features.csv", index_col=0)
features_df = features_df[features_df["node type"] == "PATIENT"] # drop features about non-patients (others only used in TGN)

# mlp patient risks dataframe encodes the risk as a date after which the patient has this risk (or is contaminated)
# the youngest date is the end of the timespan of the dataset
labels_df = pd.read_csv("./data/vre_mlp_patient_risks.csv", dtype={"Patient ID": str}, index_col=0)
all_df = pd.merge(features_df, labels_df, how="left", left_on="node id", right_on="Patient ID")
all_df["date"] = pd.to_datetime(all_df["Risk Date"]).dt.date
max_date = all_df["date"].max()
all_df.loc[all_df["date"] < max_date,"label"] = 1
all_df.loc[all_df["date"] == max_date,"label"] = 0


X = all_df[["age", "gender", "surgery qty", "is_icu"]]
X = pd.DataFrame(MinMaxScaler().fit_transform(X.values), columns=X.columns, index=X.index).to_numpy()
y = all_df[["label"]].values.astype(int)
X = np.concatenate([X, X])
y = np.concatenate([y,y])

undersample = RandomUnderSampler(sampling_strategy=0.75)

X, y = undersample.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

# oversample class 1 (but not too much)
# difference = sum((y_train==0)*1) - sum((y_train==1)*1)
# difference[0] = np.ceil(difference[0] / 2)
# indices = np.where(y_train==1)[0]
# rand_subsample = np.random.randint(0, len(indices), difference)
# X_train, y_train = np.concatenate((X_train, X_train[indices[rand_subsample]])), np.concatenate((y_train, y_train[indices[rand_subsample]]))

class NumpyArrayDataset(Dataset):

  def __init__(self, X, y):
    self.X = X
    self.y = y
    self.len = len(self.X)

  def __getitem__(self, index):
    return self.X[index], self.y[index]

  def __len__(self):
    return self.len

  def X(self):
    return X

  def y(self):
    return y

epochs = 1000+1
print_epoch = 100
lr = 1e-2
batch_size = 64

classes = np.unique(y).shape[0]


train_data = NumpyArrayDataset(X_train, y_train)
test_data = NumpyArrayDataset(X_test, y_test)


train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# %%

class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(X.shape[1], hidden_channels)
        self.lin2 = Linear(hidden_channels, classes)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

model = MLP(hidden_channels=16)
print(model)

# %% [markdown]
# Our MLP is defined by two linear layers and enhanced by [ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html?highlight=relu#torch.nn.ReLU) non-linearity and [dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html?highlight=dropout#torch.nn.Dropout).
# Here, we first reduce the 1433-dimensional feature vector to a low-dimensional embedding (`hidden_channels=16`), while the second linear layer acts as a classifier that should map each low-dimensional node embedding to one of the 7 classes.

# Let's train our simple MLP by following a similar procedure as described in [the first part of this tutorial](https://colab.research.google.com/drive/1h3-vJGRVloF5zStxL5I0rSy4ZUPNsjy8).
# We again make use of the **cross entropy loss** and **Adam optimizer**.
# This time, we also define a **`test` function** to evaluate how well our final model performs on the test node set (which labels have not been observed during training).

# %%

model = MLP(hidden_channels=16)

criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion
accuracy = accuracy_score
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer

test_result = []
test_probs = []

for epoch in range(epochs):
    
    iteration_loss = 0.0
    iteration_accuracy = 0.0
    
    model.train()
    for i, (X, y) in enumerate(train_loader):
      y_pred = model(X.float())
      loss = criterion(y_pred, y.squeeze())     
      
      iteration_loss = loss
      iteration_accuracy = accuracy(torch.argmax(y_pred, dim=1).detach().cpu().numpy(), y.detach().cpu().numpy().astype(int))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
        

    if(epoch % print_epoch == 0):
        print('Train: epoch: {0} - loss: {1:.5f}; acc: {2:.3f}'.format(epoch, iteration_loss, iteration_accuracy))    

    model.eval()
    for i, (X, y) in enumerate(test_loader):
      y_pred = model(X.float())
      loss = criterion(y_pred, y.squeeze())

      if(epoch == epochs-1):
        test_result.append((y, torch.argmax(y_pred, dim=1)))
        test_probs.append(y_pred)

      iteration_loss = loss
      iteration_accuracy = accuracy(torch.argmax(y_pred, dim=1).detach().cpu().numpy(), y.detach().cpu().numpy().astype(int))

    if(epoch % print_epoch == 0):
        print('Test: epoch: {0} - loss: {1:.5f}; acc: {2:.3f}'.format(epoch, iteration_loss, iteration_accuracy))


# def test():
#       model.eval()
#       out = model(data.x)
#       pred = out.argmax(dim=1)  # Use the class with highest probability.
#       test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
# #       test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
# #       return test_acc
# 
# 
# # %%
# test_acc = test()
# print(f'Test Accuracy: {test_acc:.4f}')

# %%

true, pred = zip(*test_result)
true, pred = torch.cat(true), torch.cat(pred)

cm = confusion_matrix(true, pred, labels=[0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot()

# %%

print('class 0 accuracy: {0:.3f}'.format(cm[0,0]/sum(cm[0])))
print('class 1 accuracy: {0:.3f}'.format(cm[1,1]/sum(cm[1])))

# %%

from sklearn.metrics import auc, average_precision_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

lr_probs = torch.cat(test_probs)

lr_probs = lr_probs.detach().cpu().numpy()

lr_probs = lr_probs[:, 1]

# calculate auc of roc
auc_roc = roc_auc_score(true, lr_probs)

# calculate roc curve
fpr, tpr, _ = roc_curve(true, lr_probs)

# plot ROC curve
plt.figure()
lw = 2
plt.plot(
fpr,
tpr,
color="darkorange",
lw=lw,
label="ROC curve (area = %0.2f)" % auc_roc,
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
