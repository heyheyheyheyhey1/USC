import random
import torch
from torch import nn
from datetime import datetime
from WGANGP import Generator
import pandas as pd
import pickle
import os
from sklearn.utils import shuffle
import tqdm
import numpy as np
from sklearn.model_selection import *
from sklearn.svm import SVC
from sklearn.metrics import *

CV_DIR = os.path.join("CV")
MODEL_DIR = os.path.join("model")
SAVE_DIR = os.path.join(MODEL_DIR,"DNN")
DATA_DIR = os.path.join("data")
LATENT_DIM = 128
generator = Generator(in_dim=LATENT_DIM, out_dim=1517)
generator.load_state_dict(torch.load(os.path.join(MODEL_DIR, "wgangp", "generator", "generator_n_283_acc_0.507.pth")))
generator.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.exists(CV_DIR):
    os.makedirs(CV_DIR)

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

selected_data = "selected_dataset.tsv"
rnmts = "RNMT.list"
dataset_matrix = os.path.join("data", selected_data)
selected_data = pd.read_csv(dataset_matrix, delimiter="\t", index_col=0, low_memory=False)
all_genes = selected_data.index.unique().to_list()
positive_genes = [line.rstrip('\n') for line in open(os.path.join("data", rnmts))]
negative_genes = set(all_genes).difference(positive_genes)  # 负样本

selected_data["Y"] = [1 if idx in positive_genes else 0 for idx in selected_data.index.to_list()]
test_positive_genes = [line.rstrip('\n') for line in open(os.path.join("data", "test_positive_genes.txt"))]
test_negative_genes = [line.rstrip('\n') for line in open(os.path.join("data", "test_negative_genes.txt"))]
train_positive_genes = set(positive_genes).difference(test_positive_genes)
train_negative_genes = set(negative_genes).difference(test_negative_genes)
train_positive_frame = selected_data.loc[list(train_positive_genes)]
train_negative_frame = selected_data.loc[list(train_negative_genes)]

print("all genes num: %d\n" % (len(all_genes)))
print("test genes num: %d\n" % (len(test_negative_genes) + len(test_positive_genes)))

os_rate = 0.5
ir = float(len(train_negative_genes) / len(train_positive_genes))
epoch_num = 20000

train_X = np.concatenate([train_positive_frame.values, train_negative_frame.values])
train_Y = train_X[:, -1]
train_X = train_X[:, 0:-1]


class RNMT_NET(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        def block(in_feat, out_feat, drop_out=0.5):
            layers = [nn.Linear(in_feat, out_feat)]
            if not drop_out == -1:
                layers.append(nn.Dropout(drop_out))
            return layers

        self.model = nn.Sequential(
            *block(in_dim, in_dim // 2),
            *block(in_dim // 2, in_dim // 2 // 2),
            *block(in_dim // 2 // 2, in_dim // 2 // 2 // 2),
            nn.Linear(in_dim // 2 // 2 // 2, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.model(x)
        return out


def dataset():
    postv_num = len(train_positive_genes)
    negtv_num = len(train_negative_genes)
    for i in range(int(negtv_num / postv_num)):
        neg_batch = train_negative_frame.values[i * postv_num:(i + 1) * postv_num,0:-1]
        x = np.concatenate([np.array(neg_batch), train_positive_frame.values[:,0:-1]])
        y0 = np.zeros([len(neg_batch), ])
        y1 = np.ones([postv_num, ])
        y = np.concatenate([y0, y1], axis=0)
        yield torch.Tensor(x).to(device), torch.Tensor(np.eye(2)[y.astype(int)]).to(device), i

def val_set():
    postv_num = len(train_positive_genes)
    negtv_num = len(train_negative_genes)
    for i in range(int(negtv_num / postv_num)):
        neg_batch = train_negative_frame.values[i * postv_num:(i + 1) * postv_num,0:-1]
        x = np.concatenate([np.array(neg_batch), train_positive_frame.values[:,0:-1]])
        y0 = np.zeros([len(neg_batch), ])
        y1 = np.ones([postv_num, ])
        y = np.concatenate([y0, y1], axis=0)
        yield torch.Tensor(x).to(device), torch.Tensor(np.eye(2)[y.astype(int)]).to(device), i

model = RNMT_NET(in_dim=1517)
model.to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
random.seed(42)
cv_scorings = pd.DataFrame(columns=['Classifier', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])


def train_one_epoch(n_epoch):
    model.train()
    for X, Y , _ in dataset():
        X,Y = shuffle(X,Y)
        y_hat = model(X)
        loss = criterion(y_hat, Y)
        loss.backward()
        optimizer.step()
    model.eval()
    cv_scorings = pd.DataFrame(columns=[ 'Accuracy', 'Precision', 'Recall', 'F1'])
    with torch.no_grad():
        for X, Y, _ in dataset():
            X, Y = shuffle(X, Y)
            y_hat = model(X)
            y_hat = torch.argmax(y_hat,dim=-1).cpu()
            y_true = torch.argmax(Y,dim=-1).cpu()
            scorings = [accuracy_score(y_true, y_hat), precision_score(y_true, y_hat),
                        recall_score(y_true, y_hat),
                        f1_score(y_true, y_hat),
                        ]
            cv_scorings.loc[len(cv_scorings)] = scorings
        mean = cv_scorings.mean()
        print(f"\nepoch {n_epoch}: [acc:{mean.loc['Accuracy']:.3f}] [prec:{mean.loc['Precision']:.3f}] [recall:{mean.loc['Recall']:.3f}] [F1:{mean.loc['F1']:.3f}]"  )

def main():
    os_rate = 0.6
    ir = float(len(train_negative_genes) / len(train_positive_genes))
    epoch_num = 2000
    batch_size = int(len(train_negative_genes) / epoch_num)

    for i in tqdm.tqdm(range(epoch_num)):
        train_one_epoch(i)
    torch.save(model.state_dict(),
               os.path.join(SAVE_DIR, f'rnmt_net_{datetime.now().strftime("%Y-%m-%d %Hh%Mm")}.pth'))


if __name__ == "__main__":
    main()
