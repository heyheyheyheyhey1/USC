import random
import torch
from torch import nn
from datetime import datetime
from WGANGP import Generator
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
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
SAVE_DIR = os.path.join(MODEL_DIR, "DNN")
DATA_DIR = os.path.join("data")
LATENT_DIM = 128
generator = Generator(in_dim=LATENT_DIM, out_dim=1517)
generator.load_state_dict(torch.load(os.path.join(MODEL_DIR, "wgangp", "generator", "generator_n_574_acc_0.494.pth")))
generator.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'using device {device}\n')
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

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.zero_()


class RNMT_NET(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        def block(in_feat, out_feat, drop_out=0.5, norm=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if norm:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.Sigmoid())
            if not drop_out == -1:
                layers.append(nn.Dropout(drop_out))

            return layers

        self.model = nn.Sequential(
            *block(in_dim, in_dim // 2 // 2),
            *block(in_dim // 2 // 2, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.model(x)
        return out


def oversample(x, y):
    synthetic_num = abs(np.sum(y == 0) - np.sum(y == 1))
    with torch.no_grad():
        synthetic_data = generator(torch.rand(synthetic_num, LATENT_DIM))
    x_os = np.concatenate([synthetic_data.detach(), x]).copy()
    y_os = np.concatenate([np.ones([synthetic_num, ]), y]).copy()
    return x_os, y_os

class RMNT_DATA(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 获取索引为index的样本和标签
        sample = self.data[index]
        label = self.labels[index]

        # 将样本和标签转换为Tensor格式
        sample = torch.Tensor(sample).to(device)
        label = torch.Tensor(np.eye(2)[label.astype(int)]).to(device)

        return sample, label


def dataset():
    rnmt_data = pd.concat([train_negative_frame, train_positive_frame], axis=0)
    rnmt_data_y = rnmt_data.iloc[:, -1].values
    rnmt_data_x = rnmt_data.iloc[:, 0:-1].values
    os_x, os_y = oversample(rnmt_data_x, rnmt_data_y)
    return RMNT_DATA(os_x,os_y)


# def val_set():
#     postv_num = len(test_positive_genes)
#     negtv_num = len(test_negative_genes)
#     val_neg_data = selected_data.loc[test_negative_genes]
#     val_pos_data = selected_data.loc[test_positive_genes]
#     for i in range(int(negtv_num / postv_num)):
#         neg_batch = val_neg_data.values[i * postv_num:(i + 1) * postv_num, 0:-1]
#         x = np.concatenate([np.array(neg_batch), val_pos_data.values[:, 0:-1]])
#         y0 = np.zeros([len(neg_batch), ])
#         y1 = np.ones([postv_num, ])
#         y = np.concatenate([y0, y1], axis=0)
#         yield torch.Tensor(x).to(device), torch.Tensor(np.eye(2)[y.astype(int)]).to(device), i

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

model = RNMT_NET(in_dim=1517).apply(initialize_weights)
model.to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
random.seed(42)
batch_size = 500
dataloader = DataLoader(dataset(),batch_size=batch_size,shuffle=True)

def train_one_epoch(n_epoch):
    model.train()
    losses = []
    for X, Y in dataloader:
        optimizer.zero_grad()
        X, Y = shuffle(X, Y)
        y_hat = model(X)
        loss = criterion(y_hat, Y)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu())
    model.eval()
    cv_scorings = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1','MCC'])
    with torch.no_grad():
        for X, Y, _ in val_set():
            X, Y = shuffle(X, Y)
            y_hat = model(X)
            y_hat = torch.argmax(y_hat, dim=-1).cpu()
            y_true = torch.argmax(Y, dim=-1).cpu()
            scorings = [accuracy_score(y_true, y_hat), precision_score(y_true, y_hat),
                        recall_score(y_true, y_hat),
                        f1_score(y_true, y_hat),
                        matthews_corrcoef(y_true,y_hat)]
            cv_scorings.loc[len(cv_scorings)] = scorings
        mean = cv_scorings.mean()
        print(
            f"\nepoch {n_epoch}: [acc:{mean.loc['Accuracy']:.3f}] [prec:{mean.loc['Precision']:.3f}] [recall:{mean.loc['Recall']:.3f}] [F1:{mean.loc['F1']:.3f}] [MCC:{mean.loc['MCC']:.3f}]  [loss: {np.mean(losses):.5f}]")


def main():
    epoch_num = 800
    for i in tqdm.tqdm(range(epoch_num)):
        train_one_epoch(i)
    torch.save(model.state_dict(),
               os.path.join(SAVE_DIR, f'rnmt_net_{datetime.now().strftime("%Y-%m-%d %Hh%Mm")}.pth'))


if __name__ == "__main__":
    main()
