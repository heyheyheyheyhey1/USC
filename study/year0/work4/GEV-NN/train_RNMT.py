import random
import pandas as pd
import os
from random import sample
from sklearn.utils import shuffle
from GEV_NN_torch import GEV
import torch
from torch import  nn
import tqdm
import numpy as np

harmonizome_data = "harmonizome_data_combined.tsv"
selected_data = "selected_dataset.tsv"
rnmts = "RNMT.list"
dataset_matrix = os.path.join("data", selected_data)
selected_data = pd.read_csv(dataset_matrix, delimiter="\t", index_col=0, low_memory=False)

all_genes = selected_data.index.unique().to_list()
positive_genes = [line.rstrip('\n') for line in open(os.path.join("data", rnmts))]
negative_genes = set(all_genes).difference(positive_genes)  # 负样本

selected_data["Y"] = [1 if idx in positive_genes else 0 for idx in selected_data.index.to_list()]
test_positive_genes = sample(positive_genes, int(0.2 * len(positive_genes)))  # 取正样本20%作为测试
test_negative_genes = sample(negative_genes, int(0.2 * len(negative_genes)))  # 测试负样本
train_positive_genes = set(positive_genes).difference(test_positive_genes)
train_negative_genes = set(negative_genes).difference(test_negative_genes)
train_positive_frame = selected_data.loc[list(train_positive_genes)]
train_negative_frame = selected_data.loc[list(train_negative_genes)]

print("all genes num: %d\n" % (len(all_genes)))
print("test genes num: %d\n" % (len(test_negative_genes) + len(test_positive_genes)))

random.seed(42)


def data_block_n(i, batch_size):
    n_train = train_negative_frame.iloc[i * batch_size:i * batch_size + batch_size]
    block = pd.concat([n_train, train_positive_frame], axis=0)
    block_y = block.iloc[:, -1].values
    block_x = block.iloc[:, 0:-1].values
    # os_x, os_y = SMOTEN(random_state=42).fit_resample(block_x, block_y)
    # out_x, out_y = shuffle(os_x, os_y)
    return block_x, block_y


def get_data_n(batch_size):
    for i in range(int(len(train_negative_genes) / batch_size)):
        x, y = data_block_n(i, batch_size)
        y = np.eye(2)[y]
        yield torch.tensor(x,dtype=torch.float32), torch.tensor(y,dtype=torch.float32)

def loss():
    return nn.CrossEntropyLoss()

def optimizer(model):
    return torch.optim.SGD(model.parameters(),1e-3)

def train_one_epoch( n_size,model,loss,optimizer):
    for x, y in get_data_n(n_size):
        optimizer.zero_grad()
        y_hat = model(x)
        l = loss(y,y_hat)
        l.backward()
        optimizer.step()

    return


def main():
    os_rate = 0.6
    epoch_num = 300
    ir = float(len(train_negative_genes) / len(train_positive_genes))
    n_size = int(len(train_negative_genes) / int(ir * os_rate))
    feature_size = train_negative_frame.values.shape[1]-1
    model = GEV(dimIn=feature_size)
    for i in tqdm.tqdm(range(epoch_num)):
        train_one_epoch(n_size,model,loss(),optimizer(model))


if __name__ == "__main__":
    main()
