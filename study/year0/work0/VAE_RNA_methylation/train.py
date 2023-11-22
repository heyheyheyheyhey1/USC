import torch
from sklearn.metrics import accuracy_score, make_scorer, precision_score
from sklearn.model_selection import LeaveOneOut, cross_validate
from sklearn.svm import SVC
from sklearn.utils import shuffle
from torch import nn, optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# from VAE import VAE
from MMD_VAE import MMDVAE as VAE
from random import sample
import os

DATA_DIR = os.path.join("data")
MODEL_DIR = os.path.join("model", "VAE")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


def C2ST(model, data):
    model.eval()
    len = data.shape[0]
    fake_data = model.generate_data(len)
    c2st_x = np.concatenate([fake_data, data])
    c2st_y = np.concatenate([np.zeros([len, ]), np.ones([len, ])])
    c2st_x, c2st_y = shuffle(c2st_x, c2st_y)
    SVM_model = SVC()
    loocv = LeaveOneOut()
    scorings = {'accuracy': make_scorer(accuracy_score),
                'precision': make_scorer(precision_score),
                }
    cv_result = cross_validate(SVM_model, c2st_x, c2st_y, scoring=scorings, cv=5)
    return cv_result["test_accuracy"].mean(), cv_result["test_precision"].mean()


# 定义训练函数
def train_vae(model, data_loader, num_epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()  # 使用二进制交叉熵作为损失函数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, data in enumerate(data_loader):
            x = data.to(device)
            loss = VAE.train_one_epoch(model, x, optimizer, criterion)
            total_loss += loss.item()
        acc, prec = C2ST(model, data_loader.dataset)
        print('Epoch [{}/{}], Loss: {:.4f} ,C2ST: {:.4f}'.format(epoch + 1, num_epochs, total_loss / len(data_loader),
                                                                 acc))


def main():
    # 设置参数
    input_dim = 1517
    latent_dim = 100
    num_epochs = 10000
    batch_size = 73
    lr = 0.0001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    selected_data = "selected_dataset.tsv"
    rnmts = "RNMT.list"

    dataset_matrix = os.path.join(DATA_DIR, selected_data)
    selected_data = pd.read_csv(dataset_matrix, delimiter="\t", index_col=0, low_memory=False)
    positive_genes = [line.rstrip('\n') for line in open(os.path.join(DATA_DIR, rnmts))]
    unlabel_genes = list(selected_data.index.difference(positive_genes))

    train_positive_genes = sample(positive_genes, int(len(positive_genes) * 0.8))
    train_unlable_genes = sample(unlabel_genes, int(len(unlabel_genes) * 0.8))

    positive_data = selected_data.loc[train_positive_genes].values
    unlabel_data = selected_data.loc[train_unlable_genes].values

    ft = open(os.path.join(DATA_DIR, "test_positive_genes.txt"), "w")
    fu = open(os.path.join(DATA_DIR, "test_negative_genes.txt"), "w")

    for gene in set(positive_genes).difference(train_positive_genes):
        ft.write(gene)
        ft.write("\n")
    ft.close()
    for gene in set(unlabel_genes).difference(train_unlable_genes):
        fu.write(gene)
        fu.write("\n")
    fu.close()

    # 创建数据集并加载到数据加载器
    # 假设你已经有了名为X的数据，需要将其转换成torch.Tensor的形式
    data_loader = DataLoader(torch.Tensor(positive_data), batch_size=batch_size, shuffle=True)

    # 创建VAE模型并将其移动到GPU（如果可用）
    model = VAE(input_dim, latent_dim).to(device)

    # 训练VAE模型
    train_vae(model, data_loader, num_epochs, lr)


if __name__ == "__main__":
    main()
