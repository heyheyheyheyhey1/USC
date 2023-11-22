import os
import random

import pandas as pd
import torch
from WGANGP import Generator
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_validate, LeaveOneOut
from sklearn.metrics import make_scorer, accuracy_score, precision_score
from sklearn.utils import shuffle
from imblearn.over_sampling import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

DATA_DIR = os.path.join("data")
MODEL_DIR = os.path.join("model", "wgangp")
generator = Generator(in_dim=128, out_dim=1517)
generator.load_state_dict(torch.load(os.path.join(MODEL_DIR, "generator", "generator_n_1500_acc_0.466.pth")))
generator.eval()

selected_data = "selected_dataset.tsv"
rnmts = "test_positive_genes.txt"
negtv = "test_negative_genes.txt"
dataset_matrix = os.path.join(DATA_DIR, selected_data)
selected_data = pd.read_csv(dataset_matrix, delimiter="\t", index_col=0, low_memory=False)
all_genes = selected_data.index.unique().to_list()
positive_genes = [line.rstrip('\n') for line in open(os.path.join(DATA_DIR, rnmts))]
negative_genes = [line.rstrip('\n') for line in open(os.path.join(DATA_DIR, negtv))]
num_postv = len(positive_genes)
# 正样本数据
positive_data = selected_data.loc[positive_genes].values
# 负样本数据
negative_genes = random.sample(negative_genes, num_postv * 4)
negative_data = selected_data.loc[negative_genes].values
# GAN生成数据
GAN_data = generator(torch.rand([num_postv*3, 128]))

y0 = np.zeros([num_postv*3, ])
y1 = np.ones([num_postv, ])
y2 = np.full([num_postv * 4, ], 2)

# 定义x
y = np.concatenate([y0, y1, y2], axis=0)
# 定义y
x = np.concatenate([GAN_data.detach(), positive_data, negative_data])



tsne2d = TSNE(n_components=2, learning_rate=200).fit_transform(x)
plt.scatter(tsne2d[y == 0, 0], tsne2d[y == 0, 1], c="blue", label="GAN data")
plt.scatter(tsne2d[y == 1, 0], tsne2d[y == 1, 1], c="red", label="positive data")
plt.scatter(tsne2d[y == 2, 0], tsne2d[y == 2, 1], c="black", label="negative data")
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join("visualize_gan_negtv_postv.png"))
plt.show()
