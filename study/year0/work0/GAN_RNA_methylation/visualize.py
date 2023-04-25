import os
import random

import pandas as pd
import torch
from WGAN import Generator
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_validate, LeaveOneOut
from sklearn.metrics import make_scorer, accuracy_score, precision_score
from sklearn.utils import shuffle
from imblearn.over_sampling import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

DATA_DIR = os.path.join("data")
MODEL_DIR = os.path.join("model")
SAMPLE_NUMS = 92
generator = Generator(in_dim=128, out_dim=1517)
generator.load_state_dict(torch.load(os.path.join(MODEL_DIR, "generator", "generator_n_20000.pth")))
generator.eval()
smoter = SMOTE(random_state=42)

selected_data = "selected_dataset.tsv"
rnmts = "RNMT.list"
dataset_matrix = os.path.join(DATA_DIR, selected_data)
selected_data = pd.read_csv(dataset_matrix, delimiter="\t", index_col=0, low_memory=False)
all_genes = selected_data.index.unique().to_list()
positive_genes = [line.rstrip('\n') for line in open(os.path.join(DATA_DIR, rnmts))]
positive_genes = random.sample(positive_genes, SAMPLE_NUMS)
negative_genes = set(all_genes).difference(positive_genes)
# 真实正样本数据
positive_data = selected_data.loc[positive_genes].values
# 真实负样本数据
negative_genes = random.sample(negative_genes, SAMPLE_NUMS)
negative_data = selected_data.loc[negative_genes].values
# GAN生成数据
GAN_data = generator(torch.rand([SAMPLE_NUMS*2, 128]))

y0 = np.zeros([SAMPLE_NUMS*2, ])
y1 = np.ones([SAMPLE_NUMS, ])


# 定义x
y = np.concatenate([y0, y1], axis=0)
# 定义y
x = np.concatenate([GAN_data.detach(), positive_data])
# SMOTE生成数据
x,y = smoter.fit_resample(x,y)
#去掉gan数据
x = x[SAMPLE_NUMS-1:-1]
y = y[SAMPLE_NUMS-1:-1]
y[2*SAMPLE_NUMS-1:-1] = 2.



tsne2d = TSNE(n_components=2, learning_rate=100).fit_transform(x)
plt.scatter(tsne2d[y == 0, 0], tsne2d[y == 0, 1], c="blue", label="GAN data")
plt.scatter(tsne2d[y == 1, 0], tsne2d[y == 1, 1], c="red", label="positive data")
plt.scatter(tsne2d[y == 2, 0], tsne2d[y == 2, 1], c="black", label="SMOTE data")
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join("visualize_gan_smote_postv_negtv.png"))
