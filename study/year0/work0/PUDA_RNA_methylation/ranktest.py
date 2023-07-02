import os
import pandas as pd
import torch
from PUDA_RNA_methylation import Discriminator
import numpy as np
import matplotlib.pyplot as plt
from random import sample
from sklearn.metrics import roc_curve, auc

DATA_DIR = os.path.join("data")
MODEL_DIR = os.path.join("model", "PUGAN")
discriminator = Discriminator(in_dim=1517)
discriminator.load_state_dict(torch.load(os.path.join(MODEL_DIR, "discriminator", "discriminator_n_1000.pth")))
discriminator.eval()

selected_data = "selected_dataset.tsv"
dataset_matrix = os.path.join(DATA_DIR, selected_data)
selected_data = pd.read_csv(dataset_matrix, delimiter="\t", index_col=0, low_memory=False)
positive_genes = [line.rstrip('\n') for line in open(os.path.join(DATA_DIR, "test_positive_genes.txt"))]
negative_genes = [line.rstrip('\n') for line in open(os.path.join(DATA_DIR, "test_negative_genes.txt"))]
positive_data = selected_data.loc[positive_genes].values
negative_data = selected_data.loc[negative_genes].values

y0 = np.zeros([len(negative_data), ])
y1 = np.ones([len(positive_data), ])
# 定义y
y = np.concatenate([y0, y1], axis=0)
# 定义x
x = np.concatenate([np.array(negative_data), positive_data])
x = torch.tensor(x, dtype=torch.float32)
# 打乱
# x,y = shuffle(x,y)

pred = discriminator(x)

pred = pred.view(pred.shape[0], ).detach().numpy()

fpr, tpr, thread = roc_curve(y, pred)
roc_auc_score = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic examples')
plt.legend(loc=4)
plt.grid(True)
plt.tight_layout()
plt.savefig('roc.png')
plt.show()

# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(loc=4)
# plt.savefig(os.path.join("visualize_syn_real.png"))
