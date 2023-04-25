import os
import pandas as pd
import torch
from WGAN import Generator
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_validate, LeaveOneOut
from sklearn.metrics import make_scorer, accuracy_score, precision_score
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

DATA_DIR = os.path.join("data")
MODEL_DIR = os.path.join("model")
generator = Generator(in_dim=128, out_dim=1517)
generator.load_state_dict(torch.load(os.path.join(MODEL_DIR, "generator", "generator_n_20000.pth")))
generator.eval()

selected_data = "selected_dataset.tsv"
rnmts = "RNMT.list"
dataset_matrix = os.path.join(DATA_DIR, selected_data)
selected_data = pd.read_csv(dataset_matrix, delimiter="\t", index_col=0, low_memory=False)
positive_genes = [line.rstrip('\n') for line in open(os.path.join(DATA_DIR, rnmts))]
positive_data = selected_data.loc[positive_genes].values
# 生成数据
synthetic_data = generator(torch.rand([92, 128]))
# 保存数据
pd.DataFrame(synthetic_data.detach().numpy()).to_csv(os.path.join(DATA_DIR, "synthetic_data.csv"))
y0 = np.zeros([92, ])
y1 = np.ones([92, ])
#定义x
y = np.concatenate([y0, y1], axis=0)
#定义y
x = np.concatenate([synthetic_data.detach(), positive_data])
#打乱
x,y = shuffle(x,y)
#定义svm
SVM_model = SVC()
loocv = LeaveOneOut()
scorings = {'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score),
            }
# SVM_model.fit(x,y)
cv_result = cross_validate(SVM_model,x,y,scoring=scorings,cv=5)
print(cv_result["test_accuracy"].mean())
print(cv_result["test_precision"].mean())

tsne = TSNE(n_components=2, learning_rate=500).fit_transform(x)
plt.scatter(tsne[y==1,0], tsne[y==1,1],c="red" ,label = "real data")
plt.scatter(tsne[y==0,0], tsne[y==0,1],c="blue" ,label = "GAN synthetic data")
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join("visualize_syn_real.png"))


