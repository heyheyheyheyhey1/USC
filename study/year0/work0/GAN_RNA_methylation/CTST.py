import os
import pandas as pd
import torch
from WGANGP import Generator
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_validate, LeaveOneOut
from sklearn.metrics import make_scorer, accuracy_score, precision_score
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

DATA_DIR = os.path.join("data")
MODEL_DIR = os.path.join("model", "wgangp")
LATENT_DIM = 128
generator = Generator(in_dim=LATENT_DIM, out_dim=1517)

selected_data = "selected_dataset.tsv"
rnmts = "RNMT.list"
dataset_matrix = os.path.join(DATA_DIR, selected_data)
selected_data = pd.read_csv(dataset_matrix, delimiter="\t", index_col=0, low_memory=False)
positive_genes = [line.rstrip('\n') for line in open(os.path.join(DATA_DIR, rnmts))]
test_positive_genes = [line.rstrip('\n') for line in open(os.path.join(DATA_DIR, "test_positive_genes.txt"))]
train_positive_genes = set(positive_genes).difference(test_positive_genes)
positive_data = selected_data.loc[train_positive_genes].values
mdls = os.listdir(os.path.join(MODEL_DIR, "generator"))
acc_average = pd.DataFrame(index=mdls)
loocv = LeaveOneOut()


def weight_reset(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


# c2st 测试
for mdl in tqdm(mdls):
    generator.apply(weight_reset)
    generator.load_state_dict(torch.load(os.path.join(MODEL_DIR, "generator", mdl)))
    generator.eval()

    SVM_model = SVC()
    scorings = {'accuracy': make_scorer(accuracy_score)}
    for i in range(200):
        # 生成数据
        synthetic_data = generator(torch.rand([73, LATENT_DIM]))
        x = np.concatenate([synthetic_data.detach(), positive_data])

        # 定义y
        y0 = np.zeros([73, ])
        y1 = np.ones([73, ])
        y = np.concatenate([y0, y1], axis=0)
        x, y = shuffle(x, y)

        cv_result = cross_validate(SVM_model, x, y, scoring=scorings, cv=5)
        acc_average.loc[mdl, f'round_{i}'] = cv_result["test_accuracy"].mean()

    acc_average.loc[mdl, "mean"] = acc_average.loc[mdl].mean().round(5)
    acc_average.loc[mdl, "var"] = np.float64(acc_average.loc[mdl].var()).round(5)
    acc_average.loc[mdl, "threshold"] = abs(acc_average.loc[mdl].mean() - 0.5).round(5)

# 导出
acc_average.sort_values("threshold", ascending=True ,inplace=True)
acc_average.to_csv(os.path.join(MODEL_DIR, "CTST_result.csv"), index=True)
plt.figure()
plt.axis(ymax = 1.05,xmax = 62,xmin = -1,ymin=-0.02)

greeny = [value for index,value in enumerate(mean) if abs(value-0.5)<=0.02]
greenx = [index for index,value in enumerate(mean) if abs(value-0.5)<=0.02]
plt.scatter(greenx,greeny, c="green" )

orangey = [value for index,value in enumerate(mean) if 0.02< abs(value-0.5)<=0.1]
orangex = [index for index,value in enumerate(mean) if 0.02< abs(value-0.5)<=0.1]
plt.scatter(orangex,orangey, c="orange" )

redy = [value for index,value in enumerate(mean) if 0.1< abs(value-0.5)]
redx = [index for index,value in enumerate(mean) if 0.1< abs(value-0.5)]
plt.scatter(redx,redy, c="red" )

plt.xlabel('X: model index')
plt.ylabel('Y: CTST accuracy')
plt.grid(True,linestyle = '--')
plt.yticks([0,0.2,0.4,0.5,0.6,0.8,1.0])
plt.grid(axis='x')
# plt.tight_layout()
# plt.subplots_adjust(top=1)
plt.gca().set_aspect(30)
plt.savefig("visualize_CTST.png", bbox_inches='tight')
plt.show()

# # 载入最佳模型
# generator.apply(weight_reset)
# selected_mdl = acc_average.index[0]
# generator.load_state_dict(torch.load(os.path.join(MODEL_DIR, "generator", selected_mdl)))
# generator.eval()
#
# # 生成数据
# synthetic_data = generator(torch.rand([73, LATENT_DIM]))
# # 保存数据
# # pd.DataFrame(synthetic_data.detach().numpy()).to_csv(os.path.join(DATA_DIR, "synthetic_data.csv"))
# y0 = np.zeros([73, ])
# y1 = np.ones([73, ])
# # 定义x
# y = np.concatenate([y0, y1], axis=0)
# # 定义y
# x = np.concatenate([synthetic_data.detach(), positive_data])
# # 打乱
# x, y = shuffle(x, y)
# # 定义svm
# SVM_model = SVC()
# scorings = {'accuracy': make_scorer(accuracy_score),
#             'precision': make_scorer(precision_score),
#             }
# # SVM_model.fit(x,y)
# cv_result = cross_validate(SVM_model, x, y, scoring=scorings, cv=3)
# print(cv_result["test_accuracy"].mean())
# print(cv_result["test_precision"].mean())
#
# tsne = TSNE(n_components=2, random_state=42, learning_rate=200, init="pca").fit_transform(x)
# plt.scatter(tsne[y == 1, 0], tsne[y == 1, 1], c="red", label="real data")
# plt.scatter(tsne[y == 0, 0], tsne[y == 0, 1], c="blue", label="GAN synthetic data")
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(loc=4)
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join("visualize_syn_real.png"))
# plt.show()
#
