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
generator = Generator(in_dim=128, out_dim=1517)

selected_data = "selected_dataset.tsv"
rnmts = "test_positive_genes.txt"
dataset_matrix = os.path.join(DATA_DIR, selected_data)
selected_data = pd.read_csv(dataset_matrix, delimiter="\t", index_col=0, low_memory=False)
positive_genes = [line.rstrip('\n') for line in open(os.path.join(DATA_DIR, rnmts))]
positive_data = selected_data.loc[positive_genes].values
mdls = os.listdir(os.path.join(MODEL_DIR, "generator"))
acc_average = pd.DataFrame(index=mdls)


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
    for i in range(20):
        # 生成数据
        synthetic_data = generator(torch.rand([19, 128]))
        x = np.concatenate([synthetic_data.detach(), positive_data])

        # 定义y
        y0 = np.zeros([19, ])
        y1 = np.ones([19, ])
        y = np.concatenate([y0, y1], axis=0)
        x, y = shuffle(x, y)

        cv_result = cross_validate(SVM_model, x, y, scoring=scorings, cv=3)
        acc_average.loc[mdl, f'round_{i}'] = cv_result["test_accuracy"].mean()

    acc_average.loc[mdl, "mean"] = acc_average.loc[mdl].mean().round(5)
    acc_average.loc[mdl, "var"] = acc_average.loc[mdl].var().round(5)
    acc_average.loc[mdl, "threshold"] = abs(acc_average.loc[mdl].mean() - 0.5).round(5)

# 导出
acc_average.sort_values("threshold", ascending=True ,inplace=True)
acc_average.to_csv(os.path.join(MODEL_DIR, "CTST_result.csv"), index=True)

# 载入最佳模型
generator.apply(weight_reset)
selected_mdl = acc_average.index[0]
generator.load_state_dict(torch.load(os.path.join(MODEL_DIR, "generator", selected_mdl)))
generator.eval()

# 生成数据
synthetic_data = generator(torch.rand([19, 128]))
# 保存数据
# pd.DataFrame(synthetic_data.detach().numpy()).to_csv(os.path.join(DATA_DIR, "synthetic_data.csv"))
y0 = np.zeros([19, ])
y1 = np.ones([19, ])
# 定义x
y = np.concatenate([y0, y1], axis=0)
# 定义y
x = np.concatenate([synthetic_data.detach(), positive_data])
# 打乱
x, y = shuffle(x, y)
# 定义svm
SVM_model = SVC()
loocv = LeaveOneOut()
scorings = {'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score),
            }
# SVM_model.fit(x,y)
cv_result = cross_validate(SVM_model, x, y, scoring=scorings, cv=3)
print(cv_result["test_accuracy"].mean())
print(cv_result["test_precision"].mean())

tsne = TSNE(n_components=2, random_state=42, learning_rate=200, init="pca").fit_transform(x)
plt.scatter(tsne[y == 1, 0], tsne[y == 1, 1], c="red", label="real data")
plt.scatter(tsne[y == 0, 0], tsne[y == 0, 1], c="blue", label="GAN synthetic data")
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join("visualize_syn_real.png"))
plt.show()
