# 2023-05-30
- https://proceedings.neurips.cc/paper_files/paper/2022/file/65cbe3e21ac62553111d9ecf7d60c18e-Paper-Conference.pdf
- https://link.springer.com/chapter/10.1007/978-3-030-87199-4_44
- https://proceedings.neurips.cc/paper_files/paper/2021/file/2f37d10131f2a483a8dd005b3d14b0d9-Paper.pdf

- 这篇nips，就是我之前说的模块化的深度学习论文，我们很方便在这个方面去修改模型来解决我们的问题
这几篇都是contrastive learning来解决不平衡问题，你可
以看看，然后在我们的数据上用适用的模型

- 还有我之前和你说的transfer learning的问题来解决（不过这比模块化的更难做，我们需要思考把什么信息transfer过来是有效的，可以作为你第二个点），https://link.springer.com/article/10.1007/s10115-015-0870-3
- 取生成正样本按一定规则排序后前几名作为正样本，丢倒GAN里面继续用
# 2023-06-04
- https://arxiv.org/pdf/1805.08318.pdf
奇怪的GAN

# 2023-06-11
- https://arxiv.org/abs/2205.00904
把这篇文章的PU learning loss改为最常规的binary classification PU learnign的loss，再结合GAN构建新的loss，从而构建最终的loss
- 还有原始数据，对于那些本来是关系型数据的，我们不将其转为特征向量，而是直接用关系型数据，然后用GNN相关内容（后面这样子去做）

# 2023-06-27
- Positive-Unlabeled Learning with Adversarial Data Augmentation for Knowledge Graph Completion
- 修改discriminator ，变成几层NN