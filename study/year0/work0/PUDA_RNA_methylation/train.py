import argparse
import pandas as pd
from PUDA_RNA_methylation import PUGAN
import os
from random import sample

DATA_DIR = os.path.join("data")
MODEL_DIR = os.path.join("model", "PUGAN")
MODEL_DIR_G = os.path.join(MODEL_DIR, "generator")
MODEL_DIR_D = os.path.join(MODEL_DIR, "discriminator")
if not os.path.exists(MODEL_DIR_G):
    os.makedirs(MODEL_DIR_G)
if not os.path.exists(MODEL_DIR_D):
    os.makedirs(MODEL_DIR_D)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=23, help='size of the batches')
    parser.add_argument('--neg_num', type=int, default=6, help='num of negative generated')
    parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
    parser.add_argument('--lr_g', type=float, default=0.00001, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=0.00001, help='learning rate d')
    parser.add_argument('--n_critic', type=int, default=7, help='number of training steps for discriminator per iter')
    parser.add_argument('--latent_dim', type=int, default=256, help='dimensionality of the latent space')
    parser.add_argument('--beta1', type=int, default=0.9, help=' optimizer parameter')
    parser.add_argument('--beta2', type=int, default=0.95, help=' optimizer parameter')
    parser.add_argument('--prior', type=str, default=0.0001, help='prior')
    opt = parser.parse_args()
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

    args = {}
    args["positive_data"] = positive_data
    args["unlabel_data"] = unlabel_data
    args["train_opt"] = opt
    args["g_sav_dir"] = MODEL_DIR_G
    args["d_sav_dir"] = MODEL_DIR_D
    gan = PUGAN(args)
    gan.train()


if __name__ == "__main__":
    main()
