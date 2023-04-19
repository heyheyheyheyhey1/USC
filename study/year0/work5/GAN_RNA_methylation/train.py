import torch
import argparse
from torch import nn
import pickle
import numpy as np
import pandas as pd
from WGAN import WGAN
import os

DATA_DIR = os.path.join("data")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=4000, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=12, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--feature_size', type=int, default=934, help='size of each image dimension')
    parser.add_argument('--n_critic', type=int, default=5, help='number of training steps for discriminator per iter')
    parser.add_argument('--clip_value', type=float, default=0.003, help='lower and upper clip value for disc. weights')
    parser.add_argument('--sample_interval', type=int, default=400, help='interval betwen image samples')
    opt = parser.parse_args()
    selected_data = "selected_dataset.tsv"
    rnmts = "RNMT.list"
    dataset_matrix = os.path.join(DATA_DIR, selected_data)
    selected_data = pd.read_csv(dataset_matrix, delimiter="\t", index_col=0, low_memory=False)
    positive_genes = [line.rstrip('\n') for line in open(os.path.join(DATA_DIR, rnmts))]
    positive_data = selected_data.loc[positive_genes].values
    args = {}
    args["train_data"] = positive_data
    args["train_opt"] = opt
    wgan = WGAN(args)
    wgan.train()


if __name__ == "__main__":
    main()
