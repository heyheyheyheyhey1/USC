import torch
import argparse
from torch import nn
import pickle
import numpy as np
import pandas as pd
from WGAN import WGAN
from WGANGP import WGANGP
import os

DATA_DIR = os.path.join("data")
MODEL_DIR = os.path.join("model")
MODEL_DIR_G = os.path.join(MODEL_DIR, "generator")
MODEL_DIR_D = os.path.join(MODEL_DIR, "discriminator")
if not os.path.exists(MODEL_DIR_G):
    os.makedirs(MODEL_DIR_G)
if not os.path.exists(MODEL_DIR_D):
    os.makedirs(MODEL_DIR_D)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=20000, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
    parser.add_argument('--n_critic', type=int, default=5, help='number of training steps for discriminator per iter')
    parser.add_argument('--clip_value', type=float, default=0.01, help='lower and upper clip value for disc. weights')
    parser.add_argument('--latent_dim', type=int, default=128, help='dimensionality of the latent space')
    parser.add_argument('--beta1', type=int, default=0, help='wgangp optimizer parameter')
    parser.add_argument('--beta2', type=int, default=0.9, help='wgangp optimizer parameter')
    parser.add_argument('--r', type=int, default=10, help='define gradient penalty factor')
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
    args["g_sav_dir"] = MODEL_DIR_G
    args["d_sav_dir"] = MODEL_DIR_D
    wgan = WGANGP(args)
    wgan.train()


if __name__ == "__main__":
    main()
