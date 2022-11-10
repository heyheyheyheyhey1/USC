import os
import sys

import pandas as pd

df3 = pd.read_csv("data/bioactivity_data.csv")
pIC50 = df3["pIC50"]
df3 = df3[['canonical_smiles', 'molecule_chembl_id']]
df3.to_csv('data/molecule.smi', sep='\t', index=False, header=False)

dataset = pd.read_csv('data/descriptors_output.csv')
dataset.drop(columns=['Name'], inplace=True)
dataset = pd.concat([dataset,pd.DataFrame(pIC50)],axis=1)
dataset.to_csv('data/sars-3c-like-dataset.csv', index=False)
