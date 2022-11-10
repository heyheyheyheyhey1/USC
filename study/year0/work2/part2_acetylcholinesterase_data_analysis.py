import os.path
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import pandas as pd
import numpy as np

DATA_DIR = "data"
PROCESSED_BIOACTIVITY_DATA = "acetylcholinesterase_03_bioactivity_data_curated.csv"
BIOACTIVITY_DATA_OUT = "acetylcholinesterase_bioactivity_data.csv"
df = pd.read_csv(os.path.join(DATA_DIR, PROCESSED_BIOACTIVITY_DATA))


# lipinski分数
def lipinski(smiles, verbose=False):
    moldata = []
    for elem in smiles:
        mol = Chem.MolFromSmiles(elem)
        moldata.append(mol)

    baseData = np.arange(1, 1)
    i = 0
    for mol in moldata:

        desc_MolWt = Descriptors.MolWt(mol)
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_NumHDonors = Lipinski.NumHDonors(mol)
        desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)

        row = np.array([desc_MolWt,
                        desc_MolLogP,
                        desc_NumHDonors,
                        desc_NumHAcceptors])

        if (i == 0):
            baseData = row
        else:
            baseData = np.vstack([baseData, row])
        i = i + 1

    columnNames = ["MW", "LogP", "NumHDonors", "NumHAcceptors"]
    descriptors = pd.DataFrame(data=baseData, columns=columnNames)

    return descriptors


df_lipinski = lipinski(df.canonical_smiles)

df_combined = pd.concat([df, df_lipinski], axis=1)

# IC50 PIC50单位转换

df_combined["standard_value_norm"] = [100000000 if row.standard_value > 100000000 else row.standard_value for (idx, row)
                                      in df_combined.iterrows()]  # 缩小最大标准值


def pIC50(input):
    pIC50 = []
    for i in input['standard_value_norm']:
        molar = i * (10 ** -9)  # Converts nM to M
        pIC50.append(-np.log10(molar))

    input['pIC50'] = pIC50
    x = input.drop('standard_value_norm', 1)

    return x


df_out = pIC50(df_combined)
df_out = df_out[df_out["class"] != "intermediate"]
df_out.to_csv(os.path.join(DATA_DIR, BIOACTIVITY_DATA_OUT), index=False)
pass
