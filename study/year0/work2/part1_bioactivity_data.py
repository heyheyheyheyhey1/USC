import os
import pandas as pd
from chembl_webresource_client.new_client import new_client

DATA_DIR = "data"
BIOACTIVITY_DATA = "bioactivity_data_raw.csv"
PROCESSED_BIOACTIVITY_DATA = "bioactivity_data_preprocessed.csv"

if (not os.path.exists(DATA_DIR)):
    os.mkdir(os.path.join(DATA_DIR))

# 搜索源数据
target = new_client.target
target_query = target.search("coronavirus")
targets = pd.DataFrame.from_dict(target_query)
print(targets)

#选择目标数据
selected_target = targets.target_chembl_id[4]
activity = new_client.activity
res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")

# potency 越低越好
df = pd.DataFrame.from_dict(res)
# 写入
df.to_csv(os.path.join(DATA_DIR, BIOACTIVITY_DATA), index=False)

# 排除standard value 丢失的数据
df2 = df[df.standard_value.notna()]
print(len(df), len(df2))

# 分类成三种化合物 active inactive intermediate 10000,1000作为临界值
for idx, row in df2.iterrows():
    v = float(row.standard_value)
    if v >= 10000:
        df2.loc[idx, "class"] = "inactive"
    elif v < 1000:
        df2.loc[idx, "class"] = "active"
    else:
        df2.loc[idx, "class"] = "intermediate"

cb_columns = ["molecule_chembl_id", "canonical_smiles", "standard_value", "class"]
df3 = df2[cb_columns]
df3.to_csv(os.path.join(DATA_DIR, PROCESSED_BIOACTIVITY_DATA), index=False)
