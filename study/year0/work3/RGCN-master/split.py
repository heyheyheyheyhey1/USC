import os
#
from sklearn.model_selection  import train_test_split
# import csv
import numpy as np
import pandas as pd
data_url = r"C:\Users\17181.LAPTOP-7AFTFG18\Desktop\drug_drug_disease.csv"
data = pd.read_csv(data_url,header=None)
# data_values = data.values
train,test = train_test_split(data,test_size=0.2)
train = train[[0,2,1]]
test = test[[0,2,1]]
train.to_csv(os.path.join("train.txt"),index = False,header = False,sep = '\t')

test.to_csv(os.path.join("test.txt"),index = False,header = False,sep = '\t')

# with open(r"C:\Users\17181.LAPTOP-7AFTFG18\Desktop\drug_drug_disease.csv", 'r', encoding = 'utf-8', newline= '')as csv_file:
#    data = csv.reader(csv_file)
#    header_row = next(data)
#    highs = []
#    for row in data:
#       highs.append(row)
#       # print(highs)
#
# X, Y = train_test_split(highs, test_size=0.2, random_state=42)
# with open('test.txt','w') as f:
#    f.write(str(Y))
# with open('train.txt','w') as f:
#    f.write(str(X))