import gzip
import numpy
import pandas
import os
src_dir = 'origin_data'
compressed_filename = 'gene_attribute_matrix.txt.gz'
target_filename = 'gene_attribute_matrix.txt'
full_dataset_name = 'harmonizome_data_combined.tsv'
full_feature_names = pandas.read_csv(os.path.join("data",full_dataset_name),delimiter='\t',low_memory=False,nrows=1)
full_feature_names = ([x.strip() for x in full_feature_names])[1:-1]
datasets = []
dataset_dirs = os.listdir(src_dir)

def get_dataframe(dataset_name):
    fPath = os.path.join(src_dir,dataset_name,compressed_filename)
    dataset = pandas.read_csv(fPath,delimiter="\t",index_col=0,low_memory=False)
    dataset = dataset.fillna(0)
    if dataset_name in ['GO_BP','GO_MF','Interpro_predDomains','TISSUES_curatProtein','KEGG_Pathway'] :
        dataset = dataset.rename(columns=dataset.iloc[0])
    dataset = dataset.iloc[2:-1,2:-1]
    dataset = dataset.loc[:,(f'{dataset_name}_'+dataset.columns).isin(full_feature_names)]
    if len(dataset.columns) == 0:
        print(f'find empty dataset: {dataset_name}\n')
    return dataset


for dataset_name in dataset_dirs:
    dataset = get_dataframe(dataset_name)
    datasets.append(dataset)

dataset_combined = pandas.concat(datasets,axis=1)

dataset_combined.to_csv(os.path.join('data', 'my_combined_data.tsv'), index=True, sep="\t", header=True)