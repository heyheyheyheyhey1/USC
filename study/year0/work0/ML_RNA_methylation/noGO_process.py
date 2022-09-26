import pandas
import numpy
import os
src_dir = 'RMT_results'
out_dir = 'data'
full_dataset_name = 'selected_dataset.tsv'
out_dataset_name = 'harmonizome_data_combined_noGO.tsv'
harmonizome_data = pandas.read_csv(os.path.join(src_dir,full_dataset_name),delimiter='\t',low_memory=False,index_col=0)
harmonizome_data = harmonizome_data.iloc[:,[('GO' not in x) for x in harmonizome_data.columns]]
harmonizome_data.to_csv(os.path.join(out_dir,out_dataset_name), index=True, sep="\t", header=True)