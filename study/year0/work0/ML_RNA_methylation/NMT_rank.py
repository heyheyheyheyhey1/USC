import pandas
import os
src_dir = 'data'
src_name = 'string_interactions_short.tsv'
interactions_frame = pandas.read_csv(os.path.join(src_dir,src_name),delimiter='\t',low_memory=False)
interactions_frame = interactions_frame.iloc[:,[0,1,-1]]