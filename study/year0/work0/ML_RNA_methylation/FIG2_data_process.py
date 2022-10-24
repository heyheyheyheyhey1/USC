import os
import pandas
source_dir = 'RMT_results'
classifiers = ["GB", "GNB", "LR", "RF", "SVM"]
data_out_dir = 'data'
data_out_name = 'FIG2_data.txt'
draw_dirs = os.listdir(source_dir)
draw_dirs = [d for d in draw_dirs if d.__contains__("draw_")]
rank_nums = len(draw_dirs)
known_RNMTS = [line.rstrip('\n') for line in open(os.path.join("data", "RNMT.list"))]
data_frames = []
for classifier in classifiers:
    data_frame = None
    for draw_dir in draw_dirs:
        rank_num = draw_dir.split("_")[-1]
        file_path = os.path.join(source_dir, draw_dir, f'{classifier}_pr{rank_num}_all_genes.csv')
        tmp = pandas.read_csv(file_path, delimiter=',', index_col=0, low_memory=False)
        if data_frame is not None:
            data_frame += tmp
        else:
            data_frame = tmp
    data_frame /= rank_nums
    data_frame.sort_values(by='1', ascending=False, inplace=True)
    data_frame = data_frame.loc[~data_frame.index.isin(known_RNMTS)].head(270)
    data_frame["0"] = classifier
    data_frames.append(data_frame)

out = pandas.concat(data_frames)
out.to_csv(os.path.join(data_out_dir, data_out_name), index=True, sep='\t')
