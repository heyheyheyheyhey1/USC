import pandas
import os
source_dir = 'RMT_results'
classifiers = ["GB","GNB","LR","RF","SVM"]
draw_dirs = os.listdir(source_dir)
draw_dirs = [d for d in draw_dirs if d.__contains__("draw_")]
rank_nums = len(draw_dirs)
for draw_dir in draw_dirs:
    rank_num = draw_dir.split("_")[-1]
