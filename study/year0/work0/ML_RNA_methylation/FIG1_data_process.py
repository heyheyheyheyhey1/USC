import pandas
import os
source_dir = 'RMT_results'
draw_dirs = os.listdir(source_dir)
rank_nums = len(draw_dirs)
for draw_dir in draw_dirs:
    
