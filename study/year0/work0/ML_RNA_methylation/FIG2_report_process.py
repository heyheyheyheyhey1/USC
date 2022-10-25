import os
import pandas
src_dir = os.path.join("data","GO_REPORTS")
data_out_name = 'FIG2_REPORT_fix.csv'
report_datas = []
for report_dir in os.listdir(src_dir):
    report = pandas.read_csv(os.path.join(src_dir, report_dir), sep="\t")
    report.sort_values("Count", ascending=False, inplace=True)
    report = report.head(20)
    report["Term"] = [t.split("~")[-1] for t in report["Term"]]
    report["Classifier"] = report_dir.split(".")[0]
    report_datas.append(report)
report = pandas.concat(report_datas)
report.to_csv(os.path.join("data", data_out_name), sep='\t', index=False)
