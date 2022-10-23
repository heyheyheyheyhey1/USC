setwd("C:\\USC\\study\\year0\\work0\\ML_RNA_methylation")
library(ggplot2)
library(reshape2)
data <- read.csv("data\\FIG2_REPORT_fix.csv",sep = "\t")
x = data$Classifier
y = factor(data$Term,levels = unique(data$Term))
p = ggplot(data,aes(x,y))

#color = -log10(PValue)

p1 = p + geom_point(aes(size = Count,color=p.adjust(PValue,method = "BH"))) + 
   scale_color_gradient(low = "Red",high = "Blue") + 
   theme_bw() +
   labs(x="",y="",color="p.adjust")
p1
