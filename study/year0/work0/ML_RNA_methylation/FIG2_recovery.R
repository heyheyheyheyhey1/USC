setwd("C:\\USC\\study\\year0\\work0\\ML_RNA_methylation")
library(ggplot2)
library(reshape2)
data <- read.csv("data\\FIG2_REPORT.TXT",sep = "\t")
data = head(data,30)
x = data$Fold.Enrichment
y = factor(data$Term,levels = data$Term)

p = ggplot(data,aes(x,y))

p1 = p + geom_point(aes(size = Count,color=p.adjust(PValue,method = "BH"))) + 
   scale_color_gradient(low = "Red",high = "Blue") + 
   labs(x="",y="",color="p.adjust")
p1
