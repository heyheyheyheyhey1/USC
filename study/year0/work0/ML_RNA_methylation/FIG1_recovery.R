setwd("C:\\USC\\study\\year0\\work0\\ML_RNA_methylation")
getwd()
library(ggplot2)
library(reshape2)
install.packages("openxlsx")
library(openxlsx)
rm(list=ls())
data <- read.xlsx("density.xlsx")
data <- melt(data)
## Using gene as id variables
View(data)
# 使用geom_density函数绘制密度分布曲线
ggplot(data,aes(value,fill=C1, color=C1)) +
  xlab("Average Probility Score") +
  geom_density(alpha = 0.4) +
  geom_rug() + theme_bw()
