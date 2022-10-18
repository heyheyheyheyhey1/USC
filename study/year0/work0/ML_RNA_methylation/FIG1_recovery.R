setwd("C:\\USC\\study\\year0\\work0\\ML_RNA_methylation")
getwd()
library(ggplot2)
library(reshape2)
rm(list=ls())
data <- read.csv("data\\FIG1_data.txt",sep = "\t")
#View(data)
ggplot(data,aes(p,fill=Classifier, color=Classifier)) +
  xlab("Average Probility Score") +
  geom_density(alpha = 0.2) +
  theme_grey()
