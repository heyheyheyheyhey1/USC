setwd("C:\\USC\\study\\year0\\work0\\ML_RNA_methylation")
getwd()
library(ggplot2)
library(reshape2)
library(patchwork) # 两张图拼起来
rm(list=ls())
data <- read.csv("data\\FIG1_data.txt",sep = "\t")

data_noGO <- read.csv("data\\FIG1_data_noGO.txt",sep = "\t")
#View(data)
# adjust 平滑曲线
get_plot <- function(data,sub,tag){
  p = ggplot(data,aes(p,fill=Classifier, color=Classifier)) +
    xlab("Average Probility Score") +
    geom_density(alpha = 0.2,adjust=1.2) +
    labs(subtitle = sub,tag = tag)+
    theme_grey()
  return (p)
}

p1 = get_plot(data,"Full Set","a)")
p2 = get_plot(data_noGO,"Reduce Set","b)")
p1 + p2 
