
library(BiocManager) 
library(clusterProfiler)
library(org.Hs.eg.db)
library(dplyr)
C_DIR = getwd()
CLASSIFIERS <- c('GB','SVM','RF','LR','GNB')
scores <- read.csv(file.path( "result", "sorted_prob_genes.csv"), sep = "\t", header = TRUE,nrows = 1750)

for (classifier in CLASSIFIERS){
  c_name <- paste0(classifier, '_GAN_predict')
  c_gene_set <- scores[,c_name]
  if (classifier != 'GNB')
    c_gene_set <- c_gene_set[0:280]
  i <- 0
  result <- list()
  for (ont_ in c('BP')){
    enrich_result <- enrichGO(gene          = c_gene_set,
                              OrgDb         = org.Hs.eg.db,  # 使用人类数据库
                              keyType       = "SYMBOL",
                              ont           = ont_,          # 指定GO分支，如"BP"代表生物过程
                              pAdjustMethod = "BH",          # 多重检验校正方法，这里使用Benjamini-Hochberg方法
                              pvalueCutoff  = 0.05,           # 显著性水平
                              qvalueCutoff  = 0.05)
    result <-append(result,list(data.frame(enrich_result)))
  }
  final_df <- bind_rows(result)
  write.csv(final_df, file.path('result','GO_TERMS_REPORTS',paste0(classifier,'_GO_REPORT.csv')), row.names = TRUE)
}
