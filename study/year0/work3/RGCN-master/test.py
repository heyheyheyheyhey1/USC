import torch
checkpoint = torch.load('./data/drug_disease_drug/best_mrr_model.pth')

print(checkpoint)