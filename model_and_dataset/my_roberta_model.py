from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import textattack

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained('./STmodel/white/roberta', model_max_length=256)
# 模型
model = AutoModelForSequenceClassification.from_pretrained('./STmodel/white/roberta', num_labels=2).to(device)
model = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
