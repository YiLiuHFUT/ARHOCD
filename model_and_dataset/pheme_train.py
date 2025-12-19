import pandas as pd
import textattack

data = pd.read_csv('./dataset/PHEME/train.csv')
train_texts = data['text'].tolist()
train_labels = data['label'].tolist()
my_dataset = []
for i in range(len(train_texts)):
    my_dataset.append((train_texts[i], train_labels[i]))
dataset = textattack.datasets.Dataset(my_dataset)
