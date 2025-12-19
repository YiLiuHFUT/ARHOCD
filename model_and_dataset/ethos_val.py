import pandas as pd
import textattack

data = pd.read_csv('./dataset/ethos/val.csv')
val_texts = data['text'].tolist()
val_labels = data['label'].tolist()
my_dataset = []
for i in range(len(val_texts)):
    my_dataset.append((val_texts[i], val_labels[i]))
dataset = textattack.datasets.Dataset(my_dataset)
