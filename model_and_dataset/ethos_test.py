import pandas as pd
import textattack

data = pd.read_csv('./dataset/ethos/test.csv')
test_texts = data['text'].tolist()
test_labels = data['label'].tolist()
my_dataset = []
for i in range(len(test_texts)):
    my_dataset.append((test_texts[i], test_labels[i]))
dataset = textattack.datasets.Dataset(my_dataset)
