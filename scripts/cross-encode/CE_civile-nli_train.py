# pip install sentence_transformers
# pip install datasets

from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator
from sentence_transformers.readers import InputExample

import datasets
import pandas as pd

# from sklearn.utils import shuffle
# data = pd.read_csv('synthetic-dataset.csv')
# data = shuffle(data)
# data.to_csv('./synthetic-dataset.csv', index=False)

# restruture synthetic-dataset as flue to pass in load_dataset
df_file = {'train': 'synthetic-nli.csv'}
train_split = '10000'
test_split = '1000'

data = datasets.load_dataset("./", data_files=df_file, split = [f'train[:{train_split}]', f'train[-{test_split}:]'] )

train = data[0]
test = data[1]

# len(train)
# len(test)
# train[3:6]
# test[3:6]

""" Create Input Examples"""

train_samples = [InputExample(texts=[row['premise'], row['hypo']], label=row['label']) for row in train] 
test_sample = [InputExample(texts=[row['premise'], row['hypo']], label=row['label']) for row in test]

""" load model"""

label2int = {"contradiction": 0, "entailment": 1}
model = CrossEncoder('camembert-base', num_labels=2)
evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(test_sample)
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=8)

""" Train model"""

model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=7,
          evaluation_steps=500,
          save_best_model = True,
          warmup_steps=50,
          output_path="./output-nli-synthetic")
