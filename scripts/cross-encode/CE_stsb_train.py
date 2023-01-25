# pip install sentence_transformers

# pip install datasets

from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers.readers import InputExample

import datasets
import pandas as pd

train = datasets.load_dataset('stsb_multi_mt','fr', split='train')
test = datasets.load_dataset('stsb_multi_mt','fr', split='test')

""" Create Input Examples"""

train_samples = [InputExample(texts=[row['sentence1'], row['sentence2']], label=float(row['similarity_score']/5.0)) for row in train] #similarity score
test_sample = [InputExample(texts=[row['sentence1'], row['sentence2']], label=float(row['similarity_score']/5.0)) for row in test]

""" load model"""

model = CrossEncoder('camembert-base', num_labels=1)
evaluator = CECorrelationEvaluator.from_input_examples(test_sample)
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=8)

""" Train model"""

model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=7,
          evaluation_steps=100,
          save_best_model = True,
          warmup_steps=50,
          output_path="./output-nli-stsb")