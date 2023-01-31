# pip install sentence_transformers
# pip install datasets

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
import numpy as np
import pandas as pd
import torch
# import faiss

from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator, CECorrelationEvaluator
from sentence_transformers.readers import InputExample

import datasets
from tqdm import tqdm

def get_embeddings_from_contexts(model, contexts): # for embeddings
    """
    It takes a list of contexts and returns a list of embeddings
    
    :param model: the model you want to use to get the embeddings
    :param contexts: a list of strings, each string is a context
    :return: The embeddings of the contexts
    """
    return model.encode(contexts)

def load_semantic_search_model(model_name):
    """
    It loads the model
    
    :param model_name: The name of the model to load
    :return: A sentence transformer object
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)

def get_context(model, query, contexts, contexts_emb, top_k=50):
    """
    Given a query, a list of contexts, and their embeddings, return the top k contexts with the highest
    similarity score.
    
    :param model: the model we trained in the previous section
    :param query: the query string
    :param contexts: list of contexts
    :param contexts_emb: the embeddings of the contexts
    :param top_k: the number of contexts to return, defaults to 3 (optional)
    :return: The top_context is a list of the top 3 contexts that are most similar to the query.
    """
    # Encode query and contexts with the encode function
    query_emb = model.encode(query)
    query_emb = torch.from_numpy(query_emb.reshape(1, -1))
    contexts_emb = torch.from_numpy(contexts_emb)
    # Compute similiarity score between query and all contexts embeddings
    scores = util.cos_sim(query_emb, contexts_emb)[0].cpu().tolist()
    # Combine contexts & scores
    contexts_score_pairs = list(zip(contexts.premise.tolist(), scores))

    result = sorted(contexts_score_pairs, key=lambda x: x[1], reverse=True)[:top_k]

    top_context = []
    for c, s in result:
        top_context.append(c)
    return top_context

def evaluate_semantic_model(model, question, contexts, contexts_emb, index=None):

    """
    For each question, we use the model to find the most similar context.
    
    :param model: the model we're using to evaluate
    :param questions: a list of questions
    :param contexts: the list of contexts
    :param contexts_emb: the embeddings of the contexts
    :param index: the index of the context embeddings
    :return: The predictions are being returned.
    """
    predictions = get_context(model, question, contexts, contexts_emb)

    return predictions

semantic_search_model = load_semantic_search_model("distiluse-base-multilingual-cased-v1") # model unseen by the our trained models before

df = pd.read_csv('synthetic-nli.csv')
contexts = df.premise.unique()
contexts = pd.DataFrame(contexts, columns = ['premise'])
#encode raw contexts to embedding vectors
context_emb = np.loadtxt('contexts-emb.txt', dtype=np.float32)

model_synthetic = CrossEncoder('ssilwal/CASS-civile-nli', max_length=512)
model_stsb = CrossEncoder('ssilwal/nli-stsb-fr', max_length=512)
model_baseline = CrossEncoder('amberoad/bert-multilingual-passage-reranking-msmarco', max_length=512)

def run_inference(model, model_name, query, top_K=15):

    
    pred = evaluate_semantic_model(
        semantic_search_model,
        query,
        contexts,
        context_emb,
        # index,
        #  #if u want to use faiss
    )


    # So we create the respective sentence combinations
    sentence_combinations = [[query, corpus_sentence] for corpus_sentence in pred]

    # Compute the similarity scores for these combinations

    if model_name=='Model 1':
        similarity_scores = model.predict(sentence_combinations)
        scores = [(score_max[0],idx) for idx,score_max in enumerate(similarity_scores)]
        sim_scores_argsort = sorted(scores, key=lambda x: x[0], reverse=True)
        results = [pred[idx] for _,idx in list(sim_scores_argsort)[:int(top_K)]]

    if model_name=='Model 2':
        similarity_scores = model_stsb.predict(sentence_combinations)
        sim_scores_argsort = reversed(np.argsort(similarity_scores))
        results = [pred[idx] for idx in list(sim_scores_argsort)[:int(top_K)]]
    
    if model_name=='Model 3':
        similarity_scores = model.predict(sentence_combinations)
        scores = [(score_max[0],idx) for idx,score_max in enumerate(similarity_scores)]
        sim_scores_argsort = sorted(scores, key=lambda x: x[0], reverse=True)
        results = [pred[idx] for _,idx in list(sim_scores_argsort)[:int(top_K)]]
    if model_name=='Baseline':
        return pred[:int(top_K)]



    return results


df_file = {'train': 'synthetic-nli.csv'}

data = datasets.load_dataset("./", data_files=df_file, split = [f'train[19000:20000]'] )

queries = data[0]['hypo']

semantic_search_newmodel = load_semantic_search_model("nreimers/MiniLM-L6-H384-uncased")

def get_score(model, query, preds):
    # Encode query and contexts with the encode function
    query_emb = model.encode(query)
    query_emb = torch.from_numpy(query_emb.reshape(1, -1))
    pred_embed = get_embeddings_from_contexts(model, preds)
    preds_emb = torch.from_numpy(pred_embed)
    # Compute similiarity score between query and all contexts embeddings
    scores = util.cos_sim(query_emb, preds_emb)[0].cpu().detach().numpy()
    # print(scores)
    scores = np.mean(scores, axis=0)
    return scores

scores1 = [get_score(semantic_search_newmodel, query, run_inference(model_synthetic, 'Model 1', query)) for query in tqdm(queries)]

scores2 = [get_score(semantic_search_newmodel, query, run_inference(model_stsb, 'Model 2', query)) for query in tqdm(queries)]

scores3 = [get_score(semantic_search_newmodel, query, run_inference(model_baseline, 'Model 3', query)) for query in tqdm(queries)]


avg_scores1 = np.mean(scores1)
print(avg_scores1)
std_scores1 = np.std(scores1)
print(std_scores1)

avg_scores2 = np.mean(scores2)
print(avg_scores2)
std_scores2 = np.std(scores2)
print(std_scores2)

avg_scores3 = np.mean(scores3)
print(avg_scores3)
std_scores3 = np.std(scores3)
print(std_scores3)

# import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure

# figure(figsize=(10, 8), dpi=80)
# plt.plot(scores1, label='Civile-Law')
# plt.plot(scores2, label='STSB')
# plt.plot(scores3, label='Dense-retrieval baseline')
# plt.legend()
# plt.xlabel('Query index')
# plt.ylabel('Average similarity score')
# plt.show()

