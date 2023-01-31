import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
import numpy as np
import pandas as pd
import torch
import faiss
import json
from sentence_transformers import util
from tqdm import tqdm

#cleaning the tags on questions.csv
def clean_input(inp):
    inp = inp.replace('generate question: ','')
    inp = inp.replace(' <hl> ','')
    
    return inp.replace('</s>','')

#reading and creating the first half of the synthetic-nli dataset
df = pd.read_csv('questions.csv')
df['Context'] = df['Context'].apply(lambda x: clean_input(x))
new_df = df.drop(columns=['ID']).rename(columns={'Context' : 'premise', 'Question' : 'hypo'})
new_df['label']='1'
new_df.to_csv('synthetic-nli.csv',index=False)

def get_embeddings_from_contexts(model, contexts): # for embeddings
    """
    It takes a list of contexts and returns a list of embeddings
    
    :param model: the model you want to use to get the embeddings
    :param contexts: a list of strings, each string is a context
    :return: The embeddings of the contexts
    """
    return model.encode(contexts, convert_to_tensor=True)

def load_semantic_search_model(model_name):
    """
    It loads the model
    
    :param model_name: The name of the model to load
    :return: A sentence transformer object
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)

def get_context(query_emb, contexts, contexts_emb, top_k=2):
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

    query_emb = query_emb.reshape(1, -1)

    scores = util.cos_sim(query_emb, contexts_emb)[0].cpu().tolist()

    contexts_score_pairs = list(zip(contexts, scores))

    result = sorted(contexts_score_pairs, key=lambda x: x[1])[:top_k]

    top_context = []
    for c, s in result:
        top_context.append(c)
    return top_context

semantic_search_model = load_semantic_search_model("distiluse-base-multilingual-cased-v1") # or all-mpnet-base-v2

df = pd.read_csv('synthetic-nli.csv')
#encode raw contexts to embedding vectors
questions_emb = get_embeddings_from_contexts(
    semantic_search_model, df.hypo.values
)

# contexts = df.sample(n=9000)
contexts = df.premise.unique().tolist()

context_emb = get_embeddings_from_contexts(
    semantic_search_model, contexts
)

# only need for faiss index
# index = convert_embeddings_to_faiss_index(questions_emb, df.index.values)
questions_emb = questions_emb.to('cuda')
questions_emb = util.normalize_embeddings(questions_emb)

context_emb = context_emb.to('cuda')
context_emb = util.normalize_embeddings(context_emb)

data = []
def get_irrelevent(context, emb):

#     print(context.shape)
    pred = get_context(
        emb,
        df.hypo.values.tolist(),
        questions_emb,
    )

    for p in pred:
        data.append((context,p,0))


for idx, context in tqdm(enumerate(contexts)):
    get_irrelevent(context, context_emb[idx])

df_irr = pd.DataFrame(data,columns=['premise','hypo','label'])
df = df.drop_duplicates()

merged = pd.concat([df,df_irr], axis=0)
merged = merged.drop_duplicates()

merged.to_csv('synthetic-nli.csv',index=False)

