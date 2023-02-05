import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import pandas as pd
import torch
# import faiss
from sentence_transformers import util, LoggingHandler
from sentence_transformers.cross_encoder import CrossEncoder
import streamlit as st



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



def convert_embeddings_to_faiss_index(embeddings, context_ids):
    """
    We take in a list of embeddings and a list of context IDs, convert the embeddings to a numpy array,
    instantiate a flat index, pass the index to IndexIDMap, add the embeddings and their IDs to the
    index, instantiate the resources, and move the index to the GPU
    
    :param embeddings: The embeddings you want to convert to a faiss index
    :param context_ids: The IDs of the contexts
    :return: A GPU index
    """
    embeddings = np.array(embeddings).astype("float32")  # Step 1: Change data type

    index = faiss.IndexFlatIP(embeddings.shape[1])  # Step 2: Instantiate the index
    index = faiss.IndexIDMap(index)  # Step 3: Pass the index to IndexIDMap

    index.add_with_ids(embeddings, context_ids)  # Step 4: Add vectors and their IDs

    res = faiss.StandardGpuResources()  # Step 5: Instantiate the resources
    gpu_index = faiss.index_cpu_to_gpu(
        res, 0, index
    )  # Step 6: Move the index to the GPU
    return gpu_index



def vector_search(query, model, index, num_results=20):
    """Tranforms query to vector using a pretrained, sentence-level
    model and finds similar vectors using FAISS.
    """
    vector = model.encode(list(query))
    D, I = index.search(np.array(vector).astype("float32"), k=num_results)
    return D, I


def id2details(df, I, column):
    """Returns the paper titles based on the paper index."""
    return [list(df[df.index.values == idx][column])[0] for idx in I[0]]


def combine(user_query, model, index, df, column, num_results=10):
    """
    It takes a user query, a model, an index, a dataframe, and a column name, and returns the top 5
    results from the dataframe
    
    :param user_query: the query you want to search for
    :param model: the model we trained above
    :param index: the index of the vectorized dataframe
    :param df: the dataframe containing the data
    :param column: the column in the dataframe that contains the text you want to search
    :param num_results: the number of results to return, defaults to 5 (optional)
    :return: the top 5 results from the vector search.
    """
    D, I = vector_search([user_query], model, index, num_results=num_results)
    return id2details(df, I, column)


def get_context(model, query, contexts, contexts_emb, top_k=100):
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
    # print(contexts)
    contexts_score_pairs = list(zip(contexts.premise.tolist(), scores))

    result = sorted(contexts_score_pairs, key=lambda x: x[1], reverse=True)[:top_k]
    # print(result)
    top_context = []
    for c, s in result:
        top_context.append(c)
    return top_context



def get_answer(model, query, context):
    """
    > Given a model, a query, and a context, return the answer
    
    :param model: the model we just loaded
    :param query: The question you want to ask
    :param context: The context of the question
    :return: A string
    """

    formatted_query = f"{query}\n{context}"
    res = model(formatted_query)
    return res[0]["generated_text"]



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
    predictions =  combine(question, model, index, contexts, "premise") if index else get_context(model, question, contexts, contexts_emb) #for cosine
        

    return predictions

@st.experimental_singleton
def load_models():

    semantic_search_model = load_semantic_search_model("distiluse-base-multilingual-cased-v1")

    model_nli_stsb = CrossEncoder('ssilwal/nli-stsb-fr', max_length=512, device='cuda')

    model_nli = CrossEncoder('ssilwal/CASS-civile-nli', max_length=512, device='cuda')

    model_baseline = CrossEncoder('amberoad/bert-multilingual-passage-reranking-msmarco', max_length=512, device='cuda')

    df = pd.read_csv('../cross-encode/synthetic-nli.csv')
    contexts = df.premise.unique()
    contexts = pd.DataFrame(contexts, columns = ['premise'])
    context_emb = np.loadtxt('../cross-encode/contexts-emb.txt', dtype=np.float32)

    return semantic_search_model, model_nli, model_nli_stsb, model_baseline, contexts, context_emb


def callback(state, object):
    return
    # st.session_state[f'{state}']


if 'slider' not in st.session_state:
    st.session_state['slider'] = 0

if 'radio' not in st.session_state:
    st.session_state['radio'] = 'Civile-Law-IR'

if 'show' not in st.session_state:
    st.session_state['show'] = False

if 'results' not in st.session_state:
    st.session_state['results'] = None

# if 'run' not in st.session_state:
#     st.session_state['run'] = True

# if 'radio' not in st.session_state:
#     st.session_state['radio'] = 'Model 1'


semantic_search_model, model_nli, model_nli_stsb, model_baseline, contexts, context_emb = load_models()

@st.cache(suppress_st_warning=True)
def run_inference(model_name, query):

    
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

    if model_name=='Civile-Law-IR':
        similarity_scores = model_nli.predict(sentence_combinations)
        scores = [(score_max[0],idx) for idx,score_max in enumerate(similarity_scores)]
        sim_scores_argsort = sorted(scores, key=lambda x: x[0], reverse=True)
        results = [pred[idx] for _,idx in list(sim_scores_argsort)[:int(top_K)]]

    if model_name=='STSB':
        similarity_scores = model_nli_stsb.predict(sentence_combinations)
        sim_scores_argsort = reversed(np.argsort(similarity_scores))
        results = [pred[idx] for idx in list(sim_scores_argsort)[:int(top_K)]]
    
    if model_name=='DR-Baseline':
        similarity_scores = model_baseline.predict(sentence_combinations)
        scores = [(score_max[0],idx) for idx,score_max in enumerate(similarity_scores)]
        sim_scores_argsort = sorted(scores, key=lambda x: x[0], reverse=True)
        results = [pred[idx] for _,idx in list(sim_scores_argsort)[:int(top_K)]]



    return results






# only need for faiss index
# index = convert_embeddings_to_faiss_index(context_emb, contexts.index.values)


# query = ['Quelles protections la Loi sur la protection du consommateur accorde-t-elle aux individus?']
query = st.text_input('Civil Legal Query', 'Quelles protections la Loi sur la protection du consommateur accorde-t-elle aux individus?')
top_K = st.text_input('Choose Number of Result: ','10')


model_name = st.radio(
        "Choose Model",
        ("Civile-Law-IR", "STSB", "DR-Baseline"),
         key='radio', on_change=callback, args=('radio','CivileLaw-IR'), help="Civile-Law-IR: trained on civile-NLI-dataset, STSB: trained on STSB french dataset, DR-Baseline: existing nli model trained on ms marco dataset"
    )


if st.button('Run', key='run'):

    results= run_inference(model_name, query)

    st.session_state['show'] = True
    st.session_state['results'] = results
    st.session_state['query'] = query
    model_dict = {'Civile-Law-IR': 'NLI-Syn', 'STSB': 'NLI-stsb', 'DR-Baseline': 'NLI-baseline'}
    st.session_state['model'] = model_dict[model_name]




if st.session_state['show'] and st.session_state['results']!=None:
    st.write("-"*50)
    for result in st.session_state['results']:

        line = f'Context: {result}\n\n'

        st.write(line)
