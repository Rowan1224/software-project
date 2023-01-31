import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import pandas as pd
import torch
import faiss
import json
from sentence_transformers import util

def read_json(filepath):
    with open(filepath, 'r') as f:
        file = json.load(f)
        contexts = []
        questions = []
        for data in file:
            for context in data['items']:
                text = context['context']
                contexts.append(text)
                if len(context['questions']) > 0:
                    for q in context['questions']:
                        ques = q['question']
                        questions.append(ques)
    return contexts, questions
        

def get_embeddings_from_contexts(model, contexts): # for embeddings
    return model.encode(contexts)

def load_semantic_search_model(model_name):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def convert_embeddings_to_faiss_index(embeddings, context_ids):
    embeddings = np.array(embeddings).astype("float32")  # Step 1: Change data type

    index = faiss.IndexFlatIP(embeddings.shape[1])  # Step 2: Instantiate the index
    index = faiss.IndexIDMap(index)  # Step 3: Pass the index to IndexIDMap

    index.add_with_ids(embeddings, context_ids)  # Step 4: Add vectors and their IDs

    # res = faiss.StandardGpuResources()  # Step 5: Instantiate the resources
    # gpu_index = faiss.index_cpu_to_gpu(
    #     res, 0, index
    # )  # Step 6: Move the index to the GPU
    return index


def vector_search(query, model, index, num_results=10):
    """Tranforms query to vector using a pretrained, sentence-level
    model and finds similar vectors using FAISS.
    """
    vector = model.encode(list(query))
    D, I = index.search(np.array(vector).astype("float32"), k=num_results)
    return D, I


def id2details(df, I, column):
    """Returns the paper titles based on the paper index."""
    return [list(df[df.index.values == idx][column]) for idx in I[0]]


def combine(user_query, model, index, df, column, num_results=1):
    D, I = vector_search([user_query], model, index, num_results=num_results)
    return id2details(df, I, column)[0][0]


def get_context(model, query, contexts, contexts_emb):
    # Encode query and contexts with the encode function
    query_emb = model.encode(query)
    query_emb = torch.from_numpy(query_emb.reshape(1, -1))
    contexts_emb = torch.from_numpy(contexts_emb)
    # Compute similiarity score between query and all contexts embeddings
    scores = util.cos_sim(query_emb, contexts_emb)[0].cpu().tolist()
    # Combine contexts & scores
    contexts_score_pairs = list(zip(contexts, scores))

    return max(contexts_score_pairs, key=lambda x: x[1])[0]


def get_answer(model, query, context):
    formatted_query = f"{query}\n{context}"
    res = model(formatted_query)
    return res[0]["generated_text"]




def evaluate_semantic_model(
    model, questions, contexts, contexts_emb, index=None
):
    predictions = [
        combine(question, model, index, contexts, "contexts") #for faiss
        if index
        else get_context(model, question, contexts, contexts_emb) #for cosine
        for question in questions
    ]

    return predictions



if __name__ == "__main__":
    semantic_search_model = load_semantic_search_model("all-mpnet-base-v2")
    contexts, questions = read_json('../syntheticData_generation/generated-questions.json') # load context
    if_faiss = True

    if if_faiss:
        contexts = pd.DataFrame(contexts, columns=["contexts"])
        #encode raw contexts to embedding vectors
        contexts_emb = get_embeddings_from_contexts(
            semantic_search_model, contexts.contexts.values
        )
# only need fot faiss index
        index = convert_embeddings_to_faiss_index(contexts_emb, contexts.index.values)
    else: #cosine similarity
        contexts_emb = get_embeddings_from_contexts(semantic_search_model, contexts)
        index = None

    test_questions = questions[1]

    # start = time.time()
    print(evaluate_semantic_model(
        semantic_search_model,
        test_questions,
        contexts,
        contexts_emb,
        index,
    ))
