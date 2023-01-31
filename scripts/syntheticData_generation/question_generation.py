from transformers import T5ForConditionalGeneration, T5Tokenizer
model = T5ForConditionalGeneration.from_pretrained("JDBN/t5-base-fr-qg-fquad").to('cuda')
tokenizer = T5Tokenizer.from_pretrained("JDBN/t5-base-fr-qg-fquad")
import pandas as pd
from nltk import sent_tokenize
import re
from tqdm import tqdm
from datasets import Dataset
import torch
import json

data = pd.read_csv('../../dataset/civile-data.csv')
contexts = data['resume'].tolist()

def clean_input(inp):
    inp = inp.replace('generate question: ','')
    inp = inp.replace(' <hl> ','')
    
    return inp.replace('</s>','')

def get_inputs(context, sent):
    try: 
        st,end = re.search(sent.strip(),context).span()

        return f'generate question: {context[:st]} <hl> {context[st:end]} <hl> {context[end:]}</s>'
    except:
        return None


def generate(inputs, model, tokenizer):
    
    tokenized_input = tokenizer.batch_encode_plus(inputs,
            max_length=512,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            pad_to_max_length=True,
            return_tensors="pt")
    
    
    outputs = model.generate(input_ids=tokenized_input['input_ids'].to('cuda'),
            attention_mask=tokenized_input['attention_mask'].to('cuda'),
            max_length=128,
            num_beams=4,
        )
    
    questions = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

    return questions


inputs = []
for ind in tqdm(data.index):
    
    data_context_questions = {}
    did = data['id_file'][ind]
    
    sentences = data['resume'][ind]
    sent = re.split(';| \.', sentences)
    if " " in sent:
        sent.remove(' ')
    sent = [re.sub(r"\d", "", s, 1) for s in sent]

    sentences = " ".join(sent)
    sentences = sentences.strip()
    
    for s in sent:
        inp = get_inputs(sentences, s)
        if inp is not None:
            inputs.append([inp,did])


data_dict = {"id": [id for inp, id in inputs],"inputs": [inp for inp, id in inputs]}
dataset = Dataset.from_dict(data_dict)

dataloaders =  torch.utils.data.DataLoader(dataset, batch_size=8)

final =[]
for batch in tqdm(dataloaders):
    questions = generate(batch['inputs'], model,tokenizer)
    for i,c,q in zip(batch['id'], batch['inputs'],questions):
        final.append((i,c,q))
   


df = pd.DataFrame(final,columns=['ID','Context','Question'])

df.to_csv('questions.csv',index=False)

df = pd.read_csv('questions.csv')
df['Context'] = df['Context'].apply(lambda x: clean_input(x))

groups = df.groupby(['ID','Context'])
groups = {key[0]:{'context':key[1], 'questions': val['Question'].values.tolist()} for key,val in groups}

with open('generated-questions.json','w', encoding='utf-8') as file:
    json.dump(groups,file, ensure_ascii=False)


