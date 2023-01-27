#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !python -m nltk.downloader punkt


# In[2]:


# !git clone https://github.com/patil-suraj/question_generation.git


# In[1]:


import os
import pandas as pd
from os import listdir
from os.path import isfile, join
import re


# In[2]:


print(os.getcwd())


# In[3]:


os.chdir("../")
print(os.getcwd())


# In[4]:


from pipeline_fr import pipeline_fr


# In[5]:


data_dir = "Datasets/CASS-dataset/Freemium_cass_global_20220417-170000/cass/global/civile"


# # Run this cell and the next one just one time to generate the 'cases_after_2000.txt' file
# 
# I uploaded this file on git so normally you should find it there
# 
# Filter and select just the cases after year 2000 

# In[6]:


import re
"""
files_after_2000 = []
i=0
for path, _, files in os.walk(data_dir):
    for name in files:
        if i<1000:
            with open(os.path.join(path, name), 'r', encoding='utf-8', errors='ignore') as myfile:
                full_text = myfile.read()
                date = None
                
                if re.search('<META_JURI>.*</META_JURI>', full_text, re.DOTALL):
                    
                    search = re.search('(?<=<DATE_DEC>).*?(?=</DATE_DEC>)', full_text, re.DOTALL)
                    date = search.group(0)
                    y, m, d = date.split("-")
                    if (int(y) >= 2000):
                        files_after_2000.append(name)
                        
"""


# Write all these file names in a .txt file

# In[7]:


"""
text_file = open("cases_after_2000.txt", "wt")
for case in files_after_2000:
    case = case.replace(".xml", "")
    text_file.write(case)
    text_file.write("\n")
    
text_file.close()

"""


# Dont't forget to change the path if needed:

# In[8]:


path_cases_file = open("cases_after_2000.txt", "r")
cases_file = path_cases_file.read()
str_files = cases_file.split("\n")


# In[9]:


print(len(str_files))


# In[10]:


df = pd.DataFrame(columns = ["id_file", "decision", "resume"])
df


# In[11]:


def check_file(file):
    file = file.replace(".story", "")
    if file in str_files:
        return 1
    
    return 0


# Check if a .story file is after year 2000,  read it and save the content in a vector

# In[12]:


path = "env/cleaned_files_civile"
content_file = []
file_content_assoc={}
for file in os.listdir(path):
    if check_file(file):
        with open(os.path.join(path, file), "r") as f:
            content = f.read()
            content_file.append(content)
            file_content_assoc[file] = content
        


# Apparently in the "cleaned_files_civile" directory there were less files after year 2000:
# - in the original dataset: 17314
# - from the generate .story files: 16256
# 
# Maybe there was a bug during their generation and some of them were lost. It is not such a problem, it is just informative

# In[13]:


len(file_content_assoc)


# Split each file by "@highlight" and save the decision and resume in differet variables

# In[15]:


decision, resume = [], []
for f in content_file:
    d, r = f.split("@highlight")
    decision.append(d)
    resume.append(r)


# Delete special characters from decision and resume (for example "\n")

# In[36]:


def delete_special_characters(d):
    new_d = []
    for item in d:
        item = item.replace("\n", "")
        item = item.replace("' ", "'")
        item = item.replace('\\' , '' )
        #item = item.replace('. ' , '.' )
        new_d.append(item)
    return new_d


# In[37]:


decision = delete_special_characters(decision)
resume = delete_special_characters(resume)
print(len(decision), len(resume))


# In[38]:


df_files = pd.DataFrame(list(zip( list(file_content_assoc.keys()), decision, resume)), columns = ["id_file", "decision", "resume"])
df_files
