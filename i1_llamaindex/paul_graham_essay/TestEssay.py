#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

# os.environ["OPENAI_API_KEY"] = "INSERT OPENAI KEY"
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# In[24]:


# https://github.com/run-llama/llama_index/blob/main/examples/paul_graham_essay/TestEssay.ipynb


# In[2]:


from llama_index import TreeIndex, SimpleDirectoryReader
from IPython.display import Markdown, display


# In[3]:


documents = SimpleDirectoryReader("data").load_data()


# In[4]:


new_index = TreeIndex.from_documents(documents)


# In[5]:


# set Logging to DEBUG for more detailed outputs
query_engine = new_index.as_query_engine()
response = query_engine.query("What did the author do growing up?")


# In[6]:


display(Markdown(f"<b>{response}</b>"))


# In[7]:


# set Logging to DEBUG for more detailed outputs
response = query_engine.query("What did the author do after his time at Y Combinator?")


# In[10]:


display(Markdown(f"<b>{response}</b>"))


# # Build Tree Index with a custom Summary Prompt, directly retrieve answer from root node

# In[11]:


from llama_index.prompts import PromptTemplate


# In[13]:


documents = SimpleDirectoryReader("data").load_data()

query_str = "What did the author do growing up?"
SUMMARY_PROMPT_TMPL = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the context information and not prior knowledge, "
    f"answer the question: {query_str}\n"
)
SUMMARY_PROMPT = PromptTemplate(SUMMARY_PROMPT_TMPL)
index_with_query = TreeIndex.from_documents(documents, summary_template=SUMMARY_PROMPT)


# In[16]:


# directly retrieve response from root nodes instead of traversing tree
query_engine = index_with_query.as_query_engine(retriever_mode="root")
response = query_engine.query(query_str)


# In[17]:


display(Markdown(f"<b>{response}</b>"))


# # Using GPT Keyword Table Index

# In[18]:


from llama_index import KeywordTableIndex, SimpleDirectoryReader
from IPython.display import Markdown, display


# In[19]:


# build keyword index
documents = SimpleDirectoryReader("data").load_data()
index = KeywordTableIndex.from_documents(documents)


# In[20]:


# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do after his time at Y Combinator?")


# In[21]:


display(Markdown(f"<b>{response}</b>"))


# # Using GPT List Index

# In[22]:


from llama_index import SummaryIndex, SimpleDirectoryReader
from IPython.display import Markdown, display


# In[ ]:


# build summary index
documents = SimpleDirectoryReader("data").load_data()
index = SummaryIndex.from_documents(documents)


# In[ ]:


# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do after his time at Y Combinator?")


# In[23]:


display(Markdown(f"<b>{response}</b>"))


# In[ ]:





# In[ ]:




