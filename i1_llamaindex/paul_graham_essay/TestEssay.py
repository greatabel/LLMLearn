#!/usr/bin/env python
# coding: utf-8

# In[58]:


import os

# os.environ["OPENAI_API_KEY"] = "INSERT OPENAI KEY"
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# In[59]:


# https://github.com/run-llama/llama_index/blob/main/examples/paul_graham_essay/TestEssay.ipynb
# print(os.environ["OPENAI_API_KEY"])


# In[60]:


from llama_index import TreeIndex, SimpleDirectoryReader
from IPython.display import Markdown, display


# In[61]:


# 计数

import tiktoken
from llama_index.llms import Anthropic
from llama_index.callbacks import CallbackManager, TokenCountingHandler

from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    set_global_service_context,
)

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext

# 设置tokenizer和TokenCountingHandler
tokenizer = tiktoken.encoding_for_model("text-davinci-003").encode

token_counter = TokenCountingHandler(tokenizer=tokenizer)
callback_manager = CallbackManager([token_counter])

service_context = ServiceContext.from_defaults(callback_manager=callback_manager)
set_global_service_context(service_context)


# In[62]:


documents = SimpleDirectoryReader("data").load_data()


# In[63]:


new_index = TreeIndex.from_documents(documents)


# In[64]:


print(
    "Embedding Tokens: ",
    token_counter.total_embedding_token_count,
    "\n",
    "LLM Prompt Tokens: ",
    token_counter.prompt_llm_token_count,
    "\n",
    "LLM Completion Tokens: ",
    token_counter.completion_llm_token_count,
    "\n",
    "Total LLM Token Count: ",
    token_counter.total_llm_token_count,
    "\n",
)


# In[65]:


# set Logging to DEBUG for more detailed outputs
query_engine = new_index.as_query_engine()
response = query_engine.query("What did the author do growing up?")


# In[66]:


display(Markdown(f"<b>{response}</b>"))


# In[67]:


# set Logging to DEBUG for more detailed outputs
response = query_engine.query("What did the author do after his time at Y Combinator?")


# In[68]:


display(Markdown(f"<b>{response}</b>"))


# In[69]:


print(
    "Embedding Tokens: ",
    token_counter.total_embedding_token_count,
    "\n",
    "LLM Prompt Tokens: ",
    token_counter.prompt_llm_token_count,
    "\n",
    "LLM Completion Tokens: ",
    token_counter.completion_llm_token_count,
    "\n",
    "Total LLM Token Count: ",
    token_counter.total_llm_token_count,
    "\n",
)


# # Build Tree Index with a custom Summary Prompt, directly retrieve answer from root node

# In[70]:


from llama_index.prompts import PromptTemplate


# In[71]:


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


# In[72]:


# directly retrieve response from root nodes instead of traversing tree
query_engine = index_with_query.as_query_engine(retriever_mode="root")
response = query_engine.query(query_str)


# In[73]:


display(Markdown(f"<b>{response}</b>"))


# # Using GPT Keyword Table Index

# In[74]:


from llama_index import KeywordTableIndex, SimpleDirectoryReader
from IPython.display import Markdown, display


# In[ ]:


# build keyword index
documents = SimpleDirectoryReader("data").load_data()
index = KeywordTableIndex.from_documents(documents)


# In[ ]:


# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do after his time at Y Combinator?")


# In[ ]:


display(Markdown(f"<b>{response}</b>"))


# # Using GPT List Index

# In[ ]:


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


# In[ ]:


display(Markdown(f"<b>{response}</b>"))


# In[ ]:


print(
    "Embedding Tokens: ",
    token_counter.total_embedding_token_count,
    "\n",
    "LLM Prompt Tokens: ",
    token_counter.prompt_llm_token_count,
    "\n",
    "LLM Completion Tokens: ",
    token_counter.completion_llm_token_count,
    "\n",
    "Total LLM Token Count: ",
    token_counter.total_llm_token_count,
    "\n",
)


# In[ ]:





# In[ ]:




