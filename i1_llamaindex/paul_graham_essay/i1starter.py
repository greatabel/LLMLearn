# https://docs.llamaindex.ai/en/stable/getting_started/starter_example.html
import os.path
from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage

# check if storage already exists
if (not os.path.exists('./storage')):
    print('load the documents and create the index')
    documents = SimpleDirectoryReader('data').load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist()
else:
    print('load the existing index')
    storage_context = StorageContext.from_defaults(persist_dir='./storage')
    index = load_index_from_storage(storage_context)

# either way we can now query the index
query_engine = index.as_query_engine()
Q = "What did the author do growing up?"
response = query_engine.query(Q)
print(Q)
print('\n','------'*10)
print(response)