from llama_index import VectorStoreIndex, SimpleDirectoryReader


print('我想将我的文件解析成较小的块')
from llama_index import ServiceContext
service_context = ServiceContext.from_defaults(chunk_size=1000)


documents = SimpleDirectoryReader('data').load_data()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

query_engine = index.as_query_engine()
Q = "What did the author do growing up?"
response = query_engine.query(Q)
print(Q)
print('------'*10)
print(response)