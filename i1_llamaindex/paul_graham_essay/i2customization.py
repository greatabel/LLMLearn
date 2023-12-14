from llama_index import VectorStoreIndex, SimpleDirectoryReader


print("我想将我的文件解析成较小的块")
from llama_index import ServiceContext

service_context = ServiceContext.from_defaults(chunk_size=1000)


print("我想使用一个不同的向量存储库")
# pip3 install chromadb==0.4.0
import chromadb
from llama_index.vector_stores import ChromaVectorStore
from llama_index import StorageContext

chroma_client = chromadb.PersistentClient()
print("chroma_client.list_collections()=", chroma_client.list_collections())

if "quickstart" not in [
    collection.name for collection in chroma_client.list_collections()
]:
    chroma_collection = chroma_client.create_collection("quickstart")
else:
    chroma_collection = chroma_client.get_collection("quickstart")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)


documents = SimpleDirectoryReader("data").load_data()
# index = VectorStoreIndex.from_documents(documents, service_context=service_context)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
query_engine = index.as_query_engine(similarity_top_k=3)


query_engine = index.as_query_engine()
Q = "What did the author do growing up?"
response = query_engine.query(Q)
print(Q)
print("------" * 10)
print(response)

print("#" * 20)
for i, node_with_score in enumerate(response.source_nodes, 1):
    print(f"Top {i} Option:")
    print(node_with_score.node.text)  # 打印节点的文本内容
    print("Score:", node_with_score.score)  # 打印节点的得分
    print("#" * 20)
