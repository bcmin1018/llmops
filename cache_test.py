import time

from langchain_community.storage import RedisStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
import os
os.environ["OPENAI_API_KEY"] = ""

underlying_embeddings = OpenAIEmbeddings()

# store = LocalFileStore("./cache/")
store = RedisStore(redis_url="redis://localhost:6379")

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace=underlying_embeddings.model
)

# raw_documents = TextLoader("./wpstest.txt", encoding="utf-8").load()
raw_documents = PyPDFLoader("./화장품법(법률)(제18448호)(20220218).pdf").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

start = time.time()
db = FAISS.from_documents(documents, cached_embedder)
end = time.time()
print(f"{end - start:.5f} sec")
print(list(store.yield_keys()))