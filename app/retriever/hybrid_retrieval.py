from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
import threading
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

embed_model = HuggingFaceEmbeddings(
  model_name =  "sentence-transformers/all-MiniLM-L6-v2"
)

from .process_docs import process_document



def vec_store_retriever(chunks):
  vec_store = FAISS.from_documents(chunks, embed_model)

  retriever = vec_store.as_retriever(
    search_type = 'mmr', search_kwargs = {
      'k': 6,
      'lambda_mult' : 0.6
    }
  )
  return retriever




def bm25_retriever(chunks):
  bm25 = BM25Retriever.from_documents(chunks)

  bm25.k = 6

  return bm25

def ensemble_retriever(chunks):
  bm25 = bm25_retriever(chunks)
  vec_store = vec_store_retriever(chunks)

  retriever = EnsembleRetriever(
    retrievers=[bm25, vec_store], weights = [0.5,0.5]
  )

  return retriever




RETRIEVERS = {}

def register_retriever(user_id, retriever):
  RETRIEVERS[user_id] = retriever
  print("DONE")

def get_retriever(user_id):
  print(user_id)
  print(RETRIEVERS.get(user_id))
  return RETRIEVERS.get(user_id)


def deduplicate_docs(docs):
  seen = set()

  unique_docs = []

  for doc in docs:
    content = doc.page_content.strip()

    if content not in seen:
      seen.add(content)
      unique_docs.append(doc)

  return unique_docs


