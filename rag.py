"""
Module 13: RAG System
Problem solved: LLMs hallucinate without grounding in enterprise data.
Solution: RAG retrieves relevant documents before generation, reducing hallucination.
"""

import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import CHROMA_PERSIST_DIR, DATA_DIR

# Embedding model - runs locally, no API key needed
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embeddings():
    """Load HuggingFace embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_and_index_documents():
    """
    RAG Ingestion Pipeline:
    1. Load documents from data/ directory
    2. Split into chunks
    3. Embed each chunk
    4. Store in ChromaDB vector store
    """
    print(f"Loading documents from {DATA_DIR}...")
    loader = DirectoryLoader(DATA_DIR, glob="*.txt", loader_cls=TextLoader,
                            loader_kwargs={"encoding": "utf-8"})
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s)")

    # Split documents into chunks for better retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # each chunk = ~500 characters
        chunk_overlap=50      # 50 char overlap to preserve context between chunks
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # Embed chunks and store in ChromaDB
    print("Embedding and indexing chunks...")
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    print(f"Indexed {len(chunks)} chunks into ChromaDB at {CHROMA_PERSIST_DIR}")
    return vectorstore


def get_retriever():
    """
    Load existing ChromaDB and return a retriever.
    k=3 means top 3 most similar chunks are returned.
    """
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})
