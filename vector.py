from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pdfplumber

import sys
try:
    import pysqlite3 as sqlite3  # Forcefully use pysqlite3
    sys.modules["sqlite3"] = sqlite3
except ImportError:
    import sqlite3  # Fallback to the standard sqlite3 if pysqlite3 is not available

# Initialize the embeddings model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Define your PDF folder path and the Chroma database location
pdf_folder = "./pdfs"
db_location = "./chrome_langchain_db"

add_documents = not os.path.exists(db_location)

documents = []
ids = []

if add_documents:
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)

            with pdfplumber.open(pdf_path) as pdf:
                for page_number, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:  # Only add pages with text content
                        doc_id = f"{pdf_file}_{page_number}"
                        document = Document(
                            page_content=text,
                            metadata={"source": pdf_path, "page": page_number},
                            id=doc_id
                        )
                        documents.append(document)
                        ids.append(doc_id)

vector_store = Chroma(
    collection_name="baby_care",
    persist_directory=db_location,
    embedding_function=embeddings,
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})
