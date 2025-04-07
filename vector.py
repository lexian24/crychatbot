import os
import pdfplumber
import json
from langchain.documents import Document
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS

# Initialize embeddings model (ensure your model is set up correctly)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Define folders
pdf_folder = "./pdfs"
save_folder = "./saved_links"
os.makedirs(save_folder, exist_ok=True)
links_file = os.path.join(save_folder, "links.json")

documents = []
links_data = {}

# Process each PDF
for pdf_file in os.listdir(pdf_folder):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.abspath(os.path.join(pdf_folder, pdf_file))
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    doc_id = f"{pdf_file}_{page_number}"
                    document = Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path,
                            "page": page_number,
                            "file_name": pdf_file
                        },
                        id=doc_id
                    )
                    documents.append(document)
                    links_data[doc_id] = {
                        "source": pdf_path,
                        "page": page_number,
                        "file_name": pdf_file,
                        "snippet": text[:300]  # snippet for reference
                    }

# Save links data to JSON for persistent reference
with open(links_file, 'w', encoding='utf-8') as f:
    json.dump(links_data, f, ensure_ascii=False, indent=4)

# Create the FAISS vector store from documents
vector_store = FAISS.from_documents(documents, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
