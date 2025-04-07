import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from vector import retriever  # Import the retriever from vector.py
import os

# Initialize the language model
model = OllamaLLM(model="llama3.2")

# Define the prompt template
template = """
You are a professional baby care expert who provides helpful, practical, and evidence-based advice to caregivers.

Based on the following studies and articles, please answer the user's question in a friendly manner:

Relevant information:
{studies}

Question:
{question}

Please provide your expert response below.
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Configure Streamlit UI
st.set_page_config(page_title="Baby Care Bot", layout="centered")
st.title("Baby Care Bot")
st.write("Welcome! Ask me any baby care-related question and I'll provide evidence-based advice to help you.")

# User input
question = st.text_input("Enter your baby care question:")

if question:
    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)
    
    # Prepare the studies text for the prompt (concatenating document content)
    studies = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Get the expert advice from the model
    result = chain.invoke({"studies": studies, "question": question})
    
    st.subheader("Expert Advice")
    st.write(result)
    
    # Display unique relevant document links and snippets
    st.subheader("Relevant Documents")
    displayed_docs = set()
    for doc in retrieved_docs:
        doc_source = doc.metadata.get("source", "Unknown")
        file_name = os.path.basename(doc_source)
        if doc_source in displayed_docs:
            continue  # Skip if already displayed
        displayed_docs.add(doc_source)
        if os.path.exists(doc_source):
            st.markdown(f"**Document:** [{file_name}]({doc_source})")
        else:
            st.write(f"Document: {file_name}")
else:
    st.write("Please enter a question in the text box above.")

st.sidebar.write("## About")
st.sidebar.write("This is a baby care bot that uses FAISS as a vector store to retrieve relevant information from PDFs.")
