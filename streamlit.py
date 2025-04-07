import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from vector import retriever  # Assuming vector.py contains the vector store setup code
import os

# Initialize model
model = OllamaLLM(model="llama3.2")

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

# Streamlit UI
st.set_page_config(page_title="Baby Care Bot", layout="centered")
st.title("Baby Care Bot Testing")
st.write("Welcome! Ask me any baby care-related question and I'll provide evidence-based advice to help you.")

# User input
question = st.text_input("Enter your baby care question:")

if question:
    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)
    
    # Prepare the studies text for the prompt
    studies = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Generate the response from the model
    result = chain.invoke({"studies": studies, "question": question})

    # Display the result
    st.subheader("Expert Advice")
    st.write(result)

    # Display the relevant documents and their links
    st.subheader("Relevant Documents")
    
    # Use a set to keep track of which documents have already been displayed
    displayed_docs = set()
    
    for doc in retrieved_docs:
        doc_source = doc.metadata.get('source', 'Unknown')
        file_name = os.path.basename(doc_source)
        
        # If the document has already been displayed, skip it
        if doc_source in displayed_docs:
            continue
        
        # Otherwise, add the document to the set and display it
        displayed_docs.add(doc_source)
        
        # Display document snippet and clickable link
        if os.path.exists(doc_source):
            st.write(f"**Document:** [{file_name}]({doc_source})")
        else:
            st.write(f"Document: {file_name}")
        
st.sidebar.write("## About")
st.sidebar.write("This is a testing app.")
