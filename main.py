from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from vector import retriever  # Assuming vector.py contains the vector store setup code
model = OllamaLLM(model="llama3.2")

template = """
You are a highly knowledgeable and friendly baby care expert. Your goal is to provide clear, accurate, and empathetic advice to caregivers.

Based on the studies and articles I've found, here's some helpful information that may address your question:
{studies}

Now, let me help you further with your question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n------------------------")
    question = input("Enter your question (q to quit): ")
    print("\n\n")
    if question.lower() == "q":
        break
    
    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)
    
    # Display retrieved documents for testing purposes
    print("Retrieved Documents:")
    for doc in retrieved_docs:
        print(f"\n--- Document ID: {doc.metadata['source']} - Page: {doc.metadata.get('page', 'N/A')}")
        print(f"Content Snippet: {doc.page_content[:200]}...")  # Print a snippet of the content (first 200 characters)

    # Prepare studies for prompt
    studies = "\n\n".join([doc.page_content[:200] for doc in retrieved_docs])  # Include only snippets for testing
    
    # Generate answer from the model
    result = chain.invoke({"studies": studies, "question": question})
    
    # Display the result
    print("\nModel Response:")
    print(result)
