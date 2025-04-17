import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader
import os

# Set your HuggingFace Hub token (replace with your token or use env variable)
#os.environ["API Key"] = "Your Key"

# App title
st.title("📄 Q&A from your PDF using LLM")

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    # Read and extract text from PDF
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        
    st.success("PDF Loaded. Processing...")
    
    
    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_text(text)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docsearch = FAISS.from_texts(texts, embeddings)
    
    # Load LLM (can use gpt2 or another small model for demo)
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",  # You can try other models too
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )
    
    chain = load_qa_chain(llm, chain_type="stuff")
    
    # Ask a question
    question = st.text_input("Ask a question from the PDF:")
    if question:
        with st.spinner("Thinking..."):
            docs = docsearch.similarity_search(question)
            answer = chain.run(input_documents=docs, question=question)
            st.markdown(f"**Answer:** {answer}")



        