import os
import sys
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS  
from langchain.llms import HuggingFaceHub

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"] 

class Process:
    def __init__(self, file):
        self.filename = file
        pass

    def read_pdf(self):

        texts = ""
        pdf_reader = PdfReader(self.filename)
        for page in pdf_reader.pages:
            texts += page.extract_text()
        
        # Split the documents into chunks of 1000 characters with an overlap of 20 characters
        text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=20)
        texts = text_splitter.split_text(texts)

        return texts

    def get_vectorstore(self):
        
        texts_chunks = self.read_pdf()
        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.from_texts(texts=texts_chunks, embedding=embeddings)

        return vectorstore


    def setup_chain(self):
        
        repo_id = "google/flan-t5-xxl"

        # Memory for chat history
        memory = ConversationBufferMemory(memory_keys='chat_history', return_messages=True)

        # Load model from HuggingFace 
        # print(HUGGINGFACEHUB_API_TOKEN)
        llm = HuggingFaceHub(repo_id=repo_id, huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN, model_kwargs={"max_length": 512, "temperature": 0.8})

        # Retrieve the vectorstore
        vectordb = self.get_vectorstore()

        # Set up the Conversational Retrieval Chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectordb.as_retriever(search_kwargs={'k': 2}),
            return_source_documents=True
        )
        return qa_chain










