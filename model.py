import os
import sys
import pinecone
from langchain.llms import Replicate
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain

# Replicate API token
os.environ['REPLICATE_API_TOKEN'] = "r8_BknXiKJbqkVFFySpLTISoSWjPyp0IOo4QpDkU"

# Initialize Pinecone
pinecone.init(api_key='41d95e1b-b768-49ac-aab4-3b79510e2273', environment='gcp-starter')

class Process:
    def __init__(self, file):
        self.filename = file
        pass

    def read_pdf(self):
        loader = PyPDFLoader(self.filename)
        documents = loader.load() 

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        return texts


    def setup_chain(self):
        texts = self.read_pdf()

        embeddings = HuggingFaceEmbeddings()

        # Set up the Pinecone vector database
        index_name = "pdf-llama"

        index = pinecone.Index(index_name)
        vectordb = Pinecone.from_documents(texts, embeddings, index_name=index_name)


        llama2 = Replicate(model= "meta/llama-2-13b-chat:56acad22679f6b95d6e45c78309a2b50a670d5ed29a37dd73d182e89772c02f1",
                            model_kwargs={"temperature": 0.75, "max_length": 2500})
        
        # Set up the Conversational Retrieval Chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llama2,
            vectordb.as_retriever(search_kwargs={'k': 2}),
            return_source_documents=True
        )
        return qa_chain










