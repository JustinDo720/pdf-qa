from flask import Flask, request 
from flask_restful import Api, Resource
import fitz
from pinecone import Pinecone
import os 
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

# Initialize our app 
app = Flask(__name__)
api = Api(app)

# Adding some Pinecone configs before running our application 
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = 'company-handbook'

pc = Pinecone(api_key=PINECONE_API_KEY, environment='us-east-1')
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


class Upload(Resource):
    """
        This View will handle posting PDF to our Pinecone Vectorized Database for embeddings 
    """
    def post(self):
        pdf_file = request.files['file']

        # Need a package to convert pdf_file into a readable python object
        pdf_read = fitz.open(stream=pdf_file.read(), filetype='pdf')    # We're using PyMuPDF to open and read our pdf_file

        # Text object
        text = ''
        for page in pdf_read:
            text += page.get_text()

        # Prepping text for our vector store
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        pdf_splitter = splitter.create_documents([text])
        
        # Store in Pinecone 
        vectorstore = PineconeVectorStore.from_documents(
            documents=pdf_splitter, 
            embedding=embeddings, 
            index_name=pc.Index(INDEX_NAME))


api.add_resource(Upload, '/uploads')


if __name__ == '__main__':
    app.run()