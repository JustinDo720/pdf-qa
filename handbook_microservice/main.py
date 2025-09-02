from flask import Flask, request 
from flask_restful import Api, Resource
import fitz
from pinecone import Pinecone
import os 
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from openai import OpenAI

# Initialize our app 
app = Flask(__name__)
api = Api(app)

# Adding some Pinecone configs before running our application 
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = 'company-handbook'

pc = Pinecone(api_key=PINECONE_API_KEY, environment='us-east-1')
index = pc.Index(INDEX_NAME)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class Upload(Resource):
    """
        This View will handle posting PDF to our Pinecone Vectorized Database for embeddings 
    """
    def post(self):
        try:
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
                index_name=INDEX_NAME)
            
            return {
                "message": "Document uplaoded successfully."
            }, 200
        except Exception as e:
            return {
                "message": "Upload to Pinecone failed.",
                "error": str(e)
            }, 500


class LLMResponse(Resource):
    """
        This POST method will take in a User question and have our resource query Pinecone + 
        return a response based on OpenAI LLM   
    """
    TOP_K = 5

    def post(self):
        try:
            response = request.get_json()
            question = response['question']

            # Converting user question into Embeddings 
            embedding_resp = client.embeddings.create(
                model='text-embedding-3-small',  # Must match the one on our Pinecone DB
                input=question
            )

            # Query Pinecone with this embedding 
            pinecone_result = index.query(
                vector=embedding_resp.data[0].embedding,
                top_k=LLMResponse.TOP_K,
                include_metadata=True
            )

            # Building contexts from our pinecone results 
            llm_context = [d['metadata']['text'] for d in pinecone_result['matches']]
            # Building a string format for out context  (Double line for easier reading)
            llm_context_str = "\n\n".join(llm_context)


            # Using ChatGPT OpenAI LLM to return a response 
            prompt = f"""
            You are an assistant with access to the following context from a document:

            {llm_context_str}

            Answer the question based only on the above context.
            Question: {question}
            """

            llm_response = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{'role':'user', 'content':prompt}]
            )

            return {
                'answer': llm_response.choices[0].message.content
            }, 200 
        except Exception as e:
            return {
                'error': str(e)
            }, 500

api.add_resource(Upload, '/uploads')
api.add_resource(LLMResponse, '/question')


if __name__ == '__main__':
    app.run()