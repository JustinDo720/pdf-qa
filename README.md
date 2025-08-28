# Company Handbook Q&A with RAG Workflow

## Objective

Our primary objective is to leverage Retrieval-Augmented Generation (RAG) workflows to create a robust Question & Answer (Q&A) system based on company project documents. This approach enables users to query project documentation and receive accurate, context-aware answers powered by advanced language models.

## Tech Stack

- **LLM (Large Language Model):** For natural language understanding and generation.
- **LangChain:** Framework for building applications with LLMs and integrating external data sources.
- **RAD:** Rapid Application Development tools to accelerate prototyping and deployment.
- **Pinecone:** Vector database for efficient document retrieval and semantic search.
- **Flask** Microservice testing API Endpoint with Postman 

## Workflow Overview

1. **Document Ingestion:** Company project documents (e.g., PDFs) are processed and embedded.
2. **Indexing:** Embeddings are stored in Pinecone for fast similarity search.
3. **Query Handling:** User questions are interpreted by the LLM via LangChain.
4. **Retrieval:** Relevant document sections are fetched from Pinecone.
5. **Answer Generation:** The LLM generates answers using both retrieved context and its own knowledge.

## Getting Started

1. Place your company project documents in the designated folder.
2. Run the ingestion script to index documents in Pinecone.
3. Use the Q&A interface to ask questions about your projects.

## Benefits

- Accurate, context-aware answers
- Scalable to multiple documents and projects
- Easy integration

## Development Phase 

### 08/28

**Getting Started**
`pip install flask; flask-restful`

We create our own basic resource:

```py
class Upload(Resource):
    """
        This View will handle posting PDF to our Pinecone Vectorized Database for embeddings 
    """
    def post(self):
        pdf_file = request.files['file']

        # Need a package to convert pdf_file into a readable python object
        pass
```

But we need a package to handle our PDF 
- `pip install PyMuPDF`

```py
# Updating our post method to read the pdf files 
from flask import request 
import fitz

def post(self):
    file = request.files['file']

    # Opening our pdf file with PyMuPDF
    file_content = fitz.open(stream=file.read(), filetype='pdf')

    # Looping through the pages and reading the text 
    text = ""
    for page in file_content:
        text += page.get_text()
```

**Langchain + Pinecone Config**

`pip install pinecone langchain langchain-pinecone langchain-openai`

**Building our requirements for Pinecone**
```py
# Initializing Pinecone + Index 
from pinecone import Pinecone 
from langchain_openai import OpenAIEmbeddings

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# Create index on pinecone website
INDEX_NAME = 'INDEX-NAME'

pc = Pinecone(api_key=PINECONE_API_KEY, environment='us-east-1')
# Make sure the embedding is the same model as your index 
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

```

---

**Langchain Text Splitter and Prepping text for Vector Store**
```py
# Vectorizing our text for our vectorstore
#  
# In your post method....
from langchain.text_splitter import RecursiveCharacterTextSplitter 

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
pdf_splitter = splitter.create_documents([text])    # Remember your text object is from our PDF reader => fitz
```

---
**Ingesting splitter text to Pinecone**

```py
from langchain_pinecone import PineconeVectorStore

vectorstore = PineconeVectorStore.from_documents(
    documents=pdf_splitter,
    # These two were already defined outside of our post method 
    embedding=embeddings,
    # Vector Store expects an Index Object not a string
    index=pc.Index(INDEX_NAME)
)
```