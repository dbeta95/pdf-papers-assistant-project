"""
Module to initialize the application. This module is used to set up the
rag's hyperparameters in the GCP storage, Upload the documents to the GCP storage
and index the documents in the Elasticsearch index and initialize the postgress database.
"""
import os
import sys
import json
import tempfile

from dotenv import load_dotenv

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

from src.db import ElasticsearchManager, init_db
from src.storage import StorageManager
from src.data_ingestion import init_es_index
from src.rag import RAG

load_dotenv(override=True)

GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
GCP_PROJECT_ID=os.getenv('GCP_PROJECT_ID')
BUCKET=os.getenv('BUCKET')
INDEX_NAME = os.getenv('INDEX_NAME')
ELASTICSEARCH_HOST = os.getenv('ELASTICSEARCH_HOST')
ELASTICSEARCH_PORT = os.getenv('ELASTICSEARCH_PORT')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
TEXT_FIELDS = os.getenv('TEXT_FIELDS').split(',')
KEYWORD_FIELDS = os.getenv('KEYWORD_FIELDS').split(',')

# Get the project path
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

entry_template = """
category: {category}
paper: {paper}
text: {text}
""".strip()

prompt_template = """
You are a research assistant specializing in various academic fields. 
Your task is to provide accurate and concise answers to questions based on the information extracted from the provided research papers.

**Question:** {question}

**Context:**

{context}

**Guidelines:**

* **Cite your sources:** If you reference specific information from a paper, include the paper title in parentheses, e.g., "(Attention is all You need)".
* **Prioritize relevance:** Only use information from the context that is directly relevant to the question.
* **Be concise:** Provide clear and focused answers without unnecessary elaboration.
* **Maintain academic tone:** Use language appropriate for an academic audience.
* **If the context doesn't contain enough information to fully answer the question, clearly state that you need more information or that the context doesn't address the question.**

**Answer:**
"""

def create_rag():
    """
    Function to create a RAG object
    """
    # Initialize the storage manager and download the rag_config.json file
    storage_manager = StorageManager(gcp_project=GCP_PROJECT_ID, bucket_name=BUCKET)
    with tempfile.TemporaryDirectory() as tmp_dir:
        rag_config_path = os.path.join(tmp_dir, 'rag_config.json')
        storage_manager.download_file(rag_config_path, 'rag_config.json')
        rag_config = json.load(open(rag_config_path))
    
    # Initialize the RAG object
    rag = RAG(api_key=GOOGLE_API_KEY)
    # Update the RAG parameters
    rag.update_parameters(**rag_config)
    # Set the Elasticsearch manager
    rag.get_es_manager(
        index_name=INDEX_NAME,
        text_fields=TEXT_FIELDS,
        elasticsearch_host=ELASTICSEARCH_HOST,
        elasticsearch_port=ELASTICSEARCH_PORT,
        embedding_model_name=EMBEDDING_MODEL
    )
    # Set the prompt templates
    rag.set_prompt_templates(
        entry_template=entry_template,
        prompt_template=prompt_template
    )
    return rag

if __name__=="__main__": 
    
    # Initialize the storage manager and upload the rag_config.json file
    storage_manager = StorageManager(gcp_project=GCP_PROJECT_ID, bucket_name=BUCKET)
    config_file = os.path.join(project_path, "src", "parameters", "rag_config.json")
    storage_manager.upload_file(config_file, "rag_config.json")
    
    # Initialize the Elasticsearch manager
    es_manager = ElasticsearchManager(
        host="localhost",
        port=ELASTICSEARCH_PORT,
        embedding_model_name=EMBEDDING_MODEL
    )
    
    # Create the Elasticsearch index and load the documents
    init_es_index(
        storage_manager=storage_manager,
        elasticsearch_manager=es_manager,
        docs_path=os.path.join(project_path, "docs"),
        index_name=INDEX_NAME,
        text_fields=TEXT_FIELDS,
        keyword_fields=KEYWORD_FIELDS
    )
    
    print("Initializig the database...")
    init_db()
    print("Database initialized successfully.")
    