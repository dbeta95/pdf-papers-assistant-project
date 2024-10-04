"""
Module to initialize the application. This module is used to set up the
rag's hyperparameters in the GCP storage, Upload the documents to the GCP storage
and index the documents in the Elasticsearch index and initialize the postgress database.
"""
import os
import json

from dotenv import load_dotenv

from src.db import ElasticsearchManager, init_db
from src.storage import StorageManager
from src.data_ingestion import init_es_index

load_dotenv()

GCP_PROJECT_ID=os.getenv('GCP_PROJECT_ID')
BUCKET=os.getenv('BUCKET')
INDEX_NAME = os.getenv('INDEX_NAME')
ELASTICSEARCH_HOST = os.getenv('ELASTICSEARCH_HOST')
ELASTICSEARCH_PORT = os.getenv('ELASTICSEARCH_PORT')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
TEXT_FIELDS = os.getenv('TEXT_FIELDS').split(',')
KEYWORD_FIELDS = os.getenv('KEYWORD_FIELDS').split(',')

if __name__=="__main__":
    
    # Get the project path
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Initialize the storage manager and upload the rag_config.json file
    storage_manager = StorageManager(gcp_project=GCP_PROJECT_ID, bucket_name=BUCKET)
    config_file = os.path.join(project_path, "src", "parameters", "rag_config.json")
    storage_manager.upload_file(config_file, "rag_config.json")
    
    # Initialize the Elasticsearch manager
    es_manager = ElasticsearchManager(
        host=ELASTICSEARCH_HOST,
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
    