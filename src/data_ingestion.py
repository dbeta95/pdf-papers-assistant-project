"""
Module defining the class and functions for data ingestion
both for the initial data loading and for the incremental
data loading.
"""
import os
import sys
import hashlib
import tempfile

from tqdm import tqdm

project_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_path)

from src.preprocess import  extract_text_from_pdf, get_sequential_semantic_chunks
from src.storage import StorageManager
from src.db import ElasticsearchManager

def upload_documents(docs_path:str, storage_manager:StorageManager) -> None:
    """Uploads documents to GCP bucket.

    Args:
        docs_path (str): Path to documents.
        storage_manager (StorageManager): Storage manager object.
    """
    doc_categories = os.listdir(docs_path)
    for category in doc_categories:
        category_path = os.path.join(docs_path, category)
        storage_manager.upload_dir(category_path, f'docs/{category}')
        
def index_document_from_blob(
    blob, 
    doc_index:int, 
    elasticsearch_manager:ElasticsearchManager,
    index_name:str
) -> None:
    """Indexes a document from a GCP bucket blob.

    Args:
        blob (Blob): GCP bucket blob.
        doc_id (str): Document ID.
        elasticsearch_manager (ElasticsearchManager): Elasticsearch manager object.
    """
    category, paper = blob.name.split('/')[1:]
    
    doc_id = hashlib.sha256(
        f'{category}-{paper}-{doc_index}'.encode('utf-8')
    ).hexdigest()
    
    pdf_path = os.path.join(tempfile.gettempdir(), 'paper.pdf')
    blob.download_to_filename(pdf_path)
    pdf_text = extract_text_from_pdf(pdf_path)
    doc_chunks = get_sequential_semantic_chunks(pdf_text,doc_id)
    
    docs = []
    
    for doc_chunk in doc_chunks:                
        docs.append({
            'id': f'{doc_id}-{doc_chunk['chunk']}',
            'category':category,
            'paper': paper,
            'text': doc_chunk['text']
        })
        
    elasticsearch_manager.index_documents(
        docs=docs,
        index_name=index_name
    )
    
def index_documents_from_bucket(
    storage_manager:StorageManager,
    elasticsearch_manager:ElasticsearchManager,
    index_name:str
) -> None:
    """Indexes documents from a GCP bucket.

    Args:
        storage_manager (StorageManager): Storage manager object.
        elasticsearch_manager (ElasticsearchManager): Elasticsearch manager object.
        index_name (str): Elasticsearch index name.
    """
    blobs = storage_manager.bucket.list_blobs(prefix='docs/')
    for index, blob in tqdm(enumerate(blobs)):
        index_document_from_blob(
            blob=blob,
            doc_index=index+1,
            elasticsearch_manager=elasticsearch_manager,
            index_name=index_name
        )
        
def init_es_index(
    storage_manager:StorageManager,
    elasticsearch_manager:ElasticsearchManager,
    docs_path:str,
    index_name:str,
    text_fields:list,
    keyword_fields:list
) -> None:
    """
    Initializes the database by uploading the documents to the GCP bucket
    and indexing the documents in Elasticsearch.
    
    Args:
        storage_manager (StorageManager): Storage manager object.
        elasticsearch_manager (ElasticsearchManager): Elasticsearch manager object.
        docs_path (str): Path to the documents.
        index_name (str): Name of the Elasticsearch index.
        text_fields (list): List of text fields.
        keyword_fields (list): List of keyword fields.  
    """
    print("Uploading documents to GCP bucket...")
    upload_documents(docs_path, storage_manager)
    print("Creating Elasticsearch index...")
    elasticsearch_manager.create_index(
        index_name=index_name,
        text_fields=text_fields,
        keyword_fields=keyword_fields   
    )
    print("Indexing documents in Elasticsearch...")
    index_documents_from_bucket(
        storage_manager=storage_manager, 
        elasticsearch_manager=elasticsearch_manager,
        index_name=index_name
    )
    print("Data ingestion completed.")

