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
from dotenv import load_dotenv

project_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_path)

from src.preprocess import  extract_text_from_pdf, get_sequential_semantic_chunks
from src.storage import StorageManager
from src.db import ElasticsearchManager

load_dotenv(override=True)

GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID')
BUCKET = os.getenv('BUCKET')
INDEX_NAME = os.getenv('INDEX_NAME')
ELASTICSEARCH_HOST = os.getenv('ELASTICSEARCH_HOST')
ELASTICSEARCH_PORT = os.getenv('ELASTICSEARCH_PORT')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
TEXT_FIELDS = os.getenv('TEXT_FIELDS').split(',')
KEYWORD_FIELDS = os.getenv('KEYWORD_FIELDS').split(',')

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
        
def main():
    storage_manager = StorageManager(project_id=GCP_PROJECT_ID, bucket_name=BUCKET)
    elasticsearch_manager = ElasticsearchManager(
        host=ELASTICSEARCH_HOST,
        port=ELASTICSEARCH_PORT,
        embedding_model=EMBEDDING_MODEL,
        text_fields=TEXT_FIELDS,
        keyword_fields=KEYWORD_FIELDS
    )
    docs_path = os.path.join(project_path, 'docs')
    upload_documents(docs_path, storage_manager)
    index_documents_from_bucket(storage_manager, elasticsearch_manager, INDEX_NAME)
    
if __name__ == '__main__':
    main()
