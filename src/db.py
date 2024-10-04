"""
Module with various utilities to manage the databases that
interact with the app.
"""
import os
import psycopg2


from psycopg2.extras import DictCursor
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from typing import List
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from tqdm.auto import tqdm

load_dotenv()


RUN_TIMEZONE_CHECK = os.getenv('RUN_TIMEZONE_CHECK', '1') == '1'
TZ_INFO = os.getenv('TZ')
POSTGRES_HOST = 'localhost'
POSTGRES_DB = os.getenv('POSTGRES_DB')
POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')

tz = ZoneInfo(TZ_INFO)

def create_schema_dict(
    text_fields:List[str], 
    keyword_fields:List[str],
    embbedings_dim:int 
):
    """
    Creates a dictionary representing an Elasticsearch schema with properties 
    for text and keyword fields, along with dense vectors for text fields for 
    similarity search.

    Args:
        text_fields: A list of strings representing the names of text fields 
            in the schema.
        keyword_fields: A list of strings representing the names of keyword 
            fields in the schema.

    Returns:
        A dictionary representing the Elasticsearch schema, including:
            - properties: A dictionary containing the field definitions, where
              a dense vector is created joining the text fields
    """
    properties = {}
    
    # Add text fields
    for field in text_fields:
        properties[field] = {"type": "text"}        
    
    # Add keyword fields
    for field in keyword_fields:
        properties[field] = {"type": "keyword"}
        
    properties["full_text_vector"] = {
        "type": "dense_vector",
        "dims": embbedings_dim,
        "index": True,
        "similarity": "cosine"
    }

    return {
        "properties": properties
    }
    
def create_search_queries(
    query:str,
    field_names:List[str],
    vector,
    n_results:int,
    alpha:float,        
    field_weights:dict,    
    filter_dict:dict=None,
):
    """
    Creates two Elasticsearch search query dictionaries: one for keyword-based search 
    and another for k-nearest neighbors (KNN) search.

    Args:
        query (str): The search query string.
        alpha (float): A float value representing the weight assigned to the KNN search. 
        (1 - alpha) is assigned to the keyword search.        
        n_rsults (int): Number of results to be retrieved by the vector search.
        vector: An array representing the query vector for KNN search.
        filter_dict Optional(dict): A dictionary containing filter conditions for the search.

    Returns:
        A tuple containing two dictionaries:
        - keyword: The keyword-based search query dictionary.
        - knn: The KNN search query dictionary.
    """
    # Construct the "fields" list with weights
    fields_with_weights = [f"{field}^{field_weights[field]}" for field in field_names]

    keyword = {
        "bool": {
            "must": {
                "multi_match": {
                    "query": query,
                    "fields": fields_with_weights,
                    "type": "best_fields",
                    "boost": 1 - alpha,
                }
            }
        }
    }

    knn = {
        "field": "full_text_vector",
        "query_vector": vector,
        "k": n_results,
        "num_candidates": 10000,
        "boost": alpha
    }
    # Add filter if provided and not empty
    if filter_dict and filter_dict.keys():
        keyword["bool"]["filter"] = {"term": filter_dict}
        knn["filter"] = {"term": filter_dict}

    return keyword, knn

class ElasticsearchManager():
    
    def __init__(self, 
        host:str, 
        port:str,
        embedding_model_name:str,
    ) -> None:
        """
        Initializes the class with the given parameters.

        Args:
            host (str): The hostname of the Elasticsearch server.
            port (str): The port number of the Elasticsearch server.
            embedding_model_name (str): The name of the embedding model to be used.
            

        Attributes:
            es_client (Elasticsearch): An instance of the Elasticsearch client connected to the specified host and port.
            embbeding_model (SentenceTransformer): An instance of the SentenceTransformer model for generating embeddings.
            embbeding_dim (int): The dimension of the sentence embeddings generated by the embedding model.
        """
        self.es_client = Elasticsearch(f"http://{host}:{port}")
        self.embbeding_model = SentenceTransformer(embedding_model_name)
        self.embbeding_dim = self.embbeding_model.get_sentence_embedding_dimension()
            
    def create_index(self,
        index_name:str,
        text_fields:List[str],
        keyword_fields:List[str],
        shards:int=1,
        replicas:int=0
    ):
        """
        Creates an Elasticsearch index with the specified settings and mappings.

        Args:
            index_name (str): The name of the index to be created.
            text_fields (List[str]): A list of field names to be indexed as text fields.
            keyword_fields (List[str]): A list of field names to be indexed as keyword fields.
            shards (int, optional): The number of primary shards for the index. Defaults to 1.
            replicas (int, optional): The number of replica shards for the index. Defaults to 0.
        """ 
        self.text_fields = text_fields
        mappings_schema = create_schema_dict(
            text_fields=text_fields,
            keyword_fields=keyword_fields,
            embbedings_dim=self.embbeding_dim
        )
        
        index_settings = {
            "settings":{
                "number_of_shards":shards,
                "number_of_replicas":replicas
            },
            "mappings": mappings_schema
        }
        
        self.es_client.indices.delete(
            index=index_name, ignore_unavailable=True
        )
        self.es_client.indices.create(
            index=index_name, body=index_settings
        )
        print(f"Index {index_name} created.")
        
    def index_documents(self,
      docs: List[dict],
      index_name: str  
    ) -> None:
        """
        Indexes a list of documents into the specified Elasticsearch index.

        Args:
            docs (List[dict]): A list of documents to be indexed. Each document is represented as a dictionary.
            index_name (str): The name of the Elasticsearch index where the documents will be indexed.
        """
        for doc in tqdm(docs):
            full_text = " ".join([value for key, value in doc.items() if key in self.text_fields])
            doc["full_text_vector"] = self.embbeding_model.encode(full_text)
            self.es_client.index(index=index_name, document=doc)
            
    def hybrid_search(self,
        index_name:str,
        query:str,
        field_names:List[str],
        vector,
        n_results:int=10,
        alpha:float=0.5,
        filter_dict:dict=None,
        field_weights:dict=None
    ):
        if not field_weights:
            field_weights = {field:1 for field in field_names}
            
        fields_to_return = ["id"] + field_names

        keyword, knn = create_search_queries(
            query=query,
            field_names=field_names,
            vector=vector,
            n_results=n_results,
            alpha=alpha,
            filter_dict=filter_dict,
            field_weights=field_weights
        ) 
        
        search_query = {
            "knn": knn,
            "query": keyword,
            "size": n_results,
            "_source": fields_to_return
        }
        
        es_results = self.es_client.search(
            index=index_name,
            body=search_query
        )
        
        return [hit['_source'] for hit in es_results['hits']['hits']]
    
def get_db_connection():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD
    )
    
def init_db():
    conn = get_db_connection()
    
    try:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS feedback")
            cur.execute("DROP TABLE IF EXISTS conversations")
            
            cur.execute("""
                CREATE TABLE conversations (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    response_time FLOAT NOT NULL,
                    relevance TEXT NOT NULL,
                    relevance_explanation TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    eval_prompt_tokens INTEGER NOT NULL,
                    eval_completion_tokens INTEGER NOT NULL,
                    eval_total_tokens INTEGER NOT NULL,
                    google_cost FLOAT NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )            
            """)
            cur.execute("""
                CREATE TABLE feedback (
                    id SERIAL PRIMARY KEY,
                    conversation_id TEXT REFERENCES conversations(id),
                    feedback INTEGER NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)
            conn.commit()
    finally:
        conn.close()
        
def save_conversation(conversation_id, question, answer_data, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now(tz)

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations 
                (id, question, answer, model_used, response_time, relevance, 
                relevance_explanation, prompt_tokens, completion_tokens, total_tokens, 
                eval_prompt_tokens, eval_completion_tokens, eval_total_tokens, google_cost, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    conversation_id,
                    question,
                    answer_data["answer"],
                    answer_data["model_used"],
                    answer_data["response_time"],
                    answer_data["relevance"],
                    answer_data["relevance_explanation"],
                    answer_data["prompt_tokens"],
                    answer_data["completion_tokens"],
                    answer_data["total_tokens"],
                    answer_data["eval_prompt_tokens"],
                    answer_data["eval_completion_tokens"],
                    answer_data["eval_total_tokens"],
                    answer_data["google_cost"],
                    timestamp
                ),
            )
        conn.commit()
    finally:
        conn.close()
        
def save_feedback(conversation_id, feedback, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now(tz)

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (conversation_id, feedback, timestamp) VALUES (%s, %s, COALESCE(%s, CURRENT_TIMESTAMP))",
                (conversation_id, feedback, timestamp),
            )
        conn.commit()
    finally:
        conn.close()

def get_recent_conversations(limit=5, relevance=None):
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            query = """
                SELECT c.*, f.feedback
                FROM conversations c
                LEFT JOIN feedback f ON c.id = f.conversation_id
            """
            if relevance:
                query += f" WHERE c.relevance = '{relevance}'"
            query += " ORDER BY c.timestamp DESC LIMIT %s"

            cur.execute(query, (limit,))
            return cur.fetchall()
    finally:
        conn.close()

def get_feedback_stats():
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("""
                SELECT 
                    SUM(CASE WHEN feedback > 0 THEN 1 ELSE 0 END) as thumbs_up,
                    SUM(CASE WHEN feedback < 0 THEN 1 ELSE 0 END) as thumbs_down
                FROM feedback
            """)
            return cur.fetchone()
    finally:
        conn.close()


def check_timezone():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW timezone;")
            db_timezone = cur.fetchone()[0]
            print(f"Database timezone: {db_timezone}")

            cur.execute("SELECT current_timestamp;")
            db_time_utc = cur.fetchone()[0]
            print(f"Database current time (UTC): {db_time_utc}")

            db_time_local = db_time_utc.astimezone(tz)
            print(f"Database current time ({TZ_INFO}): {db_time_local}")

            py_time = datetime.now(tz)
            print(f"Python current time: {py_time}")

            # Use py_time instead of tz for insertion
            cur.execute("""
                INSERT INTO conversations 
                (id, question, answer, model_used, response_time, relevance, 
                relevance_explanation, prompt_tokens, completion_tokens, total_tokens, 
                eval_prompt_tokens, eval_completion_tokens, eval_total_tokens, openai_cost, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING timestamp;
            """, 
            ('test', 'test question', 'test answer', 'test model', 0.0, 0.0, 
             'test explanation', 0, 0, 0, 0, 0, 0, 0.0, py_time))

            inserted_time = cur.fetchone()[0]
            print(f"Inserted time (UTC): {inserted_time}")
            print(f"Inserted time ({TZ_INFO}): {inserted_time.astimezone(tz)}")

            cur.execute("SELECT timestamp FROM conversations WHERE id = 'test';")
            selected_time = cur.fetchone()[0]
            print(f"Selected time (UTC): {selected_time}")
            print(f"Selected time ({TZ_INFO}): {selected_time.astimezone(tz)}")

            # Clean up the test entry
            cur.execute("DELETE FROM conversations WHERE id = 'test';")
            conn.commit()
    except Exception as e:
        print(f"An error occurred: {e}")
        conn.rollback()
    finally:
        conn.close()
