"""
hybsearch.py - Hybrid search with TF-IDF, embeddings and cosine similarity for text fields
and exact matching for keyword fields.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

def compute_rrf(rank:int, k:int=60) -> float:
    """
    Computes the Reciprocal Rank Fusion (RRF) score for a given rank.

    RRF is a method used in information retrieval to combine the results of multiple ranked lists.

    Args:
        rank (int): The rank of the item.
        k (int, optional): A constant used in the RRF formula. Defaults to 60.

    Returns:
        float: The RRF score for the given rank.
    """
    return 1 / (k + rank)

class Index:
    """
    A simple search index hybrid search with TF-IDF, embeddings and cosine similarity for text fields and 
    exact matching for keyword fields.

    Attributes:
        text_fields (list): List of text field names to index.
        keyword_fields (list): List of keyword field names to index.
        vectorizers (dict): Dictionary of TfidfVectorizer instances for each text field.
        keyword_df (pd.DataFrame): DataFrame containing keyword field data.
        text_matrices (dict): Dictionary of TF-IDF matrices for each text field.
        docs (list): List of documents indexed.
    """

    def __init__(self, 
            text_fields:List[str], 
            keyword_fields:List[str],
            vectorizer_params:dict={}, 
            embedding_model_name:str='all-mpnet-base-v2'
        ):
        """
        Initializes the Index with specified text and keyword fields.

        Args:
            text_fields (list): List of text field names to index.
            keyword_fields (list): List of keyword field names to index.
            vectorizer_params (dict): Optional parameters to pass to TfidfVectorizer.
            embedding_model_name (str): Name of the SentenceTransformer model to use for text embeddings.
        """
        self.text_fields = text_fields
        self.keyword_fields = keyword_fields

        self.vectorizers = {field: TfidfVectorizer(**vectorizer_params) for field in text_fields}
        self.keyword_df = None
        self.text_matrices = {}
        self.embeddings = {}
        self.docs = []
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def fit(self, docs):
        """
        Fits the index with the provided documents.

        Args:
            docs (list of dict): List of documents to index. Each document is a dictionary.
        """
        self.docs = docs
        keyword_data = {field: [] for field in self.keyword_fields}

        for field in self.text_fields:
            texts = [doc.get(field, '') for doc in docs]
            self.text_matrices[field] = self.vectorizers[field].fit_transform(texts)        

        # Generate sentence embeddings for text fields
        for field in self.text_fields:
            texts = [doc.get(field, '') for doc in docs]
            embeddings = self.embedding_model.encode(texts)
            self.embeddings[field] = embeddings  # Store embeddings directly
            
        for doc in docs:
            for field in self.keyword_fields:
                keyword_data[field].append(doc.get(field, ''))

        self.keyword_df = pd.DataFrame(keyword_data)

        return self

    def search(self, 
            query:str, 
            filter_dict:dict={}, 
            boost_dict:dict={}, 
            alpha:float=0.5, 
            num_results:int=10,
            rrf:bool=False,
            k:int=60
        ):
        """
        Searches the index with the given query, filters, and boost parameters.

        Args:
            query (str): The search query string.
            filter_dict (dict): Dictionary of keyword fields to filter by. Keys are field names and values are the values to filter by.
            boost_dict (dict): Dictionary of boost scores for text fields. Keys are field names and values are the boost scores.
            num_results (int): The number of top results to return. Defaults to 10.
            rrf (bool): Whether to apply Reciprocal Rank Fusion to the search results. Defaults to False.
            k (int): Constat for the RRF formula
        Returns:
            list of dict: List of documents matching the search criteria, ranked by relevance.
        """
        query_vecs = {field: self.vectorizers[field].transform([query]) for field in self.text_fields}
        match_scores = np.zeros(len(self.docs))
        vec_scores = np.zeros(len(self.docs))

        # Compute cosine similarity for each text field and apply boost
        for field, query_vec in query_vecs.items():
            sim = cosine_similarity(query_vec, self.text_matrices[field]).flatten()
            boost = boost_dict.get(field, 1)
            match_scores += sim * boost

        # Generate sentence embedding for the query
        query_embedding = self.embedding_model.encode([query])[0] 

        # Compute cosine similarity for each text field and apply boost
        for field in self.text_fields:
            sim = cosine_similarity(np.array(query_embedding).reshape(1, -1), self.embeddings[field])[0]
            boost = boost_dict.get(field, 1)
            vec_scores += sim * boost            
        
        
        if rrf:
            # Apply Reciprocal Rank Fusion
            
            # Use argpartition to get top num_results indices for each score            
            # match_scores
            top_match_indices = np.argpartition(match_scores, -num_results)[-num_results:]
            top_match_indices = top_match_indices[np.argsort(-match_scores[top_match_indices])]

            # Filter out zero-score results
            filtered_match_indices = [i for i in top_match_indices if match_scores[i] > 0]
            
            # vector_scores
            top_vec_indices = np.argpartition(vec_scores, -num_results)[-num_results:]
            top_vec_indices = top_vec_indices[np.argsort(-vec_scores[top_vec_indices])]

            # Filter out zero-score results
            filtered_vec_indices = [i for i in top_vec_indices if vec_scores[i] > 0]
                        
            rrf_scores = {}
            # Compute RRF scores for match and vector scores
            for rank, index in enumerate(filtered_match_indices):                
                rrf_scores[index] = compute_rrf(rank + 1, k)
                
            for rank, index in enumerate(filtered_vec_indices):
                if index in rrf_scores:
                    rrf_scores[index] += compute_rrf(rank + 1, k)
                else:
                    rrf_scores[index] = compute_rrf(rank + 1, k)
                    
            # Sort RRF scores indescending order
            reranked_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
                    
            # Get top num_results indices
            top_indices = reranked_indices[:num_results]
            
            return [self.docs[i] for i in top_indices]
            
        else:
            scores = (1-alpha)*match_scores + alpha*vec_scores
            
            # Apply keyword filters
            for field, value in filter_dict.items():
                if field in self.keyword_fields:
                    mask = self.keyword_df[field] == value
                    scores = scores * mask.to_numpy()
            
            # Use argpartition to get top num_results indices
            top_indices = np.argpartition(scores, -num_results)[-num_results:]
            top_indices = top_indices[np.argsort(-scores[top_indices])]

            # Filter out zero-score results
            top_docs = [self.docs[i] for i in top_indices if scores[i] > 0]

            return top_docs