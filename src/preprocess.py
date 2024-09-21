"""Module containing the definition of funcitons that process
the documents and converts them into dictionaries of
chunks
"""
import PyPDF2
import numpy as np

from nltk import sent_tokenize
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Union

load_dotenv()

def extract_text_from_pdf(pdf_path:str) -> str:
    """Function to extract the text from a given PDF

    Args:
        pdf_path (str): PDF file to be processed

    Returns:
        str: Document's text
    """
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
def get_sentences(text:str, doc_id:str) -> List[dict]:
    """Function that takes a text an chunks it into
    sentences, then structures it as a list of dictionaries

    Args:
        text (str): Text to be splitted
        doc_id (str): sha256 codified ID for each document

    Returns:
        List[dict]: List of dictionaries containing the
        document id, chunk_number and sentence each
    """
    sentence_dicts = []
    sentences = sent_tokenize(text) 
    

    for n, sentence in enumerate(sentences):
        sentence_dicts.append({
            "doc_id": doc_id,
            "chunk":n + 1,
            "text":sentence            
        })
        
    return sentence_dicts

def get_semantic_chunks(text:str, doc_id:str) -> List[dict]:
    """Function that takes a text and splitts it into chunks
    grouping them by sematic proximiy (using kmeans)

    Args:
        text (str): Text to be splitted
        doc_id (str): sha256 codified ID for each document

    Returns:
        List[dict]: List of dictionaries containing the
        document id, chunk_number and text each
    """
    
    sentences = sent_tokenize(text)

    # Create a sentence embedding model
    model = SentenceTransformer('all-mpnet-base-v2')

    # Convert sentences to vectors
    embeddings = model.encode(sentences)
    
    
    silhouette_scores = []
    for i in range(2, 100):  # Start from 2 clusters
        try:
            kmeans = KMeans(n_clusters=i, random_state=0)
            kmeans.fit(embeddings)
            score = silhouette_score(embeddings, kmeans.labels_)
            silhouette_scores.append(score)
        except:
            break
        
    clusters = np.argmax(silhouette_scores) + 2
    
    kmeans = KMeans(n_clusters=clusters, random_state=0)
    kmeans.fit(embeddings)

    # Create chunks based on cluster assignments
    chunks = [[] for _ in range(clusters)]
    for i, cluster_id in enumerate(kmeans.labels_):
        chunks[cluster_id].append(sentences[i])

    chunks_dicts = []
    for i, chunk in enumerate(chunks):
        chunks_dicts.append({
            "doc_id": doc_id,
            "chunk": i + 1,
            "text": "\n".join(chunk)
        })
        
    return chunks_dicts


def combine_sentences(sentences:List[dict], buffer_size:int=1) -> List[dict]:
    """Function that combines sentences according to a buffer size to give
    stability to the distance comparison between chunks

    Args:
        sentences (List[dict]): List of dictionaries where each one contains
            a sentece an it's index
        buffer_size (int, optional): _description_. Number of sentences to combine
            from before and after the present one

    Returns:
        List[dict]: List of dictionaries where each one contains
            a sentece, it's index and the combined text
    """
    # Create a sentence embedding model
    model = SentenceTransformer('all-mpnet-base-v2')
    # Go through each sentence dict
    for i in range(len(sentences)):

        # Create a string that will hold the sentences which are joined
        combined_sentence = ''

        # Add sentences before the current one, based on the buffer size.
        for j in range(i - buffer_size, i):
            # Check if the index j is not negative (to avoid index out of range like on the first one)
            if j >= 0:
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += sentences[j]['sentence'] + ' '

        # Add the current sentence
        combined_sentence += sentences[i]['sentence']

        # Add sentences after the current one, based on the buffer size
        for j in range(i + 1, i + 1 + buffer_size):
            # Check if the index j is within the range of the sentences list
            if j < len(sentences):
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += ' ' + sentences[j]['sentence']

        # Then add the whole thing to your dict
        # Store the combined sentence in the current sentence dict
        sentences[i]['combined_sentence'] = combined_sentence
        sentences[i]['combined_sentence_embedding'] = model.encode(combined_sentence)

    return sentences

def calculate_cosine_similarities(sentences:List[dict]) -> Union[List, List[dict]]:
    """_summary_

    Args:
        sentences (List[dict]): List of dictionaries where each one contains
            a sentece, it's index and the combined text

    Returns:
        Union[List, List[dict]]: A list of all similarities and a list of 
        dictionaries where each one contains a sentece, it's index, the combined text
        and embedding and the embedding's similarity with the next.
    """
    similarities = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]['combined_sentence_embedding']
        embedding_next = sentences[i + 1]['combined_sentence_embedding']
        
        # Calculate cosine similarity
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]

        # Append cosine distance to the list
        similarities.append(similarity)

        # Store distance in the dictionary
        sentences[i]['similarity_with_next'] = similarity

    # Optionally handle the last sentence
    sentences[-1]['similarity_with_next'] = None  # or a default value

    return similarities, sentences

def get_sequential_clusters(
    sentences:List[str], buffer_size:int=1, breakpoint_percentile:int=5
) -> np.ndarray:
    """Function to get the clusters for a semantic chunking maintaining
    the sentences order

    Args:
        sentences (List[str]): List of sentences
        buffer_size (int, optional): _description_. Number of sentences to combine
            from before and after the present one
        breakpoint_percentile (int, optional): _description_. Defaults to 5.

    Returns:
        np.ndarray: Array containing the clusters' labels
    """

    sentences = [
        {'index': i, 'sentence': sentence} for i, sentence in enumerate(sentences)
    ]
    
    sentences = combine_sentences(sentences, buffer_size=buffer_size)
    
    similarities, sentences = calculate_cosine_similarities(sentences)
    
    threshold = np.percentile(similarities, breakpoint_percentile)
    
    breakpoints = np.where(similarities < threshold)[0]
    
    sentences_per_cluster = np.diff(
        np.concatenate(([0], breakpoints, [len(sentences)]))
    )
    
    labels = np.concatenate(
        [np.repeat((i+1), n) for i,n in enumerate(sentences_per_cluster)]
    )
        
    return labels

def get_sequential_semantic_chunks(text:str, doc_id:str) -> List[dict]:
    """Function that takes a text and splitts it into chunks
    using semantic grouping and maintaining the sentences order

    Args:
        text (str): Text to be splitted
        doc_id (str): sha256 codified ID for each document

    Returns:
        List[dict]: List of dictionaries containing the
        document id, chunk_number and text each
    """
    sentences = sent_tokenize(text)
    # Create a sentence embedding model
    model = SentenceTransformer('all-mpnet-base-v2')

    embeddings = model.encode(sentences)
    
    silhouette_scores = []
    parameters = []
    for buffer_size in [1,2,3]:
        for breakpoint_percentile in [5, 10]:
            clusters_labels = get_sequential_clusters(sentences, buffer_size, breakpoint_percentile)
            score = silhouette_score(embeddings, clusters_labels)
            silhouette_scores.append(score)
            parameters.append({
                "buffer_size": buffer_size,
                "breakpoint_percentile": breakpoint_percentile
            })
        
    best_params = parameters[np.argmax(silhouette_scores)]
    clusters_labels = get_sequential_clusters(sentences, **best_params)
    
    # Create chunks based on cluster assignments
    chunks = [[] for _ in range(clusters_labels.max())]
    for i, cluster_id in enumerate(clusters_labels):
        chunks[cluster_id-1].append(sentences[i])

    chunks_dicts = []
    for i, chunk in enumerate(chunks):
        chunks_dicts.append({
            "doc_id": doc_id,
            "chunk": i + 1,
            "text": "\n".join(chunk)
        })
        
    return chunks_dicts