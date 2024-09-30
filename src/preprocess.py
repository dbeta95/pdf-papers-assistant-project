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
from typing import List, Union, Iterable
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

def map_progress(pool:ThreadPoolExecutor, seq:Iterable, f):
    """Function to map a function to a sequence and show a progress bar
    
    Args:
        pool (ThreadPoolExecutor): ThreadPoolExecutor object
        seq (Iterable): Iterable object to be processed
        f (function): Function to be applied to each element in the sequence
        
    Returns:
        List: List of results from the function applied to each element
    """
    results = []

    with tqdm(total=len(seq)) as progress:
        futures = []

        for el in seq:
            future = pool.submit(f, el)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)

        for future in futures:
            result = future.result()
            results.append(result)

    return results

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


def combine_sentences(
    sentences:List[dict], 
    model:SentenceTransformer,
    buffer_size:int=1
) -> List[dict]:
    """Function that combines sentences according to a buffer size to give
    stability to the distance comparison between chunks

    Args:
        sentences (List[dict]): List of dictionaries where each one contains
            a sentece an it's index
        model (SentenceTransformer): Sentence embedding model.
        buffer_size (int, optional): _description_. Number of sentences to combine
            from before and after the present one
        
    Returns:
        List[dict]: List of dictionaries where each one contains
            a sentece, it's index and the combined text
    """
    
    combined_sentences = []
    n = len(sentences)
    # Precompute the combined sentences
    for i in range(n):
        # Get the range of indices for sentences before, current, and after
        start_idx = max(0, i - buffer_size)
        end_idx = min(len(sentences), i + 1 + buffer_size)
        
        # Combine sentences using list comprehension and str.join
        combined_sentences.append(' '.join([sentences[j]['sentence'] for j in range(start_idx, end_idx)]))
    
    combined_sentences_embeddings = list(map(lambda x: model.encode(x), combined_sentences))
    # # Store the combined sentence and its embedding in the current sentence dict
    for i in range(n):
        sentences[i]['combined_sentence'] = combined_sentences[i]
        sentences[i]['combined_sentence_embedding'] = combined_sentences_embeddings[i]
        
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

    return similarities, sentences

def get_sequential_clusters(
    sentences:List[str],
    model:SentenceTransformer, 
    buffer_size:int=1, 
    breakpoint_percentile:int=5    
) -> np.ndarray:
    """Function to get the clusters for a semantic chunking maintaining
    the sentences order

    Args:
        sentences (List[str]): List of sentences
        model (SentenceTransformer, optional): Sentence embedding model.
        buffer_size (int, optional): _description_. Number of sentences to combine
            from before and after the present one
        breakpoint_percentile (int, optional): _description_. Defaults to 5.
         
    Returns:
        np.ndarray: Array containing the clusters' labels
    """

    sentences = [
        {'index': i, 'sentence': sentence} for i, sentence in enumerate(sentences)
    ]
    
    sentences = combine_sentences(sentences, model=model, buffer_size=buffer_size)
    
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

def get_sequential_semantic_chunks(
    text:str, 
    doc_id:str,
    workers:int=4
) -> List[dict]:
    """Function that takes a text and splitts it into chunks
    using semantic grouping and maintaining the sentences order

    Args:
        text (str): Text to be splitted
        doc_id (str): sha256 codified ID for each document
        workers (int, optional): Number of workers to use. Defaults to 4.

    Returns:
        List[dict]: List of dictionaries containing the
        document id, chunk_number and text each
    """
    sentences = sent_tokenize(text)
    
    if len(sentences) < 2:
        return [{
            "doc_id": doc_id,
            "chunk": 1,
            "text": text
        }]
        
    # Create a sentence embedding model
    model = SentenceTransformer('all-mpnet-base-v2')
    pool = ThreadPoolExecutor(max_workers=workers)

    embeddings = model.encode(sentences)
    
    silhouette_scores = []
    parameters = [
        {"buffer_size":buffer_size, "breakpoint_percentile":breakpoint_percentile} 
        for buffer_size in [1,2,3] for breakpoint_percentile in [5, 10]
    ]
    def get_silhoutte_score(enum_param):
        i, params = enum_param
        try:
            clusters_labels = get_sequential_clusters(sentences, model, **params)
            score = silhouette_score(embeddings, clusters_labels)
        except:
            score = -1
        return i, score
    
    results_list = map_progress(pool, list(enumerate(parameters)), get_silhoutte_score)
    silhouette_scores = [score for _, score in results_list]
    best_index = results_list[np.argmax(silhouette_scores)][0]
    best_params = parameters[best_index]
    clusters_labels = get_sequential_clusters(sentences, model, **best_params)
    
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