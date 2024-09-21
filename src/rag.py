"""
Module definig classes and fucntions to create a basic RAG
system using gemini.
"""
import os 
import sys
import json

import google.generativeai as genai # Change for preffered LLM library

from time import time
from functools import reduce
from typing import List

src_path = os.path.dirname(__file__)
sys.path.append(src_path)
import hybsearch
from minsearch import Index
from optimization import simple_optimize
from evaluation import evaluate

## Settings for Gemini
# modify according to the model you are using
default_safety_settings={
    'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
    'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
    'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
    'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
}

# AS of september 2024, the cost of using the LLM is as follows:
def calculate_google_cost(model_choice, tokens):
    google_cost = 0

    if model_choice in ['models/gemini-1.5-flash-latest', 'models/gemini-1.5-flash-exp-0827']:
        google_cost = (tokens['prompt_tokens'] * 0.075 + tokens['completion_tokens'] * 0.3) / 1000000
    elif model_choice in ['models/gemini-1.5-pro-exp-0827', 'models/gemini-1.5-pro-latest']:
        google_cost = (tokens['prompt_tokens'] * 3.5 + tokens['completion_tokens'] * 10.5) / 1000000

    return google_cost

class RAG():
    """Class defyning the RAG
    """
    
    def __init__(self, 
        api_key:str,        
        llm_model:str="models/gemini-1.5-flash-latest",
        candidate_count:int=1,
        temperature:int=0,
        safety_settings:dict[str,str]=default_safety_settings,
        filter_dict:dict={}, 
        boost_dict:dict={},
        alpha:float=0.5,
        rrf:bool=False,
        k:int=60,
        num_results:int=10
    ) -> None:
        """
        Initialization method which configures the llm model
        to be used in the RAG
        
        Args:
            llm_model (str): Name of the model for the Gemini app. Can be models like:
                - 'models/gemini-1.5-flash-latest'
                - 'models/gemini-1.5-pro-exp-0827'
                - 'models/gemini-1.5-pro-latest'
                - 'models/gemini-1.5-flash-exp-0827'
            candidat_count (int): Number of candidates to be returned
            temperature (int):Temperature for the LLM
            safety_settings (dict): Dictionary of settings for gemini's safetty configuration
            filter_dict (dict): Dictionary of keyword fields to filter by. Keys are field names and values are the values to filter by.
            boost_dict (dict): Dictionary of boost scores for text fields. Keys are field names and values are the boost scores.
            rrf (bool): Whether to apply Reciprocal Rank Fusion to the search results. Defaults to False.
            k (int): Constat for the RRF formula
            num_results (int): The number of top results to return. Defaults to 10.
        """    
    
        genai.configure(api_key=api_key)
        self.model = llm_model
        self.llm_model = genai.GenerativeModel(llm_model)
        
        self.generation_config = genai.types.GenerationConfig(
            candidate_count=candidate_count,
            temperature=temperature
        )
        
        self.safety_settigns = safety_settings
        
        self.filter_dict = filter_dict
        self.boost_dict = boost_dict
        self.h_boost_dict = boost_dict
        self.num_results = num_results
        self.alpha = alpha
        self.rrf = rrf,
        self.k = k
        
    def update_parameters(self, **kwargs):
        """Updates class attributes based on the provided keyword arguments.

        Args:
            **kwargs: Keyword arguments where the keys correspond to class attributes
                   and the values are the new values to assign.

        Raises:
            AttributeError: If a provided key does not correspond to an existing attribute.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Attribute '{key}' does not exist.")
        
    def llm(self, prompt:str) -> str:
        """Method to get the llm response given a prompt

        Args:
            prompt (str): Input prompt
            
        Returns:
            str: Response given by the LLM
        """
        
        responses = self.llm_model.generate_content(
            contents=prompt,
            safety_settings=self.safety_settigns,
            generation_config=self.generation_config
        )
        
        answer = "".join(response.text for response in responses)
        
        prompt_tokens = reduce(lambda a, b: a + b, [
            response.usage_metadata.prompt_token_count for response in responses
        ])
        completion_tokents = reduce(lambda a, b: a + b, [
            response.usage_metadata.candidates_token_count for response in responses
        ])
        
        total_tokents = reduce(lambda a, b: a + b, [
            response.usage_metadata.total_token_count for response in responses
        ])

        token_stats = {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokents,
            'total_tokens': total_tokents
        }
        
        return answer, token_stats
    
    def minsearch_index(self,
            docs:list[dict], 
            text_fields:list[str], 
            keyword_fields:list[str], 
            vectorizer_params:dict={}
        ) -> None:
        """Initializes the Index with specified text and keyword fields.

        Args:
            docs (list of dict): List of documents to index. Each document is a dictionary.
            text_fields (list): List of text field names to index.
            keyword_fields (list): List of keyword field names to index.
            vectorizer_params (dict): Optional parameters to pass to TfidfVectorizer.
        """
        self.index = Index(
            text_fields=text_fields, 
            keyword_fields=keyword_fields,
            vectorizer_params=vectorizer_params
        )
        self.index.fit(docs=docs)
        
    def hybserach_index(self,
            docs:list[dict], 
            text_fields:list[str], 
            keyword_fields:list[str], 
            vectorizer_params:dict={},
            embedding_model_name='all-mpnet-base-v2'
        ) -> None:
        """Initializes the Index with specified text and keyword fields.

        Args:
            docs (list of dict): List of documents to index. Each document is a dictionary.
            text_fields (list): List of text field names to index.
            keyword_fields (list): List of keyword field names to index.
            vectorizer_params (dict): Optional parameters to pass to TfidfVectorizer.
        """
        self.h_index = hybsearch.Index(
            text_fields=text_fields, 
            keyword_fields=keyword_fields,
            vectorizer_params=vectorizer_params,
            embedding_model_name=embedding_model_name
        )
        self.h_index.fit(docs=docs)
    
    def minsearch(self, 
        query:str,
        filter_dict:dict=None, 
        boost_dict:dict=None,
        num_results:int=None
    ) -> list: 
        """
        Searches the index with the given query.

        Args:
            query (str): The search query string.
            filter_dict (dict): Dictionary of keyword fields to filter by. Keys are field names and values are the values to filter by.
            boost_dict (dict): Dictionary of boost scores for text fields. Keys are field names and values are the boost scores.
            num_results (int): The number of top results to return. Defaults to 10.         

        Returns:
            list of dict: List of documents matching the search criteria, ranked by relevance.
        """
        if self.index is None:
            raise NotImplementedError("Documents not indexed")
        
        if filter_dict is None:
            filter_dict=self.filter_dict
            
        if boost_dict is None:
            boost_dict=self.boost_dict
            
        if num_results is None:
            num_results=self.num_results

        return self.index.search(
            query=query,
            filter_dict=filter_dict,
            boost_dict=boost_dict,
            num_results=num_results
        )
        
    def hybsearch(self, 
        query:str,
        filter_dict:dict=None, 
        boost_dict:dict=None,
        alpha:float=None,
        num_results:int=None,
        rrf:bool=None,
        k:int=None
    ) -> list: 
        """
        Searches the index with the given query.

        Args:
            query (str): The search query string.
            filter_dict (dict): Dictionary of keyword fields to filter by. Keys are field names and values are the values to filter by.
            boost_dict (dict): Dictionary of boost scores for text fields. Keys are field names and values are the boost scores.
            num_results (int): The number of top results to return. Defaults to 10.         

        Returns:
            list of dict: List of documents matching the search criteria, ranked by relevance.
        """
        if self.h_index is None:
            raise NotImplementedError("Documents not indexed")
        
        if filter_dict is None:
            filter_dict=self.filter_dict
            
        if boost_dict is None:
            boost_dict=self.h_boost_dict
            
        if num_results is None:
            num_results=self.num_results
            
        if alpha is None:
            alpha=self.alpha
            
        if rrf is None:
            rrf=self.rrf
        
        if k is None:
            k=self.k    
        
        return self.h_index.search(
            query=query,
            filter_dict=filter_dict,
            boost_dict=boost_dict,
            alpha=alpha,
            num_results=num_results,
            rrf=rrf,
            k=k
        )
        
    def __minsearch_objective__(self, params:dict) -> float:
        """Abstract method defining the objective function for the minsearch
        hyperparameters optimization

        Args:
            params (dict): Dictionary of parameters

        Returns:
            float: score with the passed parameters
        """
        def search_function(query):
            return self.minsearch(query['question'], boost_dict=params)
        
        results = evaluate(self.ground_truth, search_function)
        
        return -results['mmr']
    
    def __hybsearch_objective__(self, params:dict) -> float:
        """Abstract method defining the objective function for the hybsearch
        hyperparameters optimization

        Args:
            params (dict): Dictionary of parameters

        Returns:
            float: score with the passed parameters
        """
        boost_params = params.copy()
        alpha = boost_params.pop('alpha')
        rrf = boost_params.pop('rrf')
        k = boost_params.pop('k')
        def search_function(query):
            return self.hybsearch(
                query['question'], 
                boost_dict=boost_params,
                alpha=alpha,
                rrf=rrf,
                k=k
            )
        
        results = evaluate(self.ground_truth, search_function)
        
        return -results['mmr']
    
    def minserach_fit(self, 
            ground_truth: List[dict],
            param_ranges: dict[tuple],
            n_iterations:int=100
        ) -> None:
        """
        Fits the model using the provided ground truth data.

        Args:
            ground_truth (List[dict]): List of dictionaries containing 
                ground truth data.
            parameters_ranges (dict[tuple]): Dictionary containing the
                ranges for each parameter to be optimized.
            n_iterations (int): Numbers of iterations in the optimization
                process. Defaults to 100.
        """
        self.ground_truth = ground_truth
        
        best_params, best_score = simple_optimize(
            param_ranges=param_ranges,
            objective_function=self.__minsearch_objective__, 
            n_iterations=n_iterations
        )
        
        self.boost_dict = best_params
        print("Model fitted with provided ground truth data.")
        print(f"Best parameters are:\n{best_params}")
        print(f"Best score was:\n{-best_score}")
        
        
    def hybsearch_fit(self, 
            ground_truth: List[dict],
            param_ranges: dict[tuple],
            n_iterations:int=100
        ) -> None:
        """
        Fits the model using the provided ground truth data.

        Args:
            ground_truth (List[dict]): List of dictionaries containing 
                ground truth data.
            parameters_ranges (dict[tuple]): Dictionary containing the
                ranges for each parameter to be optimized.
            n_iterations (int): Numbers of iterations in the optimization
                process. Defaults to 100.
        """
        self.ground_truth = ground_truth
        
        best_params, best_score = simple_optimize(
            param_ranges=param_ranges,
            objective_function=self.__hybsearch_objective__, 
            n_iterations=n_iterations
        )
        
        self.h_boost_dict = best_params.copy()
        self.alpha = self.h_boost_dict.pop('alpha')
        self.rrf = self.h_boost_dict.pop('rrf')
        self.k = self.h_boost_dict.pop('k')
      
        print("Model fitted with provided ground truth data.")
        print(f"Best parameters are:\n{best_params}")
        print(f"Best score was:\n{-best_score}")
        
        
    def set_prompt_templates(self,
        entry_template:str = None,
        prompt_template:str = None        
    ):
        """Sets the template to be used in the prompt

        Args:
            entry_template (str): Template to format each entry in the context
            prompt_template (str): Template for the prompt body
        """
        self.entry_template = entry_template
        self.prompt_template = prompt_template
          
        
    def build_prompt(self, query, search_results) -> str:
        """_summary_

        Args:
            query (_type_): User's query
            search_results (_type_): Results form the retrieval part

        Returns:
            str: prompt for the LLM
        """
        if self.entry_template is None or self.prompt_template is None:
            raise NotImplementedError("The entry o prompt template hasn't been defined")
        
        context = ""
        
        for doc in search_results:
            context += self.entry_template.format(**doc) + "\n\n"
            
        return self.prompt_template.format(question = query, context = context).strip()
    
    
    def evaluate_relevance(self, question, answer):
        prompt_template = """
        You are an expert evaluator for a RAG system.
        Your task is to analyze the relevance of the generated answer to the given question.
        Based on the relevance of the generated answer, you will classify it
        as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

        Here is the data for evaluation:

        Question: {question}
        Generated Answer: {answer}

        Please analyze the content and context of the generated answer in relation to the question
        and provide your evaluation in parsable JSON without using code blocks nor including the an
        string stating that it's json code:

        {{
        "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
        "Explanation": "[Provide a brief explanation for your evaluation]"
        }}
        """.strip()

        prompt = prompt_template.format(question=question, answer=answer)
        evaluation, tokens = self.llm(prompt)
        
        try:
            json_eval = json.loads(evaluation)
            return json_eval, tokens
        except json.JSONDecodeError:
            result = {
                "Relevance":"UNKNOWN",
                "Explanation": "Failed to parse evaluation"
            }
            return result, tokens    
    
    def answer(self, query:str, search:str="minsearch") -> str:
        """_summary_

        Args:
            query (str): User's query

        Returns:
            str: Response
        """
        
        if search.lower() == "minsearch":
            t0 = time()
            
            search_results = self.minsearch(query)
            prompt = self.build_prompt(query, search_results)
            answer, token_stats = self.llm(prompt)
            relevance, rel_token_stats = self.evaluate_relevance(query, answer)
            
            t1 = time()            
            took = t1 - t0
            
            google_cost_response = calculate_google_cost(self.model, token_stats)
            google_cost_eval = calculate_google_cost(self.model, rel_token_stats)
            
            answer_data = {
                "answer": answer,
                "model_used": self.model,
                "response_time": took,
                "relevance": relevance.get('Relevance', "UNKNOWN"),
                "relevance_explanation": relevance.get(
                    'Explanation', "Failed to parse evaluation"
                ),
                "prompt_tokens": token_stats['prompt_tokens'],
                "completion_tokens": token_stats['completion_tokens'],
                "total_tokens": token_stats['total_tokens'],
                "eval_prompt_tokens": rel_token_stats['prompt_tokens'],
                "eval_completion_tokens": rel_token_stats['completion_tokens'],
                "eval_total_tokens": rel_token_stats['total_tokens'],
                "google_cost": google_cost_response + google_cost_eval         
            }
             
            return answer_data
        
        if search.lower() == "hybsearch":
            
            t0 = time()
            
            search_results = self.hybsearch(query)
            prompt = self.build_prompt(query, search_results)
            answer, token_stats = self.llm(prompt)
            relevance, rel_token_stats = self.evaluate_relevance(query, answer)
            
            t1 = time()            
            took = t1 - t0
            
            google_cost_response = calculate_google_cost(self.model, token_stats)
            google_cost_eval = calculate_google_cost(self.model, rel_token_stats)
            
            answer_data = {
                "answer": answer,
                "model_used": self.model,
                "response_time": took,
                "relevance": relevance.get('Relevance', "UNKNOWN"),
                "relevance_explanation": relevance.get(
                    'Explanation', "Failed to parse evaluation"
                ),
                "prompt_tokens": token_stats['prompt_tokens'],
                "completion_tokens": token_stats['completion_tokens'],
                "total_tokens": token_stats['total_tokens'],
                "eval_prompt_tokens": rel_token_stats['prompt_tokens'],
                "eval_completion_tokens": rel_token_stats['completion_tokens'],
                "eval_total_tokens": rel_token_stats['total_tokens'],
                "google_cost": google_cost_response + google_cost_eval         
            }
             
            return answer_data

        