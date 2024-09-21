"""
Module for the definition of general functions for optimization
"""
import random
from tqdm.auto import tqdm

def simple_optimize(
    param_ranges:dict, 
    objective_function,
    n_iterations:int=100
):
    best_params = None
    best_score = float("inf") # For minimizing. Use float('-inf') if maximizing 
    
    for _ in tqdm(range(n_iterations)):
        
        # Generate random parameters
        current_params = {}        
        for param, (min_val, max_val) in param_ranges.items():
            # Generate random integer
            if isinstance(min_val, int) and isinstance(max_val, int):
                current_params[param] = random.randint(min_val, max_val)
            # Generate random boolean
            elif isinstance(min_val, bool) and isinstance(max_val, bool):
                current_params[param] = random.choice([True, False])
            # Generate random float
            else:
                current_params[param] = random.uniform(min_val, max_val)
        
        # Evaluate the objective function
        current_score = objective_function(current_params)
        
        # Update best if current is better
        if current_score < best_score: # minimizing
            best_score = current_score
            best_params = current_params
    
    return best_params, best_score