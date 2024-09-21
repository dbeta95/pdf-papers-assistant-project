import functools
import operator
from tqdm.auto import tqdm

def hit_rate(relevance_total):
    cnt = functools.reduce(operator.add, [sum(row) for row in relevance_total])
    return cnt/len(relevance_total)

def mmr(relevance_total):
    total_score = functools.reduce(
        operator.add, 
        [
            sum(row) for row in [
                [int(element)/(i + 1) for i, element in enumerate(row)] 
                for row in relevance_total
            ]
        ]
    )
    return total_score / len(relevance_total)

def evaluate(ground_truth, search_function):

    relevance_total = []
    for q in tqdm(ground_truth):
        doc_id = q['id']
        results = search_function(q)
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)

    return {
        "hit_rate": hit_rate(relevance_total),
        "mmr": mmr(relevance_total)
    }