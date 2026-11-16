from hm_clothes_rec.splits import load_split


def load_truth(split):
    '''customer_id -> set of article_ids they bought during the split.'''
    rows = load_split(split).aggregate(
        'customer_id, list(DISTINCT article_id) AS items', 'customer_id'
    ).fetchall()
    return {customer: set(items) for customer, items in rows}


def _ranked(items):
    # a repeated item must not earn credit twice, so keep its first position
    return list(dict.fromkeys(items))


def recall_at_k(ranked, actual, k):
    return len(set(ranked[:k]) & actual) / len(actual)


def average_precision_at_k(ranked, actual, k=12):
    hits = 0
    score = 0.0
    for position, item in enumerate(ranked[:k], start=1):
        if item in actual:
            hits += 1
            score += hits / position
    return score / min(len(actual), k)


def evaluate(predictions, truth, ks=(12, 50, 100)):
    '''
    Score ranked predictions against what customers really bought.

    predictions: customer_id -> list of article_ids, best first
    truth: output of load_truth

    Averages over every customer in truth. A customer with no prediction
    scores 0, so a model can't raise its score by skipping hard customers.
    '''
    recall_totals = dict.fromkeys(ks, 0.0)
    map_total = 0.0

    for customer, actual in truth.items():
        ranked = _ranked(predictions.get(customer, []))
        map_total += average_precision_at_k(ranked, actual, 12)
        for k in ks:
            recall_totals[k] += recall_at_k(ranked, actual, k)

    n = len(truth)
    results = {'map@12': map_total / n}
    results.update({f'recall@{k}': total / n for k, total in recall_totals.items()})
    return results
