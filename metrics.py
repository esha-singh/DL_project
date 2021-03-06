from typing import Dict, Tuple, Any
def global_average_precision_score_test(
        y_true: Dict[Any, Any],
        y_pred: Dict[Any, Tuple[Any, float]]
) -> float:
    """
    Compute Global Average Precision score (GAP)
        GAP score
    """

    indexes = list(y_pred.keys())
    indexes.sort(
        key=lambda x: -y_pred[x][1],
    )
    queries_with_target = len([i for i in y_true.values() if i is not None])
    print(queries_with_target)
    correct_predictions = 0
    total_score = 0.
    accuracy = 0
    for i, k in enumerate(indexes, 1):
        relevance_of_prediction_i = 0
        if y_true[k] is not None:
            for y_t in y_true[k]:
                if y_t == y_pred[k][0]:
                    correct_predictions += 1
                    accuracy += 1
                    relevance_of_prediction_i = 1
        precision_at_rank_i = correct_predictions / i
        total_score += precision_at_rank_i * relevance_of_prediction_i
    accuracy /= queries_with_target
    
    return 1 / queries_with_target * total_score, accuracy*100

def global_average_precision_score_val(
        y_true: Dict[Any, Any],
        y_pred: Dict[Any, Tuple[Any, float]]
) -> float:
    """
    Compute Global Average Precision score (GAP)
        GAP score
    """
    indexes = list(y_pred.keys())
    indexes.sort(
        key=lambda x: -y_pred[x][1],
    )
    queries_with_target = len([i for i in y_true.values() if i is not None])
    correct_predictions = 0
    total_score = 0.
    for i, k in enumerate(indexes, 1):
        relevance_of_prediction_i = 0
        if y_true[k] == y_pred[k][0]:
            correct_predictions += 1
            relevance_of_prediction_i = 1
        precision_at_rank_i = correct_predictions / i
        total_score += precision_at_rank_i * relevance_of_prediction_i

    return 1 / queries_with_target * total_score