from sklearn.metrics import precision_score, recall_score

def evaluateModel(trueLabels, predictedLabels):
    """
    Evaluate the model using precision and recall.

    Parameters:
        trueLabels (list or array): Ground truth (correct) labels.
        predictedLabels (list or array): Predicted labels by the model.

    Returns:
        tuple: Precision and recall scores.
    """
    precision = precision_score(trueLabels, predictedLabels)
    recall = recall_score(trueLabels, predictedLabels)
    return precision, recall

if __name__ == "__main__":
    trueLabels = [1, 0, 1, 1, 0]
    predictedLabels = [1, 0, 1, 0, 0]

    precision, recall = evaluateModel(trueLabels, predictedLabels)

    print(f"Precision: {precision}, Recall: {recall}")