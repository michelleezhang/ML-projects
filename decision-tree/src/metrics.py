import numpy as np
# Note: do not import additional libraries to implement these functions


def compute_confusion_matrix(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the confusion matrix. The confusion
    matrix for a binary classifier would be a 2x2 matrix as follows:

    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]

    You do not need to implement confusion matrices for labels with more
    classes. You can assume this will always be a 2x2 matrix.

    Compute and return the confusion matrix.

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        confusion_matrix (np.array): 2x2 confusion matrix between predicted and actual labels

    """

    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    true_positives = 0

    for i in range(predictions.shape[0]):
        if actual[i] == False: # if negative 
            if actual[i] == predictions[i]: 
                true_negatives += 1
            else:
                false_positives += 1
        else:
            if actual[i] == predictions[i]: 
                true_positives += 1
            else:
                false_negatives += 1

    confusion_matrix = np.array([
        [true_negatives, false_positives], 
        [false_negatives, true_positives]])

    return confusion_matrix


def compute_accuracy(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the accuracy:

    Hint: implement and use the compute_confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        accuracy (float): accuracy
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    confusion_matrix = compute_confusion_matrix(actual, predictions)

    numerator = confusion_matrix[0][0] + confusion_matrix[1][1]
    denominator = numerator + confusion_matrix[1][0] + confusion_matrix[0][1]

    if denominator == 0:
        return 0

    accuracy = numerator / denominator
    return accuracy
    # (tp + tn) / (tp +  tn + fp + fn)


def compute_precision_and_recall(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the precision and recall:

    https://en.wikipedia.org/wiki/Precision_and_recall

    You MUST account for edge cases in which precision or recall are undefined
    by returning np.nan in place of the corresponding value.

    Hint: implement and use the compute_confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output a tuple containing:
        precision (float): precision
        recall (float): recall
    """
   

    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    
    # precision and recall
    confusion_matrix = compute_confusion_matrix(actual, predictions)
    
    p_denominator = confusion_matrix[1][1] + confusion_matrix[0][1]
    r_denominator = confusion_matrix[1][1] + confusion_matrix[1][0]

    if p_denominator == 0 or r_denominator == 0:
        return np.nan, np.nan

    precision = confusion_matrix[1][1] / p_denominator
    # tp / (tp + fp)
    recall = confusion_matrix[1][1] / r_denominator
    # tp / (tp + fn)

    return precision, recall


def compute_f1_measure(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the F1-measure:

    https://en.wikipedia.org/wiki/Precision_and_recall#F-measure

    Because the F1-measure is computed from the precision and recall scores, you
    MUST handle undefined (NaN) precision or recall by returning np.nan. You
    should also consider the case in which precision and recall are both zero.

    Hint: implement and use the compute_precision_and_recall function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        f1_measure (float): F1 measure of dataset (harmonic mean of precision and
        recall)
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    
    p, r = compute_precision_and_recall(actual, predictions)

    if (p + r) == 0 or (p == 0 and r == 0):
        return np.nan
    
    f_measure = (2 * p * r) / (p + r)
    return f_measure
