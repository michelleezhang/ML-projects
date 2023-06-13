import numpy as np


def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.

    (Hint: You're free to implement this with numpy.linalg.norm)

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """

    result = []
    for i_row in X: 
        curr_row = []
        for j_row in Y:
            euclidean_dist = np.sqrt(np.sum((i_row - j_row)**2))
            curr_row.append(euclidean_dist)
        result.append(curr_row)
    
    result = np.row_stack(result)
    return result


def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.

    (Hint: You're free to implement this with numpy.linalg.norm)

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """

    result = []
    for i_row in X:
        curr_row = []
        for j_row in Y:
            manhattan_dist = np.sum(np.absolute(i_row - j_row))
            curr_row.append(manhattan_dist)
        result.append(curr_row)
    
    result = np.row_stack(result)
    return result


def cosine_distances(X, Y):
    """Compute pairwise Cosine distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Cosine distance between two rows.

    (Hint: You're free to implement this with numpy.linalg.norm)

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Cosine distances between rows of X and rows of Y.
    """
    
    result = []
    for i_row in X:
        curr_row = []
        for j_row in Y:
            # calculate the cosine similarity
            cos_sim = np.sum(i_row * j_row) / (0.000000001 + np.sqrt(np.sum(i_row ** 2)) * np.sqrt(np.sum(j_row ** 2)))
            # append cosine distance (1 - similarity) to current row vector
            curr_row.append(1 - cos_sim)
        # once we've gone through the entire j_row, we append the row to the overall result vector
        result.append(curr_row)
    
    # stack the vector together to make a matrix 
    result = np.row_stack(result)
    return result
