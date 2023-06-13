import numpy as np
import warnings

from src.utils import softmax, stable_log_sum


class NaiveBayes:
    """
    A Naive Bayes classifier for binary data.
    """

    def __init__(self, smoothing=1):
        """
        Args:
            smoothing: controls the smoothing behavior when computing p(x|y).
                If the word "jackpot" appears `k` times across all documents with
                label y=1, we will instead record `k + self.smoothing`. Then
                `p("jackpot" | y=1) = (k + self.smoothing) / Z`, where Z is a
                normalization constant that accounts for adding smoothing to
                all words.
        """
        self.smoothing = smoothing

    def predict(self, X):
        """
        Return the most probable label for each row x of X.
        You should not need to edit this function.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        """
        Using self.alpha and self.beta, compute the probability p(y | X[i, :])
            for each row X[i, :] of X.  While you will have used log
            probabilities internally, the returned array should be
            probabilities, not log probabilities.

        See equation (9) in `naive_bayes.pdf` for a convenient way to compute
            this using your self.alpha and self.beta. However, note that
            (9) produces unnormalized log probabilities; you will need to use
            your src.utils.softmax function to transform those into probabilities
            that sum to 1 in each row.

        Args:
            X: a sparse matrix of shape `[n_documents, vocab_size]` on which to
               predict p(y | x)

        Returns 
            probs: an array of shape `[n_documents, n_labels]` where probs[i, j] contains
                the probability `p(y=j | X[i, :])`. Thus, for a given row of this array,
                np.sum(probs[i, :]) == 1.
        """
        n_docs, vocab_size = X.shape
        n_labels = 2

        assert hasattr(self, "alpha") and hasattr(self, "beta"), "Model not fit!"
        assert vocab_size == self.vocab_size, "Vocab size mismatch"

        # create probability array
        result = self.alpha + X @ self.beta

        # normalize using softmax (transform result so that each row sums to 1)
        result = softmax(result)

        return result
    

    def fit(self, X, y):
        """
        Compute self.alpha and self.beta using the training data.
        You should store log probabilities to avoid underflow.
        This function *should not* use unlabeled data. Wherever y is NaN, that
        label and the corresponding row of X should be ignored.

        See equations (10) and (11) in `naive_bayes.pdf` for the math necessary
            to compute your alpha and beta.

        self.alpha should be set to contain the marginal probability of each class label.

        self.beta should be set to the conditional probability of each word
            given the class label: p(w_j | y_i). This should be an array of shape
            [n_vocab, n_labels].  Remember to use `self.smoothing` to smooth word counts!
            See __init__ for details. If we see M total words across all N documents with
            label y=1, have a vocabulary size of V words, and see the word "jackpot" `k`
            times, then: `p("jackpot" | y=1) = (k + self.smoothing) / (M + self.smoothing *
            V)` Note that `p("jackpot" | y=1) + p("jackpot" | y=0)` will not sum to 1;
            instead, `sum_j p(word_j | y=1)` will sum to 1.

        Hint: when self.smoothing = 0, some elements of your beta will be -inf.
            If `X_{i, j} = 0` and `\beta_{j, y_i} = -inf`, your code should
            compute `X_{i, j} \beta_{j, y_i} = 0` even though numpy will by
            default compute `0 * -inf` as `nan`.

            This behavior is important to pass both `test_smoothing` and
            `test_tiny_dataset_a` simultaneously.

            The easy way to do this is to leave `X` as a *sparse array*, which
            will solve the problem for you. You can also explicitly define the
            desired behavior, or use `np.nonzero(X)` to only consider nonzero
            elements of X.

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: None; sets self.alpha and self.beta
        """
        n_docs, vocab_size = X.shape
        n_labels = 2
        self.vocab_size = vocab_size

        # soup

        # calculate alpha
        # make array of number of occurances of 0 in y and number of occurances of 1 in y
        alpha_array = np.array([np.count_nonzero(y == 0), np.count_nonzero(y == 1)])
        # divide by the number of labels (length of y), then log
        self.alpha = np.log(alpha_array / y.shape)

        # calculate beta
        # initialize variables for the fractions in beta
        top_0 = np.zeros((self.vocab_size, 1))
        bot_0 = 0
        top_1 = np.zeros((self.vocab_size, 1))
        bot_1 = 0
        
        # iterate through X (recall X is N x V)
        for i in range(n_docs):
            for j in range(self.vocab_size):
                # add the elements X_ij to the top and bottom
                if y[i] == 0:
                    top_0[j] += X[i, j]
                    bot_0 += X[i, j]
                elif y[i] == 1:
                    top_1[j] += X[i, j]
                    bot_1 += X[i, j]
        
        # concatenate the columns to create a matrix
        beta_array = np.hstack((
            (top_0 + self.smoothing) / (bot_0 + self.vocab_size * self.smoothing),
            (top_1 + self.smoothing) / (bot_1 + self.vocab_size * self.smoothing)))

        # beta is the log of this
        self.beta = np.log(beta_array)


    def likelihood(self, X, y):
        r"""
        Using fit self.alpha and self.beta, compute the log likelihood of the data.
            You should use logs to avoid underflow.
            This function should not use unlabeled data. Wherever y is NaN,
            that label and the corresponding row of X should be ignored.

        Equation (5) in `naive_bayes.pdf` contains the likelihood, which can be written:

            \sum_{i=1}^N \alpha_{y_i} + \sum_{i=1}^N \sum_{j=1}^V X_{i, j} \beta_{j, y_i}

            You can visualize this formula in http://latex2png.com

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: the log likelihood of the data
        """
        assert hasattr(self, "alpha") and hasattr(self, "beta"), "Model not fit!"

        n_docs, vocab_size = X.shape
        n_labels = 2

        # matrix to store the X @ beta sums
        bmat_sum = np.zeros((n_docs, n_labels)) # N x 2 (also length of beta)
        inf_check = False

        for i in range(n_docs):
            # ignore when y is NaN
            if not np.isnan(y[i]):
                for j in range(vocab_size):
                    for yi in range(n_labels):
                        # 0 and -inf cause NaN issues, so check for those before continuing
                        if X[i, j] != 0 or self.beta[j, yi] != -np.inf:
                            bmat_sum[i, yi] += X[i, j] * self.beta[j, yi]
                            if bmat_sum[i, yi] == -np.inf and y[i] == yi:
                                inf_check = True
        
        # if inf check, return negative infinity
        if inf_check:
            likelihood = -np.inf
        else:
            likelihood = stable_log_sum(self.alpha + bmat_sum) 
        
        return likelihood