import numpy as np


class BinaryCrossEntropyLoss:
    def forward(self, y_pred, y_true):
        """
        Save the inputs to self.input_ and then
            compute the binary cross-entropy loss

        You will not need to edit this function.
        """
        assert set(np.unique(y_true)).issubset(set([0, 1]))
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)

        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)

        self.input_ = (y_pred, y_true)
        grad = np.where(y_true, -np.log(y_pred), -np.log(1 - y_pred))
        return np.mean(grad)

    def backward(self, grad=None, lr=None):
        """
        Compute the gradient of the loss function
        `grad` and `lr` are left as arguments to match the other
            `backward` functions, but will never be passed anything.

        You will not need to edit this function.
        """
        assert grad is None
        (y_pred, y_true) = self.input_
        ret = (- y_true + y_pred) / (y_pred - y_pred ** 2)
        return ret


class SquaredLoss:
    def forward(self, y_pred, y_true):
        """
        Save the inputs to self.input_ and then compute the **mean** squared
        error loss.
        While sometimes MSE is written including a factor of 1/2, because
          `d/dx (1/2 x ^ 2) = x`, please do not include that factor here.
        """
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)

        # save the inputs to self.input_
        # and compute mean squared error loss

        # calculate error for each example (need to square the term),
        # sum them up
        # divide by the total number of examples.

        # so y_pred is a column
        # y_true is a column

        self.input_ = (y_pred, y_true)

        mse = np.mean((y_true - y_pred)**2)

        # equivalent to:
        # mse = 0
        # for i in range(y_pred.size):
        #     mse += (y_true[i][0] - y_pred[i][0])**2 
        # mse = mse / y_pred.size

        return mse
    

    def backward(self, grad=None, lr=None):
        """
        Compute the gradient of the loss function
        Should use the arguments saved to self.input_
            from the last time `forward()` was called.
        `grad` and `lr` are left as arguments to match the other
            `backward` functions, but will never be passed anything.
        """
        assert grad is None

        # get the y_pred and y_truee
        (y_pred, y_true) = self.input_

        # find the gradient
        # the derivative is simply 2 (y_pred - y_true)
        loss_gradient = 2 * (y_pred - y_true)

        return loss_gradient