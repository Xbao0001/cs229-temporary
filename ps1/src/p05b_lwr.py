import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***

    # Fit a LWR model
    lwr_model = LocallyWeightedLinearRegression(tau)
    lwr_model.fit(x_train, y_train)
    
    # Get MSE value on the validation set
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)
    y_pred = lwr_model.predict(x_valid)
    m = y_pred.shape
    MSE = (y_valid - y_pred).T.dot(y_valid - y_pred) / m
    print(f"For tau = {tau}, MSE on the validation set is: {MSE}.")

    # Plot validation predictions on top of training set
    plt.clf()
    plt.plot(x_train, y_train, 'bx', x_valid, y_pred, 'ro')
    plt.savefig('output/p05b.png')
    # No need to save predictions
    # Plot data

    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        pred = np.zeros(m)
        p, q = self.x.shape
        w = np.eye(p, p)

        for i in range(m):
            # print(m)
            for j in range(p):
                # print(p)
                w[j, j] = np.exp(-(np.linalg.norm(self.x[j, :] - x[i, :], ord=2)) ** 2/(2 * self.tau ** 2))
            theta = np.linalg.inv(self.x.T.dot(w).dot(self.x)).dot(self.x.T).dot(w).dot(self.y)
            pred[i] = x[i, :].dot(theta)
        return pred
        # *** END CODE HERE ***
