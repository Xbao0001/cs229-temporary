import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***

    # Train a logistic regression classifier
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # Plot decision boundary on top of validation set
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)
    util.plot(x_valid, y_valid, model.theta, '{}.png'.format(pred_path[:-4]))

    # Use np.savetxt to save predictions on eval set to pred_path
    y_pred = model.predict(x_valid)
    np.savetxt(pred_path, y_pred)

    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***

        # init theta
        m, n = x.shape
        # print(m, n)
        self.theta = np.zeros(n)

        # Newton's method
        while True:
            # save old theta
            theta_old = np.copy(self.theta)

            # compute Hessian Matrix
            g = 1/(1 + np.exp(-x.dot(self.theta)))
            hessian = 1 / m * np.dot(g, (1-g).T) * np.dot(x.T, x)

            # compute nabla
            nabla = 1 / m * np.dot(x.T, g - y)

            # update theta
            self.theta -= np.linalg.inv(hessian).dot(nabla)

            if np.linalg.norm(self.theta - theta_old, ord=1) < self.eps:
                break

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1/(1 + np.exp(-x.dot(self.theta)))
        # *** END CODE HERE ***
