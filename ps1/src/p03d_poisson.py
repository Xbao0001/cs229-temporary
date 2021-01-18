import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    poisson_reg = PoissonRegression(step_size=lr)
    poisson_reg.fit(x_train, y_train)

    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=False)
    y_pred = poisson_reg.predict(x_valid)
    np.savetxt(pred_path, y_pred)

    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.theta = np.zeros(n)
        
        while True:
            theta_old = np.copy(self.theta)
            h_x = np.exp(x.dot(self.theta))
            gradient = x.T.dot(y - h_x)
            self.theta += self.step_size * gradient / m # /m or about 3000 is very important, else it cannot converge
            if np.linalg.norm(self.theta - theta_old, ord=1) < self.eps:
                break
    
    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(x.dot(self.theta))
        # *** END CODE HERE ***
