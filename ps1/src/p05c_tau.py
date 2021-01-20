import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set and validation set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    min_mse = float('inf')
    i = 1
    for tau in tau_values:
        lwr_model = LocallyWeightedLinearRegression(tau)
        lwr_model.fit(x_train, y_train)
        y_pred = lwr_model.predict(x_valid)

        m = y_pred.shape
        MSE = (y_valid - y_pred).T.dot(y_valid - y_pred) / m
        print(f"For tau = {tau}, MSE on the validation set is: {MSE}.")
        if MSE < min_mse:
            best_tau = tau
            min_mse = MSE

        plt.clf()
        plt.plot(x_train, y_train, 'bx', x_valid, y_pred, 'ro')
        plt.savefig(f'output/p05c_{i}.png')
        i += 1
    print(f"Best tau is: {best_tau}.")

    # Fit a LWR model with the best tau value
    lwr_model = LocallyWeightedLinearRegression(best_tau)
    lwr_model.fit(x_train, y_train)

    # Run on the test set to get the MSE value
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    
    y_pred = lwr_model.predict(x_test)
    m = y_pred.shape
    MSE = (y_test - y_pred).T.dot(y_test - y_pred) / m
    print(f"For tau = {best_tau}, MSE on the test set is: {MSE}.")
    
    # Save predictions to pred_path
    np.savetxt(pred_path, y_pred)
    
    # Plot data


    # *** END CODE HERE ***
