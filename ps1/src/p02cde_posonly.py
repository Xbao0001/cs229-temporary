import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c

    # load train set and preprocess
    x_train, t_train = util.load_dataset(
        train_path, label_col='t', add_intercept=True)

    # train a Logistic Regression model
    logreg_t = LogisticRegression()
    logreg_t.fit(x_train, t_train)

    # load test set and preprocess
    x_test, t_test = util.load_dataset(
        test_path, label_col='t', add_intercept=True)

    # make predictions and save results
    t_pred = logreg_t.predict(x_test)
    np.savetxt(pred_path_c, t_pred)

    # plot the decision boundry on test set
    util.plot(x_test, t_test, logreg_t.theta,
              '{}.png'.format(pred_path_c[:-4]))

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    # load train set and preprocess
    x_train, y_train = util.load_dataset(
        train_path, label_col='y', add_intercept=True)

    # train a Logistic Regression model on y labels
    logreg_y = LogisticRegression()
    logreg_y.fit(x_train, y_train)

    # make predictions and save results
    y_pred = logreg_y.predict(x_test)
    np.savetxt(pred_path_d, y_pred)

    # plot the decision boundry on test set
    util.plot(x_test, t_test, logreg_y.theta,
              '{}.png'.format(pred_path_d[:-4]))

    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e

    x_valid, y_valid = util.load_dataset(
        valid_path, label_col='y', add_intercept=True)
    # x_valid, t_valid = util.load_dataset(                 # test the result in 2a
    #     valid_path, label_col='t', add_intercept=True)    # test the result in 2a
    y_pred_valid = logreg_y.predict(x_valid)

    alpha = sum(y_pred_valid[y_valid == 1]) / sum(y_valid == 1)
    # print(alpha)                                          # test the result in 2a
    # print(sum(y_valid == 1)/sum(t_valid == 1))            # test the result in 2a
    np.savetxt(pred_path_e, y_pred / alpha)
    util.plot(x_test, t_test, logreg_y.theta, '{}.png'.format(
        pred_path_e[:-4]), correction=alpha)

    # *** END CODER HERE
