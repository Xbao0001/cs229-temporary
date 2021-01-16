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
    
    ## load train set and preprocess
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)

    ## train a Logistic Regression model
    logreg = LogisticRegression()
    logreg.fit(x_train, t_train)

    ## load validation set and preprocess
    x_valid, t_valid = util.load_dataset(valid_path, label_col='t', add_intercept=True)
    
    ## plot
    util.plot(x_valid, t_valid, logreg.theta, '{}.png'.format(pred_path_c[:-4]))

    ## make predictions and save results
    y_pred = logreg.predict(x_valid)
    np.savetxt(pred_path_c, y_pred)

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    # *** END CODER HERE
