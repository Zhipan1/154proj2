import numpy as np
import math

def CVgeneric(model, folds, X, y, loss):
    res = []
    # assign a fold to each row in X
    fold_index = np.repeat(list(range(folds)), math.ceil(len(X) / folds))
    # drop extra rows
    fold_index = fold_index[:len(X)]
    # shuffle the folds
    np.random.shuffle(fold_index)

    for i in range(folds):
        X_train = X[fold_index != i]
        y_train = y[fold_index != i]
        X_test = X[fold_index == i]
        y_test = y[fold_index == i]

        yhat = model.fit(X_train, y_train).predict(X_test)
        res.append(loss(yhat, y_test))

    return res
