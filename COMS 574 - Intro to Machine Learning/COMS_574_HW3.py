## My decision tree is arround is 33 to 38% consistent with Scikit-learn's result
## I did the sorting with 0 axis in the you rock function. Otherwise I was getting 0% match 
## However, my hand calcutation matches with the output. 
import operator
import numpy, sklearn, sklearn.tree

def estimate_gini_impurity(feature_values, threshold, labels, polarity):
    """Compute the gini impurity for comparing a feature value against a threshold under a given polarity
    feature_values: 1D numpy array, feature_values for samples on one feature dimension
    threshold: float
    labels: 1D numpy array, the label of samples, only +1 and -1.
    polarity: operator type, only operator.gt or operator.le are allowed
    Examples
    -------------
    >>> feature_values = numpy.array([1,2,3,4,5,6,7,8])
    >>> labels = numpy.array([+1,+1,+1,+1, -1,-1,-1,-1])
    >>> for threshold in range(0,8):
    ...     print("%.5f" % estimate_gini_impurity(feature_values, threshold, labels, operator.gt))
    0.50000
    0.48980
    0.44444
    0.32000
    0.00000
    0.00000
    0.00000
    0.00000
    >>> for threshold in range(0,8):
    ...     print("%.5f" % estimate_gini_impurity(feature_values, threshold, labels, operator.le))
    1.00000
    0.00000
    0.00000
    0.00000
    0.00000
    0.32000
    0.44444
    0.48980
    """

    # Problem 5 estimating Gini Impurity
    if (polarity == operator.gt):
        count = numpy.finfo(float).eps
        count1 = numpy.finfo(float).eps
        count2 = numpy.finfo(float).eps

        for i in range(len(feature_values)):
            if (feature_values[i] > threshold):
                count = count + 1
                if (labels[i] == +1):
                    count1 = count1 + 1
                else:
                    count2 = count2 + 1

    elif (polarity == operator.le):
        count = numpy.finfo(float).eps
        count1 = numpy.finfo(float).eps
        count2 = numpy.finfo(float).eps

        for i in range(len(feature_values)):
            if (feature_values[i] <= threshold):
                count = count + 1
                if (labels[i] == +1):
                    count1 = count1 + 1
                else:
                    count2 = count2 + 1

    prob1 = count1 / count
    prob2 = count2 / count

    gini_impurity = abs(1 - pow(prob1, 2) - pow(prob2, 2))

    return gini_impurity


# In[3]:



feature_values = numpy.array([1, 2, 3, 4, 5, 6, 7, 8])
labels = numpy.array([+1, +1, +1, +1, -1, -1, -1, -1])
print(labels)
print(feature_values)
for threshold in range(0, 8):
    print("%.5f" % estimate_gini_impurity(feature_values, threshold, labels, operator.gt))

for threshold in range(0, 8):
    print("%.5f" % estimate_gini_impurity(feature_values, threshold, labels, operator.le))


def estimate_gini_impurity_expectation(feature_values, threshold, labels):
    """Compute the expectation of gini impurity given the feature values on one  feature dimension and a threshold
    feature_values: 1D numpy array, feature_values for samples on one feature dimension
    threshold: float
    labels: 1D numpy array, the label of samples, only +1 and -1.
    Examples
    ---------------
    >>> feature_values = numpy.array([1,2,3,4,5,6,7,8])
    >>> labels = numpy.array([+1,+1,+1,+1, -1,-1,-1,-1])
    >>> for threshold in range(0,9):
    ...     print("%.5f" % estimate_gini_impurity_expectation(feature_values, threshold, labels))
    0.50000
    0.42857
    0.33333
    0.20000
    0.00000
    0.20000
    0.33333
    0.42857
    0.50000
    """
    # Problem 6: estimate_gini_impurity_expectation
    count1 = numpy.finfo(float).eps
    count2 = numpy.finfo(float).eps

    for i in range(len(feature_values)):
        if (feature_values[i] > threshold):
            count1 = count1 + 1
        elif (feature_values[i] <= threshold):
            count2 = count2 + 1
    prob1 = count1 / len(feature_values)
    prob2 = count2 / len(feature_values)
    gini1 = estimate_gini_impurity(feature_values, threshold, labels, operator.gt)
    gini2 = estimate_gini_impurity(feature_values, threshold, labels, operator.le)
    expectation = (prob1 * gini1) + (prob2 * gini2)
    return expectation




def midpoint(x):
    """Given a sequqence of numbers, return the middle points between every two consecutive ones.
    >>> x= numpy.array([1,2,3,4,5])
    >>> (x[1:] + x[:-1]) / 2
    array([1.5, 2.5, 3.5, 4.5])
    """
    return (x[1:] + x[:-1]) / 2




feature_values = numpy.array([1, 2, 3, 4, 5, 6, 7, 8])
labels = numpy.array([+1, +1, +1, +1, -1, -1, -1, -1])
for threshold in range(0, 9):
    print("%.5f" % estimate_gini_impurity_expectation(feature_values, threshold, labels))




def grid_search_split_midpoint(X, y):
    """Given a dataset, compute the gini impurity expectation for all pairs of features and thresholds.
    Inputs
    ----------
        X: 2-D numpy array, axis 0 or row is a sample, and axis 1 or column is a feature
        y: 1-D numpy array, the labels, +1 or -1
    Returns
    ---------
        grid: 2-D numpy array, axis 0 or row is a threshold, and axis 1 or column is a feature
    Examples
    -------------
    >>> numpy.random.seed(1) # fix random number generation starting point
    >>> X = numpy.random.randint(1, 10, (8,3)) # generate training samples
    >>> y = numpy.array([+1,+1,+1,+1, -1,-1,-1,-1])
    >>> grid, feature_id, bts = grid_search_split_midpoint(X, y)
    >>> numpy.set_printoptions(precision=5)
    >>> print (grid)
    [[0.42857 0.5     0.46667]
     [0.46667 0.5     0.46667]
     [0.46667 0.46667 0.46667]
     [0.375   0.5     0.46667]
     [0.5     0.5     0.46667]
     [0.5     0.5     0.5    ]
     [0.5     0.42857 0.42857]]
    >>> clf = sklearn.tree.DecisionTreeClassifier(max_depth=1)
    >>> clf = clf.fit(X,y)
    >>> print (clf.tree_.feature[0], clf.tree_.threshold[0], feature_id, bts)
    0 7.0 0 7.0
    >>> print(clf.tree_.feature[0] == feature_id)
    True
    >>> print( clf.tree_.threshold[0] == bts)
    True
    >>> # Antoher test case
    >>> numpy.random.seed(2) # fix random number generation starting point
    >>> X = numpy.random.randint(1, 30, (8,3)) # generate training samples
    >>> grid, feature_id, bts = grid_search_split_midpoint(X, y)
    >>> numpy.set_printoptions(precision=5)
    >>> print (grid)
    [[0.42857 0.42857 0.42857]
     [0.5     0.5     0.33333]
     [0.375   0.46667 0.46667]
     [0.375   0.5     0.5    ]
     [0.46667 0.46667 0.46667]
     [0.33333 0.5     0.5    ]
     [0.42857 0.42857 0.42857]]
    >>> clf = clf.fit(X,y) # return the sklearn DT
    >>> print (clf.tree_.feature[0], clf.tree_.threshold[0], feature_id, bts)
    2 8.5 2 8.5
    >>> print(clf.tree_.feature[0] == feature_id)
    True
    >>> print( clf.tree_.threshold[0] == bts)
    True
    >>> # yet antoher test case
    >>> numpy.random.seed(4) # fix random number generation starting point
    >>> X = numpy.random.randint(1, 100, (8,3)) # generate training samples
    >>> grid, feature_id, bts = grid_search_split_midpoint(X, y)
    >>> numpy.set_printoptions(precision=5)
    >>> print (grid)
    [[0.42857 0.42857 0.42857]
     [0.5     0.5     0.33333]
     [0.46667 0.46667 0.375  ]
     [0.375   0.375   0.375  ]
     [0.46667 0.2     0.46667]
     [0.5     0.42857 0.5    ]
     [0.42857 0.42857 0.42857]]
    >>> clf = clf.fit(X,y) # return the sklearn DT
    >>> print (clf.tree_.feature[0], clf.tree_.threshold[0], feature_id, bts)
    1 47.5 1 47.5
    >>> print(clf.tree_.feature[0] == feature_id)
    True
    >>> print( clf.tree_.threshold[0] == bts)
    True
    """

    X_sorted = numpy.sort(X, axis=0)
    thresholds = numpy.apply_along_axis(midpoint, 0, X_sorted)

    grid = numpy.zeros(shape=(len(thresholds), X_sorted.shape[1]))
    numpy.set_printoptions(precision=5)
    best_feature = 0
    best_threshold = 0

    print(X_sorted)
    print(thresholds)

    min_expect=11110.0
    # Problem 7: The Split of a node 
    for i in range(0, len(thresholds)):
        gini_impurity_1 = estimate_gini_impurity_expectation(X_sorted[:, 0], thresholds[i, 0], y)
        gini_impurity_2 = estimate_gini_impurity_expectation(X_sorted[:, 1], thresholds[i, 1], y)
        gini_impurity_3 = estimate_gini_impurity_expectation(X_sorted[:, 2], thresholds[i, 2], y)

        if min_expect> gini_impurity_1:
            best_threshold=thresholds[i, 0]
            best_feature=0
            min_expect=gini_impurity_1
        if min_expect> gini_impurity_2:
            best_threshold=thresholds[i, 1]
            best_feature = 1
            min_expect=gini_impurity_2
        if min_expect> gini_impurity_3:
            best_threshold=thresholds[i, 2]
            best_feature = 2
            min_expect=gini_impurity_3

        grid[i, 0] = gini_impurity_1
        grid[i, 1] = gini_impurity_2
        grid[i, 2] = gini_impurity_3

    # (best_threshold_index, best_feature_index) = numpy.unravel_index(numpy.argmin(grid, axis=None), grid.shape)
    # best_feature = best_feature_index
    # best_threshold = grid[best_threshold_index][best_feature_index]

    return grid, best_feature, best_threshold


def you_rock(N, R, d):
    """
    N: int, number of samples, e.g., 1000.
    R: int, maximum feature value, e.g., 100.
    d: int, number of features, e.g., 3.
    """
    numpy.random.seed()  # re-random the seed
    hits = 0
    for _ in range(N):
        X = numpy.random.randint(1, R, (8, d))  # generate training samples
        y = numpy.array([+1, +1, +1, +1, -1, -1, -1, -1])
        X = numpy.sort(X, axis=0)
        _, feature_id, bts = grid_search_split_midpoint(X, y)
        clf = sklearn.tree.DecisionTreeClassifier(max_depth=1)
        clf = clf.fit(X, y)

        if clf.tree_.feature[0] == feature_id and clf.tree_.threshold[0] == bts:
            hits += 1
    print("your Decision tree is {:2.2%} consistent with Scikit-learn's result.".format(hits / N))



if __name__ == "__main__":
    import doctest

    doctest.testmod()
    you_rock(1000, 100, 3)





