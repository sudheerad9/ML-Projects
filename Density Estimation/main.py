import numpy as np
import math
import scipy.io
# import matplotlib.pyplot as plt


def load_data():
    """ to load data"""
    return scipy.io.loadmat('fashion_mnist.mat')


def get_mean(a, axis=None):
    """ Calculates mean for the data """
    return np.mean(a, axis=axis)


def get_std(a, axis=None):
    """ Calculates standard deviation for the data"""
    return np.std(a, axis=axis)


def sigmoid_function(z):
    """ Function to compute predicted values for logistic regression"""
    return 1.0 / (1 + np.exp(-z))


def get_class_data(a, b, x):
    """ Seperate data based on class values"""
    c = np.where(b == x)[0]
    d = np.take(a, c)
    return d


def calculate_probability(x, std, mean):
    """ 1D Density function for calculating the probability"""
    return (1 / (std * pow(2 * math.pi, 0.5))) * math.exp(-1 / 2 * ((x - mean) ** 2) * (1 / std ** 2))


def get_features_test_data(data):
    """ Extract features for test data"""
    test_data = np.array(data['tsX'])
    test_labels = np.array(data['tsY'][0])
    test_data_mean = get_mean(test_data, axis=1)
    test_data_std = get_std(test_data, axis=1)
    test_data_dict = {
        "test_labels": test_labels,
        "test_data_mean": test_data_mean,
        "test_data_std": test_data_std
    }
    return test_data_dict


def get_features_train_data(data):
    """ Extract features for train data"""
    train_data = np.array(data['trX'])
    train_labels = np.array(data['trY'][0])
    train_data_mean = get_mean(train_data, axis=1)
    train_data_std = get_std(train_data, axis=1)
    train_data_dict = {
        "train_labels": train_labels,
        "train_data_mean": train_data_mean,
        "train_data_std": train_data_std
    }
    return train_data_dict


def get_separated_features_by_class(data):
    """ Get seperated features by class 0 and class 1"""
    train_data_mean_class_0 = get_class_data(data["train_data_mean"], data["train_labels"], 0)
    train_data_mean_class_1 = get_class_data(data["train_data_mean"], data["train_labels"], 1)
    train_data_std_class_0 = get_class_data(data["train_data_std"], data["train_labels"], 0)
    train_data_std_class_1 = get_class_data(data["train_data_std"], data["train_labels"], 1)
    train_data_by_class = {
        "train_data_mean_class_0": train_data_mean_class_0,
        "train_data_mean_class_1": train_data_mean_class_1,
        "train_data_std_class_0": train_data_std_class_0,
        "train_data_std_class_1": train_data_std_class_1
    }
    return train_data_by_class


def estimate_parameters(train_features):
    """ Estimate parameters Mean and Standard deviation for class 0 and class 1"""
    fea1_mean_class_0 = get_mean(train_features["train_data_mean_class_0"])
    fea2_mean_class_0 = get_mean(train_features["train_data_std_class_0"])
    fea1_std_class_0 = get_std(train_features["train_data_mean_class_0"])
    fea2_std_class_0 = get_std(train_features["train_data_std_class_0"])

    fea1_mean_class_1 = get_mean(train_features["train_data_mean_class_1"])
    fea2_mean_class_1 = get_mean(train_features["train_data_std_class_1"])
    fea1_std_class_1 = get_std(train_features["train_data_mean_class_1"])
    fea2_std_class_1 = get_std(train_features["train_data_std_class_1"])
    print("Mean for class 0 feature 1 is {}".format(fea1_mean_class_0))
    print("Mean for class 0 feature 2 is {}".format(fea2_mean_class_0))
    print("Standard Deviation for class 0 feature 1 is {}".format(fea1_std_class_0))
    print("Standard Deviation for class 0 feature 2 is {}".format(fea2_std_class_0))
    print("Mean for class 1 feature 1 is {}".format(fea1_mean_class_1))
    print("Mean for class 1 feature 2 is {}".format(fea2_mean_class_1))
    print("Standard Deviation for class 1 feature 1 is {}".format(fea1_std_class_1))
    print("Standard Deviation for class 1 feature 2 is {}".format(fea2_std_class_1))
    class_0_parameters = {
        "fea1_mean_class_0": fea1_mean_class_0,
        "fea2_mean_class_0": fea2_mean_class_0,
        "fea1_std_class_0": fea1_std_class_0,
        "fea2_std_class_0": fea2_std_class_0
    }
    class_1_parameters = {
        "fea1_mean_class_1": fea1_mean_class_1,
        "fea2_mean_class_1": fea2_mean_class_1,
        "fea1_std_class_1": fea1_std_class_1,
        "fea2_std_class_1": fea2_std_class_1
    }
    return class_0_parameters, class_1_parameters


def implement_naive_bayes(class_0_prms, class_1_prms, features_test):
    count = 0
    # The value of prior is 0.5 since the number of samples in both the classes is same.
    prior = 0.5
    count_0 = 0
    count_1 = 0
    for i in range(len(features_test["test_data_mean"])):
        f_1_p_0 = calculate_probability(features_test["test_data_mean"][i],
                                        class_0_prms["fea1_std_class_0"], class_0_prms["fea1_mean_class_0"])
        f_2_p_0 = calculate_probability(features_test["test_data_std"][i],
                                        class_0_prms["fea2_std_class_0"], class_0_prms["fea2_mean_class_0"])
        # Probability of data belonging to class 0
        class_0 = f_1_p_0 * f_2_p_0 * prior
        f_1_p_1 = calculate_probability(features_test["test_data_mean"][i],
                                        class_1_prms["fea1_std_class_1"], class_1_prms["fea1_mean_class_1"])
        f_2_p_1 = calculate_probability(features_test["test_data_std"][i],
                                        class_1_prms["fea2_std_class_1"], class_1_prms["fea2_mean_class_1"])
        # Probability of data belonging to class 1
        class_1 = f_1_p_1 * f_2_p_1 * prior
        # Based on probability classify data to class 0 and class 1
        if class_0 > class_1:
            class_predicted = 0
        else:
            class_predicted = 1
        # compute accuracy for classification
        if class_predicted == features_test["test_labels"][i]:
            count = count + 1
            if class_predicted == 0:
                count_0 = count_0 + 1
            if class_predicted == 1:
                count_1 = count_1 + 1
    print("The combined accuracy for prediction is {}".format((count / 2000) * 100))
    print("The accuracy for predicting class 0 is {}".format((count_0 / 1000) * 100))
    print("The accuracy for predicting class 1 is {}".format((count_1 / 1000) * 100))


def compute_log_likelihood(y, y_p):
    # To prevent log(0) to be not defined
    eps = 1e-8
    # Modify the predicted values so that log is not undefined.
    y_p = np.maximum(np.full(y_p.shape, eps), np.minimum(np.full(y_p.shape, 1 - eps), y_p))
    return np.sum(y * np.log(y_p) + (1 - y) * np.log(1 - y_p))


def implement_logistic_regression(features_train, learning_rate, epochs):
    train_data_features = np.array([features_train["train_data_mean"], features_train["train_data_std"]]).transpose()
    # modify data to handle bias
    data_for_bias_train = np.ones(features_train["train_data_mean"].shape).reshape(12000, 1)
    # create train data for logistic regression
    train_data = np.concatenate((data_for_bias_train, train_data_features), 1)
    weights = np.array([0, 0, 0])
    log_likelihood = []
    weights_array = []
    l_r = learning_rate
    epochs = epochs
    # run the algorithm for 1000 iterations
    for i in range(epochs):
        z = np.dot(train_data, weights)
        # compute predicted values
        y_p = sigmoid_function(z)
        # compute log likelihood function
        log_likelihood.append(compute_log_likelihood(features_train["train_labels"], y_p))
        # update weights using gradient ascent algorithm
        weights = weights + l_r * np.dot((train_data.T), (features_train["train_labels"] - y_p))
        weights_array.append(weights)
    return weights_array, log_likelihood


def compute_accuracy_for_logistic_regression(features_test, weights):
    # extract test features.
    test_data_features = np.array([features_test["test_data_mean"], features_test["test_data_std"]]).transpose()
    # modify data to handle bias
    data_for_bias_test = np.ones(features_test["test_data_mean"].shape).reshape(
        features_test["test_data_mean"].shape[0], 1)
    test_data = np.concatenate((data_for_bias_test, test_data_features), 1)
    count_0 = 0
    count_1 = 0
    count = 0
    for i in range(len(features_test["test_data_mean"])):
        z = np.dot(weights[240], test_data[i])
        # compute predicted values for test data
        y_p = sigmoid_function(z)
        # classify data based on threshold.
        if y_p > 0.5:
            y_p = 1
        else:
            y_p = 0
        # compute accuracy of classification
        if y_p == features_test["test_labels"][i]:
            count = count + 1
            if y_p == 0:
                count_0 = count_0 + 1
            else:
                count_1 = count_1 + 1
    print("The combined accuracy for prediction is {}".format((count / 2000) * 100))
    print("The accuracy for predicting class 0 is {}".format((count_0 / 1000) * 100))
    print("The accuracy for predicting class 1 is {}".format((count_1 / 1000) * 100))


if __name__ == '__main__':
    data = load_data()
    learning_rate = 0.01
    iterations = 1000
    # extract train and test features
    features_train = get_features_train_data(data)
    features_test = get_features_test_data(data)
    # extract features by class
    features_separated = get_separated_features_by_class(features_train)
    print("Estimated Parameters\n")
    class_0_prms, class_1_prms = estimate_parameters(features_separated)
    print("\nAccuracy for Naive Bayes\n")
    implement_naive_bayes(class_0_prms, class_1_prms, features_test)
    weights, log_likelihood = implement_logistic_regression(features_train, learning_rate, iterations)
    #To draw a plot between loglikelihood and number of iterations.
    """
    plt.plot(range(len(log_likelihood)), log_likelihood)
    plt.title('Loglikelihood Vs Iterations')
    plt.xlabel('iterations')
    plt.ylabel('likelihood')
    plt.show()
    """
    print("\nAccuracy for Logistic Regression\n")
    compute_accuracy_for_logistic_regression(features_test, weights)
