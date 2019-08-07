
import numpy as np
import matplotlib.pyplot as plt
from load_files import load_dataset

train_data_x,test_data_x,train_output_y,test_output_y = load_dataset()

test_output_y[test_output_y == 'Iris-setosa'] = 1
test_output_y[test_output_y != 1] = 0
train_output_y[train_output_y == 'Iris-setosa'] = 1
train_output_y[train_output_y != 1] = 0
print ("Number of training examples: m_train = " + str(train_data_x.shape[0]))
print ("Number of testing examples: m_test = " + str(test_data_x.shape[0]))
print ("train_set_x shape: " + str(train_data_x.shape))
print ("train_set_y shape: " + str(train_output_y.shape))
print ("test_set_x shape: " + str(test_data_x.shape))
print ("test_set_y shape: " + str(test_output_y.shape))

train_data_x = train_data_x.reshape(train_data_x.shape[0] ,-1).T
test_data_x = test_data_x.reshape(test_data_x.shape[0] ,-1).T


def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    s = 1/ (1 + np.e ** -z)
    ### END CODE HERE ###

    return s


# In[40]:


print("sigmoid([0, 2]) = " + str(sigmoid(np.array([0, 2]))))


def initialize_with_zeros(dim):
    ### START CODE HERE ### (≈ 1 line of code)
    w = np.zeros((dim, 1))
    b = 0
    ### END CODE HERE ###

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


# In[42]:


dim = 2
w, b = initialize_with_zeros(dim)
print("w = " + str(w))
print("b = " + str(b))


def propagate(w, b, X, Y):
    m = X.shape[1]
    # FORWARD PROPAGATION (FROM X TO COST)
    ### START CODE HERE ### (≈ 2 lines of code)
    print(X)
    print(Y)
    A = sigmoid(np.dot(w.T, X) + b)  # compute activation
    print(A)
    cost = np.sum((Y * np.log(A) + (1 - Y) * np.log(1 - A))) / -m  # compute cost
    ### END CODE HERE ###

    # BACKWARD PROPAGATION (TO FIND GRAD)
    ### START CODE HERE ### (≈ 2 lines of code)
    dw = (np.dot(X, (A - Y).T)) / m
    db = (np.sum(A - Y)) / m
    ### END CODE HERE ###

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost


# In[44]:


w, b, X, Y = np.array([[1.], [2.]]), 2., np.array([[1., 2., -1.], [3., 4., -3.2]]), np.array([[1, 0, 1]])
grads, cost = propagate(w, b, X, Y)
print("dw = " + str(grads["dw"]))
print("db = " + str(grads["db"]))
print("cost = " + str(cost))


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    for i in range(num_iterations):

        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ###
        grads, cost = propagate(w, b, X, Y)
        ### END CODE HERE ###

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w = w - (learning_rate) * dw
        b = b - (learning_rate) * db
        ### END CODE HERE ###

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


# In[46]:


params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)

print("w = " + str(params["w"]))
print("b = " + str(params["b"]))
print("dw = " + str(grads["dw"]))
print("db = " + str(grads["db"]))


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    ### START CODE HERE ### (≈ 1 line of code)
    A = sigmoid(np.dot(w.T, X) + b)

    ### END CODE HERE ###
    for i in range(A.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        ### START CODE HERE ### (≈ 4 lines of code)
        if ((A[0])[i]) > 0.5:
            Y_prediction[0][i] = 1
        else:
            Y_prediction[0][i] = 0

            ### END CODE HERE ###

    assert (Y_prediction.shape == (1, m))

    return Y_prediction


# In[48]:


w = np.array([[0.1124579], [0.23106775]])
b = -0.3
X = np.array([[1., -1.1, -3.2], [1.2, 2., 0.1]])
print("predictions = " + str(predict(w, b, X)))


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.005, print_cost=True):
    ### START CODE HERE ###

    # initialize parameters with zeros (≈ 1 line of code)
    w, b = initialize_with_zeros(X_train.shape[0])
    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b,X_test)
    Y_prediction_train = predict(w, b, X_train)

    ### END CODE HERE ###

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test -Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


# Run the following cell to train your model.

# In[53]:


d = model(train_data_x, train_output_y, test_data_x, test_output_y, num_iterations=2000, learning_rate=0.005, print_cost=True)

learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print("learning rate is: " + str(i))
    models[str(i)] = model(train_data_x, test_data_x, test_output_y, train_output_y, num_iterations=1500, learning_rate=i,
                           print_cost=False)
    print('\n' + "-------------------------------------------------------" + '\n')

