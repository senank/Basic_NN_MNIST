import numpy as np
import pandas as pd
import matplotlib as plt
import sys

### Helper functions for printing ###
def get_predication(A2):
    return np.argmax(A2, 0)

def get_accuracy(pred, Y):
    print(pred, Y)
    return np.sum(pred == Y) / Y.size


# Init of paramaters for NN
def init_params():
    w1 = np.random.rand(20, 784) - 0.5 # goes from 784 nodes to 10 nodes
    w2 = np.random.rand(10, 20) - 0.5 # goes from 10 nodes to 10 nodes
    b1 = np.random.rand(20, 1) - 0.5 
    b2 = np.random.rand(10, 1) - 0.5
    iterations = 10000
    return w1, w2, b1, b2, iterations

# Activation function and derivative hard-coded 
def ReLU(Z1):
    return np.maximum(0, Z1)

def ReLU_deriv(Z1):
    return Z1 > 0

def batch_norm(Z, gamma, beta):
    # batch norm form
    eps = 1e-5
    mean = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    Z_norm = (Z - mean) / np.sqrt(var + eps)
    
    out = gamma * Z_norm + beta
    active_out = max(0, out)

    #Storing variables for back prop
    cache = (Z_norm, mean, var, eps)

    return active_out, cache

# Dropout function
def dropout(A, dropout_rate):
    # Create a mask to determine which neurons are "dropped out"
    keep_p = 1 - dropout_rate
    mask = np.random.binomial(1, keep_p, size=A.shape) / keep_p
    return A * mask

# Forcing values to be between 0-1
def softmax(Z):
    shiftZ = Z - np.max(Z, axis=0, keepdims=True)  # shift values for numerical stability
    exps = np.exp(shiftZ)
    return exps/ sum(exps) # ? Why does using np.sum here cause error forever?

# Forward_prop, i.e. input->predictions
def forward_prop_train(w1, w2, b1, b2, X, gamma, beta):
    Z1 = w1.dot(X) + b1 # Note can remove + b1 for batch normalization since it was naturally disappear
    
    a_out, cache = batch_norm(Z1, gamma, beta)
    Z_norm, mean, var, eps = cache
    final_cache = (Z_norm, mean, var, gamma, beta, eps)

    A1 = ReLU(Z1)

    dropout_rate = 0.2
    A1_dropout = dropout(A1, dropout_rate)

    Z2 = w2.dot(A1_dropout) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2, final_cache

# Need testing version as dropout reduces capacity of the network, we want full capacity during testing
def forward_prop_test(w1, w2, b1, b2, X):
    Z1 = w1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = w2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# Generating actual result matrix 
def Y_actual(Y):
    m = Y.size # i.e. number of examples/test cases/numbers,
    arr = np.zeros((m, Y.max() + 1)) # Y.max = number of classifications i.e. 9 so +1 for 10 rows
    arr[np.arange(m), Y] = 1    # ? Why does indexing like normal here lead to error
    arr = arr.T
    return arr

# Finding gradients of weights and biases for adjustments
def back_prop(Z1, A1, Z2, A2, w2, Y, X, cache):
    Z_norm, mean, var, gamma, beta, eps = cache
    m = Y.size
    y_actual = Y_actual(Y)
    dZ2 = A2 - y_actual
    dw2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * sum(dZ2)
    dZ1 = w2.T.dot(dZ2) * ReLU_deriv(Z1)
    dw1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * sum(dZ1) # ? But I can use sum/np.sum here
    return dw1, db1, dw2, db2


# Adjusting the parameters after calculating gradients
def adjust_params(w1, w2, b1, b2, dw1, dw2, db1, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2

# Iteratively updating the paramaters in relation to the cost function
def gradient_descent(X, Y, alpha, gamma, beta):
    '''
    Parameters:
        X:
        Y:
        iterations: max number of iterations
        alpha: learning rate
        test: 0 or 1 for testing
    '''
    w1, w2, b1, b2, iterations = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2, cache = forward_prop_train(w1, w2, b1, b2, X, gamma, beta)
        dw1, db1, dw2, db2 = back_prop(Z1, A1, Z2, A2, w2, Y, X, cache)
        w1, b1, w2, b2 = adjust_params(w1, w2, b1, b2, dw1, dw2, db1, db2, alpha)
        if (i % 10 == 0):
            print("Iteration: {}".format(i))
            accuracy = get_accuracy(get_predication(A2), Y)
            print("Accuracy: {}".format(accuracy))
                
    return w1, w2, b1, b2


### FOR TESTING PURPOSES ###
def make_pred(X, w1, b1, w2, b2):
    A2 = forward_prop_test(w1, w2, b1, b2, X)
    return get_predication(A2)


def test_pred(index, w1, b1, w2, b2, X, Y):
    cur_image = X[:, index, None]
    pred = make_pred(X[:, index, None], w1, b1, w2, b2)
    label = Y[index]
    print("Prediction: {}".format(pred))
    print("Label: ".format(label))
    cur_image = cur_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(cur_image, interpolation='nearest')
    plt.show


# Extract data
def extract_dev_data(data):
    data_dev = data[0:1000].T # Transpose it so examples are now each column
    Y_dev = data_dev[0] # First row of the new transposed matrix is the Y values
    X_dev = data_dev[1:n] # Everything after will contain the pixels 
    X_dev = X_dev/255 # Making all the pixel values between 0-1

    return X_dev, Y_dev

def extract_train_data(data):
    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train/255

    return X_train, Y_train

if __name__ == "__main__":
    if len(sys.argv) == 3:
        filenames = []
        outfile = None
        filename = str(sys.argv[1])
        alpha = float(sys.argv[2])
        data = pd.read_csv(filename)
        data = np.array(data)

        # Getting rows and column sizes of the training data
        m, n = data.shape # m = # of rows, n = number of columns
                        # Rows in this case is an example of a number
                        # Columns are the pixel value 0-255 that collectively represent the image

        # Randomizing data
        np.random.shuffle(data)

        # Extracting cross-validation (dev) set from data
        X_dev, Y_dev = extract_dev_data(data)
        
        # Extracting training data from overall data
        X_train, Y_train = extract_train_data(data)
        # print("", Y_train)
        

        # init gamma and beta for batch norm
        gamma = 1
        beta = 0

        # Iteratively updates the w1, w2, b1, b2 params with gradient descent
        w1, w2, b1, b2 = gradient_descent(X_train, Y_train, alpha, gamma, beta)
        # print("Error extracting/tokenizing data, check file")
        # sys.exit

        # Testing
        # this function tests singular prediction of the training data to see which type of number is struggled with
        # test_pred(5, w1, b1, w1, b2, X_train, X_dev)
        
        # Cross validating with untrained set
        # test_accuracy = get_accuracy(make_pred(X_dev, w1, b1, w2, b2))
    else:
        print("USAGE: python numb_prob_numpy.py [FILENAME] [ALPHA]")