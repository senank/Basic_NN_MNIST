import numpy as np
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
    w1 = np.random.rand(10, 784) - 0.5 # goes from 784 nodes to 10 nodes
    w2 = np.random.rand(10, 10) - 0.5 # goes from 10 nodes to 10 nodes
    b1 = np.random.rand(10, 1) - 0.5 
    b2 = np.random.rand(10, 1) - 0.5
    iterations = 100
    return w1, w2, b1, b2, iterations

# Activation function hard-coded and derivative
def ReLU(Z1):
    if Z1 >= 0:
        return Z1
    return 0

def ReLU_deriv(Z1):
    if Z1>0:
        return 1
    return 0

# Forcing values to be between 0-1
def softmax(Z):
    return np.exp(Z)/ np.sum(np.exp(Z))

# Forward_prop, i.e. input->predictions
def forward_prop(w1, w2, b1, b2, X):
    Z1 = w1*X + b1
    A1 = ReLU(Z1)
    Z2 = w2*Z2 + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# Generating actual result matrix 
def Y_actual(Y):
    m = Y.size # i.e. number of examples/test cases/numbers,
    arr = np.zero((m, Y.max + 1)) # Y.max = number of classifications i.e. 9 so +1 for 10 rows
    arr[np.arange(m)][Y] = 1
    arr = arr.T
    return arr

# Finding gradients of weights and biases for adjustments
def back_prop(Z1, A1, Z2, A2, w2, Y, X):
    m = Y.size
    y_actual = Y_actual(Y)
    dZ2 = (A2 - y_actual) # ? Check if this is actually the "cost function" from 3blue1brown's explanation
    # TODO Understand where these derivatives come from after Z2
    dw2 = 1/m * dZ2 * A1.T
    db2 = 1/m * np.sum(dZ2, 2) # ? Why did this change from np.sum(dZ2) to np.sum(dZ2, 2)
    dZ1 = w2.T * dZ2 * ReLU_deriv(Z1)
    dw1 = 1/m * dZ1 * X.T
    db1 = 1/m * np.sum(dZ1, 2)
    return dw1, db1, dw2, db2

# Adjusting the parameters after calculating gradients
def adjust_params(w1, w2, b1, b2, dw1, dw2, db1, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2

# Iteratively updating the paramaters in relation to the cost function
def gradient_descent(X, Y, alpha):
    '''
    Parameters:
        X:
        Y:
        iterations: max number of iterations
        alpha: learning rate
    '''
    w1, w2, b1, b2, iterations = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(w1, w2, b1, b2, X)
        dw1, db1, dw2, db2 = back_prop(Z1, A1, Z2, A2, w2, Y, X)
        w1, b1, w2, b2 = adjust_params(w1, w2, b1, b2, dw1, dw2, db1, db2, alpha)
        if (i % 10 == 0):
            print("Iteration: {}".format(i))
            accuracy = get_accuracy(get_predication(A2), Y)
            print("Accuracy: {}".format(accuracy))
                
    return w1, w2, b1, b2


### FOR TESTING PURPOSES ###
def make_pred(X, w1, b1, w2, b2):
    A2 = forward_prop(w1, w2, b1, b2, X)
    return get_predication(A2)


def test_pred(index, w1, b1, w2, b2, X_train, Y_train):
    cur_image = X_train[:, index, None]
    pred = make_pred(X_train[:, index, None], w1, b1, w2, b2)
    label = Y_train[index]
    print("Prediction: {}".format(pred))
    print("Label: ".format(label))
    cur_image = cur_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(cur_image, interpolation='nearest')
    plt.show


if __name__ == "__main__":
    try:
        filenames = []
        outfile = None
        filename = str(sys.argv[1])
        alpha = float(sys.argv[2])
        with open(filename) as my_file:
            # TODO Construct X and Y from the content file, what is the format of this file? Maybe helper function?
            # X_train = 
            # Y_train = 
            pass
        
        w1, w2, b1, b2 = gradient_descent(X_train, Y_train, alpha) # These datas can be used for testing
        
    except:
        print("USAGE: python numb_prob_numpy.py [FILENAME] [ALPHA]")