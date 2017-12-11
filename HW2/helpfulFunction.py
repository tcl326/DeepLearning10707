import numpy as np
import matplotlib.pyplot as plt
import glob, os

def plotCrossEntropyError(trainingError, validationError, name="CrossEntropy"):
    X = [i for i in range(len(trainingError))]
    # fig1 = plt.figure()
    training, = plt.plot(X, trainingError, '-', label = "Average Training Cross-Entropy Error")
    validation, = plt.plot(X, validationError, '--', label = "Average Validation Cross-Entropy Error")
    plt.ylabel('Cross-Entropy Error')
    plt.xlabel('Epochs')
    plt.legend()
    os.chdir("output/")
    n = 0
    for files in glob.glob("*.jpg"):
            n = n + 1;
    os.chdir('..')
    file_name = "output/" + name + str(n) + ".jpg"
    plt.savefig(file_name)
    plt.close()
    # plt.show()

def plotClassificationError(trainingError, validationError, name="MeanClassification"):
    X = [i for i in range(len(trainingError))]
    training, = plt.plot(X, trainingError, '-', label = "Mean Training Classification Error")
    validation, = plt.plot(X, validationError, '--', label = "Mean Validation Classification Error")
    plt.ylabel('Mean Classification Error')
    plt.xlabel('Epochs')
    plt.legend()
    os.chdir("output/")
    n = 0
    for files in glob.glob("*.jpg"):
            n = n + 1;
    os.chdir('..')
    file_name = "output/" + name + str(n) + ".jpg"
    plt.savefig(file_name)
    plt.close()

def visualizeParameter(W, name="W"):
    fig, axis = plt.subplots(nrows=10, ncols=10)
    # plt.style.use('grayscale')
    for col in range(10):
        for row in range(10):
            image = np.reshape(W[col*10+row],(28,28))
            axis[row, col].imshow(image, cmap='gray')
            axis[row, col].set_axis_off();
            plt.subplots_adjust(hspace=0.3)
    os.chdir("output/")
    n = 0
    for files in glob.glob("*.jpg"):
            n = n + 1;
    os.chdir('..')
    file_name = "output/" + name + str(n) + ".jpg"
    plt.savefig(file_name)

def visualize_parameter(W, shape=(28,28), name="W"):
    # assuming your filters are stored in a list called all_filters
    all_filter_image = np.zeros((10*28, 10*28))
    for filter_num in range(W.shape[0]):
        start_x = shape[0] * (filter_num % int(np.sqrt(W.shape[0])))
        start_y = shape[1] * (filter_num / int(np.sqrt(W.shape[0])))
        all_filter_image[start_x:start_x + shape[0], start_y: start_y + shape[1]] = W[filter_num].reshape(shape[0],shape[1])
    plt.axis('off')
    plt.imshow(all_filter_image, cmap='gray')
    os.chdir("output/")
    n = 0
    for files in glob.glob("*.jpg"):
            n = n + 1;
    os.chdir('..')
    file_name = "output/" + name + str(n) + ".jpg"
    plt.savefig(file_name)
    plt.close()

def shuffleXYData(a, b):
    seed = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(seed)
    np.random.shuffle(b)
    return a, b

def shuffleXData(a):
    np.random.shuffle(a)
    return a

def meanClassificationError(FX, Y):
    maxInd = np.argmax(FX, axis = 0)
    comp = np.equal(maxInd, Y)
    error = 1 - sum(comp)/float(len(comp))
    return error

def crossEntropy(FX, Y):
    eY = indicator(Y, 10)
    m = len(Y)
    crossEntropy = -1 * np.sum(eY * np.log(FX)) / float(m)
    return crossEntropy

def softmax(A):
    # print A
    expA = np.exp(A)
    softmax = expA/(np.sum(expA, axis = 0))
    assert softmax.shape == A.shape
    return softmax

def sigmoid(A):
    return 1./(1+np.exp(-1*A))

def preActivation(b, W, X):
    # assert W.shape == np.transpose(X).shape
    A = b + np.dot(W, X)
    # assert A.shape == b.shape
    return A

def hiddenLayerActivation(A, activationName):
    return activation(A, activationName)

def outputActivation(outputPreActivation):
    return softmax(outputPreActivation)

def tanh(A):
    return np.tanh(A)

def relu(A):
    return A * (A > 0)

def indicator(Y, C):
    indicator = np.zeros((C, Y.shape[0]))
    # print indicator.shape
    indicator[Y.astype(int), np.arange(Y.shape[0])] = 1
    return indicator

def softplus(A):
    return np.log(1 + np.exp(A))

def activation(input, name):
    if name == "sigmoid":
        return sigmoid(input)
    elif name == "tanh":
        return tanh(input)
    elif name == "relu":
        return relu(input)
    else:
        return "INVALID ACTIVATION"

def actDerivative(input, name):
    if name == "sigmoid":
        return input * (1 - input)
    elif name == "tanh":
        return 1 - np.power(input, 2)
    elif name == "relu":
        return 1. * (input > 0)
    else:
        return "INVALID ACTIVATION"

def save_param(param, param_name="output"):
    np.savetxt(param_name + ".txt", param, delimiter=',')

def get_param_from_text(file_name):
    data = np.loadtxt(file_name, dtype='float64', delimiter=",")
    return data
