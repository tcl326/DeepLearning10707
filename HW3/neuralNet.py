import numpy as np
import matplotlib.pyplot as plt
import loadData
import helpfulFunction as hp
# import pickle

def initializeParameters(layerInfo):
    L = len(layerInfo) # L = 3
    parameters = {}
    for l in range(L-1):
        parameters["b" + str(l+1)] = np.zeros((layerInfo[l+1],1))
        wb = np.sqrt(6)/np.sqrt(layerInfo[l] + layerInfo[l+1])
        parameters["W" + str(l+1)] = 2 * wb * np.random.random_sample((layerInfo[l+1],layerInfo[l])) - wb
    return parameters

def forwardPropagation(X, parameters, activation):
    L = len(parameters) // 2
    cache = {}
    cache["H0"] = np.transpose(X)
    for l in range(L-1):
        cache["A" + str(l+1)] = hp.preActivation(parameters["b" + str(l+1)], parameters["W" + str(l+1)],cache["H" + str(l)])
        cache["H" + str(l+1)] = hp.hiddenLayerActivation(cache["A" + str(l+1)], activation)
    cache["A" + str(L)] = hp.preActivation(parameters["b" + str(L)], parameters["W" + str(L)],cache["H" + str(L-1)])
    cache["H" + str(L)] = hp.outputActivation(cache["A" + str(L)])
    return cache

def backpropagation(parameters, cache, X, Y, lambd = 0, activation = "sigmoid"):
    eY = hp.indicator(Y, 10)
    m = X.shape[0]
    L = len(parameters)//2
    gradients = {}
    assert eY.shape == cache["A" + str(L)].shape
    gradients["dH" + str(L)] = - eY/(cache["H" + str(L)])
    gradients["dA" + str(L)] = -1 * (eY - cache["H" + str(L)])
    # print cache["H" + str(L)].shape
    for l in range(L, 0, -1):
        # print l
        gradients["dW" + str(l)] = np.dot(gradients["dA" + str(l)], np.transpose(cache["H" + str(l-1)])) / float(m) + 2 * lambd * parameters["W" + str(l)]
        gradients["db" + str(l)] = np.sum(gradients["dA" + str(l)], axis = 1, keepdims = True) /float(m)
        gradients["dH" + str(l-1)] = np.dot(np.transpose(parameters["W" + str(l)]), gradients["dA" + str(l)])
        if l == 1:
            gradients["dA" + str(l-1)] = gradients["dH" + str(l-1)]
        else:
            gradients["dA" + str(l-1)] = gradients["dH" + str(l-1)] * hp.actDerivative(cache["H" + str(l-1)], activation)
    return gradients


def updateParameters(parameters, gradients, learningRate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - gradients["dW" + str(l+1)] * learningRate
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - gradients["db" + str(l+1)] * learningRate
    return parameters

def addMomentum(prevGrads, grads, momentum):
    if momentum == 0 or len(prevGrads) == 0:
        return grads
    # print momentum
    for key, value in grads.iteritems():
        # print "curr"
        # print value
        value = value + momentum * prevGrads[key]
        # print "Prev"
        # print prevGrads[key]
        # print "future"
        # print value
    return grads

def train(X, Y, validX, validY, layerInfo, numIterations, parameters=None, learningRate = 0.1, printCost = False, momentum = 0, lambd = 0, activation = "sigmoid"):
    errorTrain = []
    errorValid = []
    meanError = []
    meanErrorValid = []
    # backProp = grad(crossEntropy)
    prevGrads = {}
    grads = {}
    if parameters == None:
        parameters = initializeParameters(layerInfo)
    else:
        parameters = parameters
    L = len(parameters) // 2
    m = X.shape[0]
    batchSize = 32
    batchNum = m//batchSize
    # cache = forwardPropagation(X, parameters, activation)
    # cost = hp.crossEntropy(cache["H" +str(L)], Y)
    # meanError = hp.meanClassificationError(cache["H" +str(L)], Y)
    # validCache = forwardPropagation(validX, parameters, activation)
    # validCost = hp.crossEntropy(validCache["H" + str(L)], validY)
    # meanErrorValid = hp.meanClassificationError(validCache["H" + str(L)], validY)
    # errorTrain.append(cost)
    # errorValid.append(validCost)
    for i in range(0, numIterations):
        cost = 0
        validCost = 0
        classificationError = 0
        validClassificationError = 0
        hp.shuffleXYData(X, Y)
        for b in range(batchNum):
            start = batchSize*b
            end = batchSize*(b+1)
            XBatch = X[start:end]
            YBatch = Y[start:end]
            prevGrads = grads
            cache = forwardPropagation(XBatch, parameters, activation)
            cost += hp.crossEntropy(cache["H" +str(L)], YBatch)
            classificationError += hp.meanClassificationError(cache["H" +str(L)], YBatch)

            validCache = forwardPropagation(validX, parameters, activation)
            validClassificationError += hp.meanClassificationError(validCache["H" + str(L)], validY)
            validCost += hp.crossEntropy(validCache["H" + str(L)], validY)
            grads = backpropagation(parameters, cache,  XBatch, YBatch, lambd, activation)
            gradsMomentum = addMomentum(prevGrads, grads, momentum)
            parameters = updateParameters(parameters,gradsMomentum, learningRate)
        # cache = forwardPropagation(X, parameters, activation)
        errorTrain.append(cost/float(batchNum))
        errorValid.append(validCost/float(batchNum))
        meanError.append(classificationError/float(batchNum))
        meanErrorValid.append(validClassificationError/float(batchNum))
        if printCost and i % 10 == 0:
            print ("Cost at iteration %i: %f : %f" %(i, cost/float(batchNum), validCost/float(batchNum)))
            print ("Mean classification Err %i: %f : %f" %(i, classificationError/float(batchNum),validClassificationError/float(batchNum)))
    error = {
    "train": errorTrain,
    "valid": errorValid,
    "trainMean": meanError,
    "validMean" : meanErrorValid
    }
    return parameters, error

def test(testX, testY, parameters, activation = "sigmoid"):
    cache = forwardPropagation(testX, parameters, activation)
    L = len(parameters)//2
    cost = hp.crossEntropy(cache["H" +str(L)], testY)
    classificationError = hp.meanClassificationError(cache["H" +str(L)], testY)
    return cost, classificationError


# np.random.seed(1)

train_x, train_y, test_x, test_y, valid_x, valid_y = loadData.loadData()

# print backProp()

# FX = np.array([[0.1, 0.1],[0.5,0.1],[0.1,0.1],[0.1,0.5],[0.1,0.1]])
# Y = np.array([0,2])
# print meanClassificationError(FX, Y)

# checkGradient(np.array([train_x[3]]), np.array([train_y[3]]), 0.00001)

# ############## Q6(a) ########################
# parameters, error = train (train_x, train_y, valid_x, valid_y, [784, 100, 10], 200,learningRate = 0.1, printCost = True, momentum = 0, lambd = 0, activation = "sigmoid")
# hp.plotCrossEntropyError(error["train"], error["valid"])
#
# #############################################

# ############## Q6(b) ########################
# parameters, error = train (train_x, train_y, valid_x, valid_y, [784, 100, 10], 200,learningRate = 0.1, printCost = True, momentum = 0, lambd = 0, activation = "sigmoid")
# hp.plotClassificationError(error["trainMean"], error["validMean"])
#
# #############################################

# ############## Q6(c) ########################
# parameters, error = train (train_x, train_y, valid_x, valid_y, [784, 100, 10], 200,learningRate = 0.1, printCost = True, momentum = 0, lambd = 0, activation = "sigmoid")
# hp.visualizeParameter(parameters["W1"])
#
# #############################################

# ############## Q6(d) ########################
# momentums = [0.0, 0.5, 0.9]
# learningRates = [0.01, 0.2, 0.5]
# trainResult = {}
# for m in momentums:
#     for l in learningRates:
#         parameters, trainResult["m" + str(m) + "l" + str(l)] = train (train_x, train_y, valid_x, valid_y, [784, 100, 10], 200,learningRate = l, printCost = True, momentum = m, lambd = 0, activation = "sigmoid")
#
# for key, value in trainResult.iteritems():
#     print key
#     hp.plotCrossEntropyError(value["train"], value["valid"])
#
# #############################################

# ############## Q6(e) ########################
# hiddenLayer = [20,100, 200, 500]
# trainResult = {}
# for h in hiddenLayer:
#     parameters, trainResult["h" + str(h)] = train (train_x, train_y, valid_x, valid_y, [784, h, 10], 200,learningRate = 0.01, printCost = True, momentum = 0.5, lambd = 0, activation = "sigmoid")
#
# for key, value in trainResult.iteritems():
#     print key
#     hp.plotCrossEntropyError(value["train"], value["valid"])
#
# #############################################


############## Q6(f) ########################

# hiddenLayers = [100, 200, 300, 400, 500]
# learningRates = [0.01, 0.1, 0.3, 0.5, 0.7]
# momentums = [0, 0.5, 0.9]
# lambds = [0, 0.00001, 0.0001, 0.001, 0.01]
# for h in hiddenLayers:
#     for l in learningRates:
#         for m in momentums:
#             for L in lambds:
#                 parameters, error = train (train_x, train_y, valid_x, valid_y, [784, h, 10], 200,learningRate = l, printCost = False, momentum = m, lambd = L, activation = "sigmoid")
#                 testEntropy, testClassificationError = test(test_x, test_y, parameters)
#                 print ("Num of hidden units: %i, Learning Rate: %f, Moment: %f, Lambda: %f" %(h, l, m, L))
#                 print testEntropy
#                 print testClassificationError

# parameters, error = train (train_x, train_y, valid_x, valid_y, [784, 200, 10], 200,learningRate = 0.65, printCost = False, momentum = 0.5, lambd = 0.0006, activation = "sigmoid")
# testEntropy, testClassificationError = test(test_x, test_y, parameters)
# print testEntropy
# print testClassificationError
# hp.plotCrossEntropyError(error["train"], error["valid"])
# hp.plotClassificationError(error["trainMean"], error["validMean"])
# hp.visualizeParameter(parameters["W1"])

#############################################

############## Q6(g) ########################
# import csv
# with open('2LayerNeuralNet.csv', 'wb') as csvfile:
#     spamwriter = csv.writer(csvfile)
#     spamwriter.writerow(["hidden units1", "hidden units2", "Learning Rate", "Moment", "Lambda", "TestEntropy", "testClassificationError", "validEntropy", "validClassificationError"])
#     hiddenLayers1 = [50, 100, 200]
#     hiddenLayers2 = [50, 100, 200]
#     learningRates = [0.5, 0.6, 0.7, 0.8]
#     momentums = [0.5]
#     lambds = [0, 0.00001, 0.00005]
#     for h1 in hiddenLayers1:
#         for h2 in hiddenLayers2:
#             for l in learningRates:
#                 for m in momentums:
#                     for L in lambds:
#                         parameters, error = train (train_x, train_y, valid_x, valid_y, [784, h1,h2, 10], 100,learningRate = l, printCost = False, momentum = m, lambd = L, activation = "sigmoid")
#                         testEntropy, testClassificationError = test(test_x, test_y, parameters)
#                         print ("hidden units1: %i, hidden units2: %i Learning Rate: %f, Moment: %f, Lambda: %f" %(h1, h2, l, m, L))
#                         print testEntropy
#                         print testClassificationError
#                         spamwriter.writerow([h1, h2, l, m, L, error["train"], error["trainMean"], error["valid"], error["validMean"]])

# parameters, error = train (train_x, train_y, valid_x, valid_y, [784, 100,50, 10], 200,learningRate = 0.8, printCost = False, momentum = 0.5, lambd = 0.00005, activation = "sigmoid")
# testEntropy, testClassificationError = test(test_x, test_y, parameters)
#
# # print ("hidden units1: %i, hidden units2: %i Learning Rate: %f, Moment: %f, Lambda: %f" %(h1, h2, l, m, L))
# print error["train"][-1]
# print error["trainMean"][-1]
# print error["valid"][-1]
# print error["validMean"][-1]
# print testEntropy
# print testClassificationError
#
# hp.plotCrossEntropyError(error["train"], error["valid"])
# hp.plotClassificationError(error["trainMean"], error["validMean"])
# hp.visualizeParameter(parameters["W1"])

#############################################
