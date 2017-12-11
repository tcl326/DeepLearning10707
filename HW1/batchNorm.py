import numpy as np
import matplotlib.pyplot as plt
import loadData
import helpfulFunction as hp

def batchInitializeParameters(layerInfo):
    L = len(layerInfo)
    parameters = {}
    for l in range(L-1):
        parameters["b" + str(l+1)] = np.zeros((layerInfo[l+1],1))
        wb = np.sqrt(6)/np.sqrt(layerInfo[l] + layerInfo[l+1])
        parameters["W" + str(l+1)] = 2 * wb * np.random.random_sample((layerInfo[l+1],layerInfo[l])) - wb
        parameters["gamma" + str(l+1)] = 1
        parameters["beta" + str(l+1)] = 0
    return parameters

def batchBackpropagation(parameters, cache, X, Y, lambd = 0, epsilon = 0.01, activation = "sigmoid"):
    eY = hp.indicator(Y, 10)
    m = X.shape[0]
    L = len(parameters)//4
    gradients = {}
    assert eY.shape == cache["A" + str(L)].shape
    gradients["dH" + str(L)] = - eY/(cache["H" + str(L)])
    gradients["dY" + str(L)] = -1 * (eY - cache["H" + str(L)])
    for l in range(L, 0, -1):
        gradients["dA" + str(l)], gradients["dGamma" + str(l)], gradients["dBeta" + str(l)] =  \
        batchNormBackward(gradients["dY" + str(l)], cache["A" + str(l)] , cache["xHat" + str(l)],cache["variance" + str(l)], parameters["gamma" + str(l)], cache["mean" + str(l)], epsilon)
        gradients["dW" + str(l)] = np.dot(gradients["dA" + str(l)], np.transpose(cache["H" + str(l-1)])) / float(m) + 2 * lambd * parameters["W" + str(l)]
        gradients["db" + str(l)] = np.dot(gradients["dA" + str(l)], np.ones((m,1))) /float(m)
        gradients["dH" + str(l-1)] = np.dot(np.transpose(parameters["W" + str(l)]), gradients["dA" + str(l)])

        if l == 1:
            gradients["dY" + str(l-1)] = gradients["dH" + str(l-1)]
        else:
            gradients["dY" + str(l-1)] = gradients["dH" + str(l-1)] * hp.actDerivative(cache["H" + str(l-1)],activation)
    return gradients

def batchUpdateParameters(parameters, gradients, learningRate):
    L = len(parameters) // 4
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - gradients["dW" + str(l+1)] * learningRate
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - gradients["db" + str(l+1)] * learningRate
        parameters["gamma" + str(l+1)] = parameters["gamma" + str(l+1)] - (gradients["dGamma" + str(l+1)])* learningRate
        parameters["beta" + str(l+1)] = parameters["beta" + str(l+1)] - (gradients["dBeta" + str(l+1)])* learningRate
    return parameters

def addMomentum(prevGrads, grads, momentum):
    if momentum == 0 or len(prevGrads) == 0:
        return grads
    for key, value in grads.iteritems():
        value = value + momentum * prevGrads[key]
    return grads

def batchNormalize(X, gamma, beta, batchSize = 32, epsilon = 0.01):
    # epsilon = 0.000001
    m = X.shape[1]
    # print np.sum(X, axis =0)
    mean = np.mean(X, axis = 1, keepdims = True)
    # print mean
    variance = np.mean(np.power((X - mean),2), axis = 1, keepdims = True)
    # print variance
    xHat = (X - mean)/np.sqrt(variance + epsilon)
    # print normX
    # print gamma
    # print beta
    Y = gamma * xHat + beta
    return Y, mean, variance, xHat
def testBatchForwardPropagation(XBatch, parameters, trainCache, epsilon = 0.01, activation = "sigmoid"):
    L = len(parameters) // 4
    cache = {}
    cache["H0"] = np.transpose(XBatch)
    for l in range(L-1):
        # L2 = lambd * np.sum(parameters["W" + str(l+1)]**2, axis = 1)
        cache["A" + str(l+1)] = hp.preActivation(parameters["b" + str(l+1)], parameters["W" + str(l+1)],cache["H" + str(l)])
        cache["mean" + str(l+1)] = trainCache["mean" + str(l+1)]
        cache["variance" + str(l+1)] = 32./float(32-1) * trainCache["variance" + str(l+1)]
        # cache["xHat" + str(l+1)] = trainCache["xHat" + str(l+1)]
        cache["Y" + str(l+1)] = parameters["gamma" + str(l+1)]/np.sqrt(cache["variance" + str(l+1)] + epsilon) * (cache["A" + str(l+1)]) \
        + parameters["beta" + str(l+1)] - parameters["gamma" + str(l+1)] * cache["mean" + str(l+1)]/np.sqrt(cache["variance" + str(l+1)] + epsilon)
        cache["H" + str(l+1)] = hp.hiddenLayerActivation(cache["Y" + str(l+1)], activation)
        # print cache["mean" + str(l+1)].shape
        # print cache["variance" + str(l+1)].shape
        # print cache["xHat" + str(l+1)].shape
    cache["A" + str(L)] = hp.preActivation(parameters["b" + str(L)], parameters["W" + str(L)],cache["H" + str(L-1)])
    cache["Y" + str(L)], cache["mean" + str(L)], cache["variance" + str(L)], cache["xHat" + str(L)] = \
    batchNormalize(cache["A" + str(L)], parameters["gamma" + str(L)], parameters["beta" + str(L)], epsilon)

    cache["H" + str(L)] = hp.outputActivation(cache["Y" + str(L)])

    return cache

def batchForwardPropagation(XBatch, parameters, epsilon = 0.01, activation = "sigmoid"):
    L = len(parameters) // 4
    cache = {}
    cache["H0"] = np.transpose(XBatch)
    for l in range(L-1):
        # L2 = lambd * np.sum(parameters["W" + str(l+1)]**2, axis = 1)
        cache["A" + str(l+1)] = hp.preActivation(parameters["b" + str(l+1)], parameters["W" + str(l+1)],cache["H" + str(l)])
        cache["Y" + str(l+1)], cache["mean" + str(l+1)], cache["variance" + str(l+1)], cache["xHat" + str(l+1)] =  \
        batchNormalize(cache["A" + str(l+1)], parameters["gamma" + str(l+1)], parameters["beta" + str(l+1)], epsilon)
        cache["H" + str(l+1)] = hp.hiddenLayerActivation(cache["Y" + str(l+1)], activation)
    cache["A" + str(L)] = hp.preActivation(parameters["b" + str(L)], parameters["W" + str(L)],cache["H" + str(L-1)])
    cache["Y" + str(L)], cache["mean" + str(L)], cache["variance" + str(L)], cache["xHat" + str(L)] = \
    batchNormalize(cache["A" + str(L)], parameters["gamma" + str(L)], parameters["beta" + str(L)], epsilon)

    cache["H" + str(L)] = hp.outputActivation(cache["Y" + str(L)])

    return cache

def batchNormBackward(dOut,X, xHat, variance, gamma, mean, epsilon):
    m = dOut.shape[1]
    # print m
    dBeta = np.sum(dOut, axis = 1, keepdims = True)
    dGamma = np.sum(dOut * xHat, axis = 1, keepdims = True)
    dXHat = dOut * gamma
    dVar = np.sum(dXHat * (X - mean) * -1./2 * np.power((variance + epsilon),(-3./2)), axis = 1, keepdims = True)
    dMean = (np.sum(dXHat * -1./(np.sqrt(variance+epsilon)), axis = 1, keepdims = True)) + dVar * np.sum(-2*(X-mean), axis = 1, keepdims = True)*1./m
    dX = dXHat * 1./np.sqrt(variance+epsilon) + dVar * 2. * (X - mean)/m + dMean * 1./m
    return dX, dGamma, dBeta

def batchTrain(X, Y, validX, validY, layerInfo, numIterations, learningRate = 0.01, printCost = False, momentum = 0, lambd = 0, batchSize = 32, epsilon = 0.01, activation = "sigmoid"):
    errorTrain = []
    errorValid = []
    meanError = []
    meanErrorValid = []
    # backProp = grad(crossEntropy)
    testParameters = {}
    prevGrads = {}
    grads = {}
    parameters = batchInitializeParameters(layerInfo)
    m = X.shape[0]
    batchNum = m//batchSize
    L = len(parameters) // 4
    for i in range(0, numIterations):
        mean1 = 0
        mean2 = 0
        variance1 = 0
        variance2 = 0
        cost = 0
        validCost = 0
        classificationError = 0
        validClassificationError = 0
        hp.shuffleData(X, Y)
        for b in range(batchNum):
            prevGrads = grads
            start = batchSize*b
            end = batchSize*(b+1)
            XBatch = X[start:end]
            YBatch = Y[start:end]
            cache = batchForwardPropagation(XBatch, parameters, epsilon = epsilon, activation = activation)
            mean1 += cache["mean1"]
            mean2 += cache["mean2"]
            variance1 += cache["variance1"]
            variance2 += cache["variance2"]
            grads = batchBackpropagation(parameters, cache, XBatch, YBatch, lambd = lambd,epsilon = epsilon, activation = activation)
            grads = addMomentum(prevGrads, grads, momentum)
            parameters = batchUpdateParameters(parameters, grads, learningRate)
            # cache = batchForwardPropagation(X, parameters, epsilon = epsilon, activation = activation)
            cost += hp.crossEntropy(cache["H" +str(L)], YBatch)
            classificationError += hp.meanClassificationError(cache["H" +str(L)], YBatch)
        testParameters ={
        "mean1": mean1/float(batchNum),
        "mean2": mean2/float(batchNum),
        "variance1": variance1/float(batchNum),
        "variance2": variance2/float(batchNum)
        }
        validCache = testBatchForwardPropagation(validX, parameters, testParameters, epsilon = epsilon, activation = activation)
        validClassificationError = hp.meanClassificationError(validCache["H" + str(L)], validY)
        validCost = hp.crossEntropy(validCache["H" + str(L)], validY)
        errorTrain.append(cost/float(batchNum))
        errorValid.append(validCost)
        meanError.append(classificationError/float(batchNum))
        meanErrorValid.append(validClassificationError)
        if printCost and i % 10 == 0:
            print ("Cost at iteration %i: %f : %f" %(i, cost/float(batchNum), validCost))
            print ("Mean classification Err %i: %f : %f" %(i, classificationError/float(batchNum),validClassificationError))
    error = {
    "train": errorTrain,
    "valid": errorValid,
    "trainMean": meanError,
    "validMean" : meanErrorValid
    }

    return parameters, error, testParameters

def testBatch(testX, testY, testParameters, parameters, epsilon = 0.001, activation = "sigmoid"):
    L = len(parameters) // 4
    # hp.shuffleData(testX, testY)
    # cacheX = batchForwardPropagation(trainX, parameters, epsilon = epsilon, activation = activation)
    cache = testBatchForwardPropagation(testX, parameters, testParameters, epsilon = epsilon, activation = activation)
    cost = hp.crossEntropy(cache["H" +str(L)], testY)
    meanError = hp.meanClassificationError(cache["H" +str(L)], testY)
    return cost, meanError

train_x, train_y, test_x, test_y, valid_x, valid_y = loadData.loadData()

# ############ 6(h) #############
# parameters, error, testParameters = batchTrain(train_x, train_y, valid_x, valid_y, [784,100,50,10], 200, learningRate = 0.8, batchSize = 32, printCost = True, lambd = 0, epsilon = 0.001, momentum = 0, activation = "sigmoid")
# testEntropy, testClassificationError = testBatch(test_x, test_y, testParameters, parameters)
# print error["train"][-1]
# print error["trainMean"][-1]
# print error["valid"][-1]
# print error["validMean"][-1]
# print testEntropy
# print testClassificationError
# hp.plotCrossEntropyError(error["train"], error["valid"])
# hp.plotClassificationError(error["trainMean"], error["validMean"])
# hp.visualizeParameter(parameters["W1"])
# ###############################

############ 6(i) #############
# activationFunctions = ["sigmoid", "relu", "tanh"]
# for act in activationFunctions:
#     parameters, error, testParameters = batchTrain(train_x, train_y, valid_x, valid_y, [784,200,100,10], 200, learningRate = 0.1, batchSize = 32, printCost = False, lambd = 0, epsilon = 0.001, momentum = 0, activation = act)
#     testEntropy, testClassificationError = testBatch(test_x, test_y, testParameters, parameters, activation = act)
#     print act
#     print error["train"][-1]
#     print error["trainMean"][-1]
#     print error["valid"][-1]
#     print error["validMean"][-1]
#     print testEntropy
#     print testClassificationError
#     # hp.plotCrossEntropyError(error["train"], error["valid"])
#     # hp.plotClassificationError(error["trainMean"], error["validMean"])
#     # hp.visualizeParameter(parameters["W1"])
# ###############################
