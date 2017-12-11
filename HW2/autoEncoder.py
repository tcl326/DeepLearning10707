import numpy as np
import helpfulFunction as hp
import loadData
import pdb
import neuralNet as nn
import RBM
class AutoEncoder():
    def __init__(
        self,
        X,
        valid_x,
        test_x,
        num_hidden=100,
        num_visible=784,
        ):
        self.X_set = X
        self.X_test = test_x
        self.X_valid = valid_x
        self.initialize(num_hidden, num_visible)

    def initialize(self, num_hidden, num_visible, mu = 0, sigma = 0.1):
        self.W = np.random.normal(mu, sigma, (num_hidden, num_visible))
        self.c = np.zeros(num_visible)
        self.b = np.zeros(num_hidden)

    def loss_function_binary(self):
        # l(f(x)) = - sum (x log(xhat) + (1 - x) log(1 - xhat))
        lfx = - np.sum(self.X * np.log(self.XHat) + (1 - self.X) * np.log(self.XHat), axis = 1)
        return lfx

    def loss_function_real_value(self):
        # l(f(x)) = 1/2 * sum (xhat - x) ^ 2
        lfx = 1.0/2 * (np.sum(np.power(self.XHat - self.X, 2), axis = 1))
        # print self.XHat.shape
        return lfx

    def encoder(self):
        self.Hx = hp.sigmoid(self.b + np.dot(self.X, self.W.T))
        # return Hx

    def decoder(self):
        self.XHat = self.c + np.dot(self.Hx, self.W)

    def forward(self):
        self.encoder()
        self.decoder()

    def backprop(self):
        self.grad_a2 = self.XHat - self.X
        self.grad_w2 = np.dot(self.Hx.T, self.grad_a2)/self.batch_size
        self.grad_c = np.sum(self.grad_a2, axis=0)/self.batch_size
        self.grad_h1 = np.dot(self.grad_a2, self.W.T)
        self.grad_a1 = self.grad_h1 * self.Hx * (1 - self.Hx)
        self.grad_w1 = np.dot(self.grad_a1.T, self.X)/self.batch_size
        self.grad_b = np.sum(self.grad_a1, axis=0)/self.batch_size
        self.grad_w = self.grad_w1 + self.grad_w2

    def update(self):
        self.W = self.W - self.learning_rate * self.grad_w
        self.c = self.c - self.learning_rate * self.grad_c
        self.b = self.b - self.learning_rate * self.grad_b

    def train_mini_batch(self, batch_index):
        start = int(self.batch_size * batch_index)
        end = int(self.batch_size *(batch_index+1))
        self.X = self.X_set[start:end]
        if self.denoising:
            self.X = self.X * np.random.binomial(1, 1-self.dropout_rate, self.X.shape)
        self.forward()
        self.backprop()
        # update()
        self.update()
        # return self.cross_entropy(prob_x).mean()

    def train(self, epochs, batch_size=32, learning_rate = 0.1, denoising=False, dropout_rate=0.1):
        self.training_entropy=[]
        self.train_entropy = []
        self.train_reconstruction_error = []
        self.validation_entropy = []
        self.dropout_rate = dropout_rate
        self.denoising = denoising
        self.batch_size = float(batch_size)
        number_of_batch = self.X_set.shape[0]//self.batch_size
        self.learning_rate = learning_rate
        for epoch in range(epochs):
            np.random.shuffle(self.X_set)
            for batch_index in range(int(number_of_batch)):
                self.train_mini_batch(batch_index)
                # pdb.set_trace()
                self.training_entropy.append(np.mean(self.loss_function_real_value()))
            self.train_entropy.append(np.mean(self.training_entropy))
            print self.train_entropy[-1]
            # print np.mean(self.loss_function_real_value())
            self.X = self.X_valid
            self.encoder()
            self.decoder()
            self.validation_entropy.append(np.mean(self.loss_function_real_value()))
            print np.mean(self.loss_function_real_value())
            # pdb.set_trace()

def test(epochs, denoising=False, batch_size=32, num_hidden=100):
    train_x, train_y, test_x, test_y, valid_x, valid_y = loadData.loadData()
    auto_encoder = AutoEncoder(train_x, valid_x, test_x, num_hidden=num_hidden)
    auto_encoder.train(epochs, learning_rate = 0.1, batch_size=batch_size, denoising=denoising)
    # hp.visualize_parameter(auto_encoder.W, name="DenoisingAutoEncoderWn30k5")
    hp.plotCrossEntropyError(auto_encoder.train_entropy,auto_encoder.validation_entropy, name="k5")
    hp.save_param(auto_encoder.W, "WAuto" + str(num_hidden) + str(denoising))
    hp.save_param(auto_encoder.b, "bAuto" + str(num_hidden) + str(denoising))
    hp.save_param(auto_encoder.c, "cAuto" + str(num_hidden) + str(denoising))

if __name__ == '__main__':
    ########### Q5(e) ##########
    # test(50, batch_size=3)
    # W = hp.get_param_from_text("WAuto100.txt")
    # b = hp.get_param_from_text("bAuto100.txt")
    # c = hp.get_param_from_text("cAuto100.txt")
    #
    # train_x, train_y, test_x, test_y, valid_x, valid_y = loadData.loadData()
    # b2 = np.zeros((10,1))
    # wb = np.sqrt(6)/np.sqrt(100 + 10)
    # W2 = 2 * wb * np.random.random_sample((10,100)) - wb
    # parameters = {
    # "W1": W,
    # "b1": b.reshape(100,1),
    # "W2": W2,
    # "b2": b2,
    # }
    # print W.shape
    # print parameters["b1"].shape
    # # = parameters["W" + str(l+1)]
    # parameters, error = nn.train(train_x, train_y, valid_x, valid_y, [784,100,10], 200, parameters=None ,learningRate = 0.1, printCost = True, momentum = 0, lambd = 0, activation = "relu")
    # hp.plotClassificationError(error["trainMean"], error["validMean"])
    ############################

    ########### Q5(f) ##########
    # test(50, denoising=True, batch_size=5)
    # W = hp.get_param_from_text("WAuto100True.txt")
    # b = hp.get_param_from_text("bAuto100True.txt")
    # c = hp.get_param_from_text("cAuto100True.txt")
    #
    # train_x, train_y, test_x, test_y, valid_x, valid_y = loadData.loadData()
    # b2 = np.zeros((10,1))
    # wb = np.sqrt(6)/np.sqrt(100 + 10)
    # W2 = 2 * wb * np.random.random_sample((10,100)) - wb
    # parameters = {
    # "W1": W,
    # "b1": b.reshape(100,1),
    # "W2": W2,
    # "b2": b2,
    # }
    # print W.shape
    # print parameters["b1"].shape
    # # = parameters["W" + str(l+1)]
    # parameters, error = nn.train(train_x, train_y, valid_x, valid_y, [784,100,10], 200, parameters=None ,learningRate = 0.1, printCost = True, momentum = 0, lambd = 0, activation = "relu")
    # hp.plotClassificationError(error["trainMean"], error["validMean"])
    ############################

    ########### Q5(g) ##########
    nH = [50, 100, 200, 500]
    for h in nH:
        test(100, denoising=True, batch_size=5, num_hidden=h)
    for h in nH:
        test(100, denoising=False, batch_size=5, num_hidden=h)
    for h in nH:
        RBM.test(100, 5, batch_size=5, num_hidden=h)
    # train_x, train_y, test_x, test_y, valid_x, valid_y = loadData.loadData()
    # for h in nH:
    #     W = hp.get_param_from_text("WAuto" + str(h) + "False.txt")
    #     b = hp.get_param_from_text("bAuto" + str(h) + "False.txt")
    #     c = hp.get_param_from_text("cAuto" + str(h) + "False.txt")
    #
    #
    #     b2 = np.zeros((10,1))
    #     wb = np.sqrt(6)/np.sqrt(h + 10)
    #     W2 = 2 * wb * np.random.random_sample((10,h)) - wb
    #     parameters = {
    #     "W1": W,
    #     "b1": b.reshape(h,1),
    #     "W2": W2,
    #     "b2": b2,
    #     }
    #     print W.shape
    #     print parameters["b1"].shape
    #     # = parameters["W" + str(l+1)]
    #     parameters, error = nn.train(train_x, train_y, valid_x, valid_y, [784,h,10], 200, parameters=parameters ,learningRate = 0.1, printCost = False, momentum = 0, lambd = 0, activation = "relu")
    #     hp.plotClassificationError(error["trainMean"], error["validMean"], name="meanClassificationErrorAuto" + str(h))
    #     testEntropy, testClassificationError = nn.test(test_x, test_y, parameters,activation="relu")
    #     print h
    #     print testEntropy
    #     print testClassificationError
    #     print error["train"][-1]
    #     print error["trainMean"][-1]
    #     print error["valid"][-1]
    #     print error["validMean"][-1]
    #
    # for h in nH:
    #     W = hp.get_param_from_text("WAuto" + str(h) + "True.txt")
    #     b = hp.get_param_from_text("bAuto" + str(h) + "True.txt")
    #     c = hp.get_param_from_text("cAuto" + str(h) + "True.txt")
    #
    #     # train_x, train_y, test_x, test_y, valid_x, valid_y = loadData.loadData()
    #     b2 = np.zeros((10,1))
    #     wb = np.sqrt(6)/np.sqrt(h + 10)
    #     W2 = 2 * wb * np.random.random_sample((10,h)) - wb
    #     parameters = {
    #     "W1": W,
    #     "b1": b.reshape(h,1),
    #     "W2": W2,
    #     "b2": b2,
    #     }
    #     print W.shape
    #     print parameters["b1"].shape
    #     # = parameters["W" + str(l+1)]
    #     parameters, error = nn.train(train_x, train_y, valid_x, valid_y, [784,h,10], 200, parameters=parameters ,learningRate = 0.1, printCost = False, momentum = 0, lambd = 0, activation = "relu")
    #     hp.plotClassificationError(error["trainMean"], error["validMean"], name="meanClassificationErrorDenoising" + str(h))
    #     testEntropy, testClassificationError = nn.test(test_x, test_y, parameters,activation="relu")
    #     print h
    #     print testEntropy
    #     print testClassificationError
    #     print error["train"][-1]
    #     print error["trainMean"][-1]
    #     print error["valid"][-1]
    #     print error["validMean"][-1]
    # for h in nH:
    #     W = hp.get_param_from_text("W" + str(h) + ".txt")
    #     b = hp.get_param_from_text("b" + str(h) + ".txt")
    #     c = hp.get_param_from_text("c" + str(h) + ".txt")
    #
    #     # train_x, train_y, test_x, test_y, valid_x, valid_y = loadData.loadData()
    #     b2 = np.zeros((10,1))
    #     wb = np.sqrt(6)/np.sqrt(h + 10)
    #     W2 = 2 * wb * np.random.random_sample((10,h)) - wb
    #     parameters = {
    #     "W1": W,
    #     "b1": b.reshape(h,1),
    #     "W2": W2,
    #     "b2": b2,
    #     }
    #     print W.shape
    #     print parameters["b1"].shape
    #     # = parameters["W" + str(l+1)]
    #     parameters, error = nn.train(train_x, train_y, valid_x, valid_y, [784,h,10], 200, parameters=parameters ,learningRate = 0.1, printCost = False, momentum = 0, lambd = 0, activation = "relu")
    #     hp.plotClassificationError(error["trainMean"], error["validMean"], name="meanClassificationErrorRBM" + str(h))
    #     testEntropy, testClassificationError = nn.test(test_x, test_y, parameters, activation="relu")
    #     print h
    #     print testEntropy
    #     print testClassificationError
    #     print error["train"][-1]
    #     print error["trainMean"][-1]
    #     print error["valid"][-1]
    #     print error["validMean"][-1]
    ############################
