import numpy as np
import helpfulFunction as hp
import loadData
import pdb
import PIL.Image as Image
import neuralNet as nn

# import test
# learningParam = {"W": W, "b": b, "c": c}
class RBM():
    def __init__ (
        self,
        input,
        input_validation=None,
        input_test=None,
        H=None,
        n_visible=784,
        n_hidden=100,
        batch_size=32,
        W=None,
        b=None,
        c=None,
        testing=False):
        if W is None:
            W = self.initialize((n_hidden, n_visible))
        if H is None:
            H = self.initialize((batch_size, n_hidden))
        if b is None:
            b = self.initialize((n_hidden,))
        if c is None:
            c = self.initialize((n_visible,))

        self.W = W
        self.H = H
        self.b = b
        self.c = c
        self.X_set = np.around(input)
        self.X_validation = input_validation
        self.X_test = input_test
        self.m = batch_size
        if testing:
            self.X = input

    def initialize(self, shape, mu = 0, sigma = 0.1):
        s = np.random.normal(mu, sigma, shape)
        return s

    # def energy(self, x, h, W, b, c):
    #     # W = learningParam["W"]
    #     # b = learningParam["b"]
    #     # c = learningParam["c"]
    #     return -np.dot(b, h) - np.dot(c, x) - np.dot(np.dot(h.T, W),x)

    # def z(self, energyArray):
    #     z = np.sum(np.exp(-energyArray))
    #     return z

    def energy_mini (self):
        # W = learningParam["W"]
        # b = learningParam["b"]
        # c = learningParam["c"]
        energy = - np.sum(np.dot(self.H, self.W) * self.X, axis = 1) - np.dot(self.X, self.c) - np.dot(self.H, self.b)
        # print np.dot(self.X, self.c)
        # print np.dot(self.H, self.b)
        # print np.sum(np.dot(self.H, self.W) * self.X, axis = 1)
        # print self.X
        # print self.c
        # pdb.set_trace()
        return energy
    def z_mini(self, energy_mini):
        # print energy_mini
        Z = np.sum(np.exp(-energy_mini))
        # pdb.set_trace()
        return Z

    # def prob_h_given_x(self, x, W, b):
    #     p = hp.sigmoid(b + np.dot(W,x))
    #     return p

    def prob_h_given_x_mini(self, X):
        P =  hp.sigmoid(self.b + np.dot(X, self.W.T))
        # print P.shape
        return P

    # def prob_x_given_h(self, h, W, c):
    #     p = hp.sigmoid(c + np.dot(h.T, W))
    #     return p

    def prob_x_given_h_mini(self):
        P = hp.sigmoid(self.c + np.dot(self.H, self.W))
        # print P
        # print P.shape
        return P

    def free_energy_mini(self):
        bias_term = np.dot(self.X,self.c)
        hidden_term = np.sum(hp.softplus(self.b + np.dot(self.X, self.W.T)), axis = 1)
        return -bias_term - hidden_term

    def prob_x_mini(self):
        F = self.free_energy_mini()
        energy_all = self.energy_mini()
        z = self.z_mini(energy_all)
        # print z
        return np.exp(-F)/z

    def cross_entropy(self):
        return np.mean(-self.X * np.log(hp.sigmoid(self.Xtilde)) - (1-self.X)*np.log(1-hp.sigmoid(self.Xtilde)))

    def reconstruction_error(self):
        return np.linalg.norm(self.X - self.Xtilde)/float(self.X.shape[0])

    def sample(self, prob, size = None):
        return np.random.binomial(1, prob, size)

    def gibbs_sampling(self,k):
        prob_h_given_x = self.prob_h_given_x_mini(self.X);
        # print prob_h_given_x
        self.H = self.sample(prob_h_given_x)
        for i in range(k):
            prob_x_given_h = self.prob_x_given_h_mini()
            self.Xtilde = self.sample(prob_x_given_h)
            prob_h_given_x = self.prob_h_given_x_mini(self.Xtilde)
            self.H = self.sample(prob_h_given_x)
        # return X, H

    def gibbs_generate(self,k):
        # prob_h_given_x = self.prob_h_given_x_mini(self.X);
        # print prob_h_given_x
        # self.H = self.sample(prob_h_given_x)
        for i in range(k):
            prob_x_given_h = self.prob_x_given_h_mini()
            # print prob_x_given_h.shape
            self.Xtilde = self.sample(prob_x_given_h)
            # print self.Xtilde.shape
            prob_h_given_x = self.prob_h_given_x_mini(self.Xtilde)
            # print prob_h_given_x.shape
            self.H = self.sample(prob_h_given_x)

    def h_x(self, X):
        return hp.sigmoid(self.b + np.dot(X, self.W.T))

    def update(self,learning_rate, k=1):
        self.gibbs_sampling(k)
        m = float(self.X.shape[0])
        self.W = self.W + learning_rate * (np.dot(self.h_x(self.X).T, self.X)/m - np.dot(self.h_x(self.Xtilde).T, self.Xtilde)/m)
        # print self.b.shape
        # print np.sum(self.h_x(self.X) - self.h_x(self.Xtilde), axis = 0).
        self.b = self.b + learning_rate * np.sum(self.h_x(self.X) - self.h_x(self.Xtilde), axis = 0)/m
        # print self.b.shape
        self.c = self.c + learning_rate * np.sum(self.X - self.Xtilde, axis = 0)/m

    def train_mini_batch(self, batch_index, learning_rate = 0.1, k=1):
        start = self.m * batch_index
        end = self.m *(batch_index+1)
        self.X = self.X_set[start:end]
        self.update(learning_rate, k=k)
        prob_x = self.prob_x_mini()
        # print prob_x
        return self.cross_entropy()

    def train(self, epochs, k=1):
        self.train_entropy = []
        self.train_reconstruction_error = []
        self.validation_entropy = []
        number_of_batch = self.X_set.shape[0]//self.m
        learning_rate = 0.01
        counter = 0
        prevStochasticRec = 100
        for epoch in range(epochs):
            np.random.shuffle(self.X_set)
            mean_cost = 0;
            # if (counter > 5):
            #     learning_rate = learning_rate/2.0
            #     counter = 0
            for batch_index in range(number_of_batch):
                mean_cost += self.train_mini_batch(batch_index, learning_rate=learning_rate, k=k)


            stochasticRec = self.reconstruction_error()
            # if (stochasticRec - prevStochasticRec > 0):
            #     counter += 1
            # prevStochasticRec = stochasticRec
            self.X = self.X_validation
            self.gibbs_sampling(k)
            self.validation_entropy.append(self.cross_entropy())
            self.train_entropy.append(mean_cost/number_of_batch)
            # validation_reconstruction = self.reconstruction_error()
            print('Training epoch %d, cost is ' % epoch, mean_cost/number_of_batch)
            print('Validation epoch %d, cost is' % epoch, self.validation_entropy[epoch])
            print('Leraning Rate is', learning_rate)
            print('Training Reconstruction is', stochasticRec)
            # print('Validation Reconstruction is', validation_reconstruction)
        self.X = self.X_test
        self.gibbs_sampling(k)
        test_entropy = self.cross_entropy()
        # test_reconstruction = self.reconstruction_error()
        print('Test cost is', test_entropy)
        # print('Test Reconstruction is', test_reconstruction)
            # pdb.set_trace()

def test(epoch, k, batch_size=32, num_hidden=100):
    train_x, train_y, test_x, test_y, valid_x, valid_y = loadData.loadData()
    rbm = RBM(train_x, input_validation=valid_x, input_test=test_x, batch_size=batch_size, n_hidden=num_hidden)
    rbm.train(epoch , k=k)
    # hp.visualize_parameter(rbm.W, name="Wn30k5")
    hp.plotCrossEntropyError(rbm.train_entropy,rbm.validation_entropy, name="k5")
    hp.save_param(rbm.W, "W" + str(num_hidden))
    hp.save_param(rbm.b, "b" + str(num_hidden))
    hp.save_param(rbm.c, "c" + str(num_hidden))

# def gibbs_sampling(k, X, W):
#     prob_h_given_x = hp.sgmoid()
#         #     v=sigmoid(W*x+repmat(b,1,N));
#         # v=binornd(1,v,size(v));
#         # x=sigmoid(W'*v+repmat(c,1,N));
#         # x=binornd(1,x,size(x));
#
#     self.prob_h_given_x_mini(self.X);
#     # print prob_h_given_x
#     self.H = self.sample(prob_h_given_x)
#     for i in range(k):
#         prob_x_given_h = self.prob_x_given_h_mini()
#         self.Xtilde = self.sample(prob_x_given_h)
#         prob_h_given_x = self.prob_h_given_x_mini(self.Xtilde)
#         self.H = self.sample(prob_h_given_x)

# if __name__ == '__main__':
    ########### Q5(a) ##########
    # epoch = 200
    # k = 1
    # test(epoch, k, batch_size=5)
    ############################

    ########### Q5(b) ##########
    # epoch = 300
    # k = [5, 10, 20]
    # for i in k:
    #     test(epoch, i)
    ############################

    ########### Q5(c) ##########
    # W = hp.get_param_from_text("W20k.txt")
    # b = hp.get_param_from_text("b20k.txt")
    # c = hp.get_param_from_text("c20k.txt")
    # X = np.random.normal(0, 0.1, (100, 784))
    #
    # rbm = RBM(X)
    # rbm.X = X
    # rbm.b = b
    # rbm.c = c
    # rbm.W = W
    #
    # rbm.gibbs_sampling(1000)
    #
    # hp.visualize_parameter(rbm.Xtilde, name="Xtildw")
    ############################

    ########### Q5(d) ##########
    # W = hp.get_param_from_text("W1k.txt")
    # b = hp.get_param_from_text("b1k.txt")
    # c = hp.get_param_from_text("c1k.txt")
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
    #
    # parameters, error = nn.train(train_x, train_y, valid_x, valid_y, [784,100,10], 200, parameters=parameters ,learningRate = 0.1, printCost = True, momentum = 0, lambd = 0, activation = "relu")
    # hp.plotClassificationError(error["trainMean"], error["validMean"])
    # parameters, error = nn.train(train_x, train_y, valid_x, valid_y, [784,100,10], 200, parameters=None ,learningRate = 0.1, printCost = True, momentum = 0, lambd = 0, activation = "relu")
    # hp.plotClassificationError(error["trainMean"], error["validMean"])
    ############################
