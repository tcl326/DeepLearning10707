import numpy as np
import helpfulFunction as hp
import pickle
import pdb
import matplotlib.pyplot as plt

class MultilayerPerception():
    def __init__(
        self,
        vocabulary_dict,
        train_text,
        val_text,
        vocab_size=8000,
        n_visible=3,
        n_hidden=128,
        embedding_size=16,
        activation='linear',
        word_embedding_weights=None,
        embed_to_hid_weight=None,
        hid_to_output_weight=None,
        hid_bias=None,
        output_bias=None ):

        if word_embedding_weights is None:
            word_embedding_weights = self.initialize((vocab_size,embedding_size))
        if embed_to_hid_weight is None:
            # embed_to_hid_weight = self.initialize((embedding_size, n_visible, n_hidden))
            embed_to_hid_weight = self.initialize((embedding_size * n_visible, n_hidden))

        if hid_to_output_weight is None:
            hid_to_output_weight = self.initialize((n_hidden,vocab_size))
        if hid_bias is None:
            hid_bias = self.initialize((n_hidden,))
        if output_bias is None:
            output_bias = self.initialize((vocab_size,))

        self.vocab_size = vocab_size
        self.activation = activation
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.embedding_size = embedding_size
        # self.input = input
        # self.output_label = output_label
        self.word_embedding_weights = word_embedding_weights
        self.embed_to_hid_weight = embed_to_hid_weight
        self.hid_to_output_weight = hid_to_output_weight
        self.hid_bias = hid_bias
        self.output_bias = output_bias
        self.train_text = train_text
        self.val_text = val_text
        self.vocabulary_dict = vocabulary_dict
        self.id_dict = {y:x for x,y in self.vocabulary_dict.iteritems()}

    def initialize(self, shape, mu=0, sigma=0.1):
        s = np.random.normal(mu, sigma, shape)
        return s

    def forward_propagation(self):
        self.word_embedding = self.word_embedding_weights[self.input].reshape(self.batch_size, self.embedding_size * self.n_visible)
        # self.word_embedding.
        # pdb.set_trace()
        # self.hidden_layer_pre_activation = np.einsum('nij, jik -> nk', self.word_embedding, self.embed_to_hid_weight) + self.hid_bias
        #  np.einsum('nij, jik->nk', word_embeding,embed_to_hid_layer_weights)
        self.hidden_layer_pre_activation = np.dot(self.word_embedding, self.embed_to_hid_weight) + self.hid_bias

        self.hidden_layer = hp.hiddenLayerActivation(self.hidden_layer_pre_activation, self.activation)
        self.output_pre_softmax = np.dot(self.hidden_layer, self.hid_to_output_weight) + self.output_bias
        self.output = hp.softmax(self.output_pre_softmax)
        # pdb.set_trace()

        # pdb.set_trace()
        # self.loss = hp.crossEntropy(self.output, self.output_label, n=8000)
    def get_loss(self):
        self.loss = -1 * np.mean(np.log(self.output[np.arange(self.batch_size),self.output_label]))

    def backward_propagation(self):
        self.d_output_pre_softmax = -1 * (hp.indicator(self.output_label, self.vocab_size).T - self.output)/float(self.batch_size)
        self.d_output_bias = np.sum(self.d_output_pre_softmax, axis=0)
        self.d_hid_to_output_weight = np.dot(self.hidden_layer.T, self.d_output_pre_softmax)

        self.d_hidden_layer = np.dot( self.d_output_pre_softmax,self.hid_to_output_weight.T)
        self.d_hidden_layer_pre_activation = self.d_hidden_layer * hp.actDerivative(self.hidden_layer, self.activation)
        self.d_hid_bias = np.sum(self.d_hidden_layer_pre_activation, axis=0)

        # self.d_embed_to_hid_weight = np.einsum('ijk, il -> kjl', self.word_embedding, self.d_hidden_layer_pre_activation)/float(self.batch_size)
        self.d_embed_to_hid_weight = np.dot(self.word_embedding.T, self.d_hidden_layer_pre_activation)

        # self.d_word_embedding = np.einsum('ijk, nk -> nji', self.embed_to_hid_weight, self.d_hidden_layer_pre_activation)/float(self.n_hidden)
        self.d_word_embedding = np.dot(self.d_hidden_layer_pre_activation, self.embed_to_hid_weight.T).reshape(self.batch_size, self.n_visible, self.embedding_size)

        self.d_word_embedding_weights = np.zeros((self.vocab_size,self.embedding_size))

        for i in range(len(self.input)):
            self.d_word_embedding_weights[self.input[i]] += self.d_word_embedding[i]

        # self.d_word_embedding_weights = self.d_word_embedding_weights
        # pdb.set_trace()

    def update(self):
        self.output_bias -= self.alpha * self.d_output_bias
        self.hid_bias -= self.alpha * self.d_hid_bias
        self.hid_to_output_weight -= self.alpha * self.d_hid_to_output_weight
        self.embed_to_hid_weight -= self.alpha * self.d_embed_to_hid_weight
        self.word_embedding_weights -= self.alpha * self.d_word_embedding_weights

    def get_sentences_list(self, text_file):
        sentences_list = []
        with open(text_file) as text:
            for sentence in text:
                sentence = sentence.lower()
                sentence = "START " + sentence + " END"
                sentence_list = sentence.split()
                sentences_list. append(sentence_list)
        return sentences_list

    def get_input_output(self, sentences_list, n):
        output = []
        input = []
        for sentence in sentences_list:
            for word_index in range(len(sentence) - n -1):
                input.append([self.vocabulary_dict[x] if x in self.vocabulary_dict else self.vocabulary_dict['UNK'] for x in sentence[word_index:word_index+n]] )
                output.append(self.vocabulary_dict[sentence[word_index + n]] if sentence[word_index+n] in self.vocabulary_dict else self.vocabulary_dict['UNK'] )
        # print output
        # print input
        return np.array(input), np.array(output)

    def get_perplexity(self):
        # return np.mean(np.log(self.output[self.output_label]))
        return np.exp(self.loss)

    def train(self, batch_size, learning_rate, epochs = 100):
        self.alpha = learning_rate
        self.perplexity_list = []
        self.batch_size = batch_size
        self.train_entropy = []
        self.validation_entropy = []
        train_text_list = self.get_sentences_list(self.train_text)
        val_text_list = self.get_sentences_list(self.val_text)
        self.train_input, self.train_output = self.get_input_output(train_text_list, self.n_visible)
        self.val_input, self.val_output =  self.get_input_output(val_text_list, self.n_visible)
        batch_num = self.train_input.shape[0]//self.batch_size
        # pdb.set_trace()
        for i in range(epochs):

            train_entropy = 0
            hp.shuffleXYData(self.train_input, self.train_output)
            self.batch_size = batch_size
            for b in range(batch_num):
                # print b
                start = self.batch_size * b
                end = self.batch_size*(b+1)
                self.input = self.train_input[start:end]
                self.output_label = self.train_output[start:end]
                self.forward_propagation()
                self.get_loss()
                self.backward_propagation()
                self.update()
                train_entropy += self.loss
            # pdb.set_trace()
            self.train_entropy.append(train_entropy/float(batch_num))
            self.input = self.val_input
            self.batch_size = self.input.shape[0]
            self.output_label = self.val_output
            self.forward_propagation()
            self.get_loss()
            self.perplexity = self.get_perplexity()
            self.perplexity_list.append(self.perplexity)
            # print self.loss
            self.validation_entropy.append(self.loss)
            print i
            if i % 1 == 0:
                print ("Cost at iteration %i: Train: %f Val: %f" %(i, self.train_entropy[-1], self.validation_entropy[-1]))
                print ("perplexity at iteration %i: %f" %(i, self.perplexity))

def test(vocabulary_dict, train_text, validation_text, learning_rate, n_hidden, activation='linear',embedding_size=16, batch_size=512):
    multilayerPerception = MultilayerPerception(vocabulary_dict, train_text, validation_text, n_hidden=n_hidden, activation=activation,embedding_size=embedding_size)
    multilayerPerception.alpha = learning_rate
    multilayerPerception.train(batch_size, learning_rate)
    # pickle.dump(multilayerPerception.word_embedding_weights, open("word_embedding_weights" + str(n_hidden) + activation + str(embedding_size) + ".p", 'wb'))
    pickle.dump(multilayerPerception.train_entropy, open("train_entropy" + str(n_hidden) + activation + str(embedding_size) +".p", 'wb'))
    pickle.dump(multilayerPerception.validation_entropy, open("validation_entropy" + str(n_hidden) + activation + str(embedding_size) +".p", 'wb'))
    pickle.dump(multilayerPerception.perplexity_list, open("perplexity" + str(n_hidden) + activation + str(embedding_size) +".p", 'wb'))
    pickle.dump(multilayerPerception.word_embedding_weights, open("word_embedding_weights" + str(n_hidden) + activation + str(embedding_size) +".p", 'wb'))
    pickle.dump(multilayerPerception.embed_to_hid_weight, open("embed_to_hid_weight" + str(n_hidden) + activation + str(embedding_size) + ".p", 'wb'))
    pickle.dump(multilayerPerception.hid_to_output_weight, open("hid_to_output_weight" + str(n_hidden) + activation + str(embedding_size) + ".p", 'wb'))
    pickle.dump(multilayerPerception.hid_bias, open("hid_bias" + str(n_hidden) + activation + str(embedding_size) + ".p", 'wb'))
    pickle.dump(multilayerPerception.output_bias, open("output_bias" + str(n_hidden) + activation + str(embedding_size) + ".p", 'wb'))

    # return error

def generate_language(three_words,word_embedding_weights, embed_to_hid_weight,hid_to_output_weight,hid_bias,output_bias):
    multilayerPerception = MultilayerPerception(vocabulary_dict, 'train.txt', 'val.txt', n_hidden=128, activation='linear',embedding_size=16)
    multilayerPerception.word_embedding_weights = word_embedding_weights
    multilayerPerception.embed_to_hid_weight = embed_to_hid_weight
    multilayerPerception.hid_to_output_weight = hid_to_output_weight
    multilayerPerception.hid_bias = hid_bias
    multilayerPerception.output_bias = output_bias
    multilayerPerception.batch_size = 1
    output_list = three_words
    c = 0
    output = ''
    while c < 10 and output != 'END':
        multilayerPerception.input = [vocabulary_dict[output_list[-3]],vocabulary_dict[output_list[-2]], vocabulary_dict[output_list[-1]]]
        multilayerPerception.forward_propagation()
        output = multilayerPerception.id_dict[np.argmax(multilayerPerception.output)]
        output_list.append(output)
        c += 1
    return output_list

def difference_between_words(word1, word2, vocabulary_dict, word_embedding_weights):
    return difference_between_embeddings(word_embedding_weights[vocabulary_dict[word1]],word_embedding_weights[vocabulary_dict[word2]])

def difference_between_embeddings(embedding_one, embedding_two):
    euclidean_dist = np.linalg.norm(embedding_one - embedding_two)
    return euclidean_dist

if __name__=="__main__":
    vocabulary_dict = pickle.load(open('preprocessed_dict.p','rb'))
    id_dict = {y:x for x,y in vocabulary_dict.iteritems()}
    # print vocabulary_dict
    ##################### Q3.2 #############################
    # test(vocabulary_dict, 'train.txt', 'val.txt', 0.1, 128, batch_size=512)
    # test(vocabulary_dict, 'train.txt', 'val.txt', 0.1, 256, batch_size=512)
    # test(vocabulary_dict, 'train.txt', 'val.txt', 0.1, 512, batch_size=512)
    # train_entroyp128 = pickle.load(open("train_entropy128linear.p",'rb'))
    # validation_entropy128linear = pickle.load(open("validation_entropy128linear.p",'rb'))
    # hp.plotCrossEntropyError(train_entroyp128,validation_entropy128linear)
    # train_entroyp256 = pickle.load(open("train_entropy256linear.p",'rb'))
    # validation_entropy256linear = pickle.load(open("validation_entropy256linear.p",'rb'))
    # hp.plotCrossEntropyError(train_entroyp256,validation_entropy256linear)
    # train_entroyp512 = pickle.load(open("train_entropy512linear.p",'rb'))
    # validation_entropy512linear = pickle.load(open("validation_entropy512linear.p",'rb'))
    # hp.plotCrossEntropyError(train_entroyp512,validation_entropy512linear)
    # perplexity128 = np.exp(pickle.load(open('validation_entropy128linear.p', 'rb')))
    # hp.plotPerplexity(perplexity128)
    # perplexity256 = np.exp(pickle.load(open('validation_entropy256linear.p', 'rb')))
    # hp.plotPerplexity(perplexity256)
    # perplexity512 = np.exp(pickle.load(open('validation_entropy512linear.p', 'rb')))
    # hp.plotPerplexity(perplexity512)
    ########################################################
    ##################### Q3.3 #############################
    # test(vocabulary_dict, 'train.txt', 'val.txt', 0.2, 512, activation='tanh')
    # test(vocabulary_dict, 'train.txt', 'val.txt', 0.01, 256, activation='tanh')
    # test(vocabulary_dict, 'train.txt', 'val.txt', 0.01, 512, activation='tanh')
    # train_entroyp128tanh = pickle.load(open("train_entropy128tanh.p",'rb'))
    # validation_entropy128tanh = pickle.load(open("validation_entropy128tanh.p",'rb'))
    # hp.plotCrossEntropyError(train_entroyp128tanh,validation_entropy128tanh)
    # train_entroyp256tanh = pickle.load(open("train_entropy256tanh.p",'rb'))
    # hp.plotTrainCrossEntropyError(train_entroyp256tanh)
    # train_entroyp512tanh = pickle.load(open("train_entropy512tanh.p",'rb'))
    # hp.plotTrainCrossEntropyError(train_entroyp512tanh)
    # perplexity128tanh = np.exp(pickle.load(open('validation_entropy128tanh.p', 'rb')))
    # hp.plotPerplexity(perplexity128tanh)
    # perplexity256tanh = np.exp(pickle.load(open('validation_entropy256tanh.p', 'rb')))
    # hp.plotPerplexity(perplexity256tanh)
    # perplexity512tanh = np.exp(pickle.load(open('validation_entropy512tanh.p', 'rb')))
    # hp.plotPerplexity(perplexity512tanh)
    ########################################################
    ##################### Q3.4 #############################
    # three_word_list = [['the','government', 'of'],['the','new','york'],['life', 'in', 'the'],['president','of','the'],['i','lost','my']]
    #
    # activation = 'tanh'
    # n_hidden = '512'
    #
    # word_embedding_weights = pickle.load(open("word_embedding_weights" + n_hidden + activation + "16.p",'rb'))
    # embed_to_hid_weight = pickle.load(open("embed_to_hid_weight" + n_hidden + activation + "16.p",'rb'))
    # hid_to_output_weight = pickle.load(open("hid_to_output_weight"  + n_hidden + activation + "16.p",'rb'))
    # hid_bias = pickle.load(open("hid_bias"  + n_hidden + activation + "16.p",'rb'))
    # output_bias = pickle.load(open("output_bias"  + n_hidden + activation + "16.p",'rb'))
    # generated_sentence_list = []
    # for three_words in three_word_list:
    #     generated_sentence = generate_language(three_words,word_embedding_weights, embed_to_hid_weight,hid_to_output_weight,hid_bias,output_bias)
    #     generated_sentence_list.append(generated_sentence)
    #
    # print generated_sentence_list
    # pickle.dump(generated_sentence_list, open("generated_sentence.p", 'wb'))
    # generated_sentence = pickle.load(open("generated_sentence.p",'rb'))
    # sentences_list = []
    # for sentence in generated_sentence:
    #     sentences_list.append(' '.join(sentence))
    #     print ' '.join(sentence)
    # print sentences_list
    #
    # print 'stock, exchange:', difference_between_words('stock','exchange',vocabulary_dict,word_embedding_weights)
    # print 'stock, price:', difference_between_words('stock','price',vocabulary_dict,word_embedding_weights)
    # print 'stock, bottle:', difference_between_words('stock','bottle',vocabulary_dict,word_embedding_weights)
    # print 'stock, president:', difference_between_words('stock','president',vocabulary_dict,word_embedding_weights)
    # print 'stock, UNK:'', difference_between_words('stock','UNK',vocabulary_dict,word_embedding_weights)
    ########################################################
    ##################### Q3.5 #############################
    # test(vocabulary_dict, 'train.txt', 'val.txt', 0.15, 512, activation='linear', embedding_size=2, batch_size=512)
    # word_embedding = pickle.load(open("word_embedding_weights128tanh2.p",'rb'))
    # chosen_index = np.random.choice(8000, 500, replace=False)
    # chosen_embedding = word_embedding[chosen_index]
    # x = chosen_embedding[:,0]
    # y = chosen_embedding[:,1]
    #
    # plt.scatter(x,y)
    # for i in range(len(chosen_index)):
    #     plt.annotate(id_dict[chosen_index[i]],xy=(x[i],y[i]))
    # plt.savefig("Plot_embedding")
    # plt.show()
    ########################################################
