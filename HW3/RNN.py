
# coding: utf-8

# In[1]:


import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN, Embedding, LSTM, Flatten
from keras.optimizers import RMSprop, SGD, Adagrad, Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model
from IPython.display import Image
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf


# In[2]:


class WordEmbeddingRNN():
    def __init__(self,
                vocab_size = 8000,
                embedding_size = 32,
                batch_size = 16,
                n_hidden=128,
                n_visible=3,
                reg_constant = 0.001,
                learning_rate = 0.001,
                truncated = False):
        self.model_path = "model_files/weights" + str(embedding_size) + ".{epoch:02d}-{val_loss:.2f}.hdf5"
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.n_visible = n_visible
        self.reg_constant = reg_constant
        self.truncated = truncated

    def build_model(self):
        self.model = Sequential()
        self.model.add(Embedding(self.vocab_size, self.embedding_size,  embeddings_initializer='uniform', embeddings_regularizer=l2(self.reg_constant), activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=self.n_visible))
        self.model.add(SimpleRNN(self.n_hidden, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=l2(self.reg_constant), recurrent_regularizer=l2(self.reg_constant), bias_regularizer=l2(self.reg_constant), activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=True))
        # model.add(Flatten())
        # model.add(Dense(n_hidden, activation='tanh'))
        self.model.add(Dense(self.vocab_size, activation='softmax',use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(self.reg_constant), bias_regularizer=l2(self.reg_constant), activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

    def config_model(self, learning_rate, reduce_factor):
        optimizer = Adam(lr=learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=reduce_factor, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
        self.checkpoint = ModelCheckpoint(self.model_path, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    def train_model(self,x_train,one_hoty_train, x_val, one_hot_y_val):
        self.history = self.model.fit(x_train, one_hot_y_train,
          batch_size=self.batch_size,
          epochs=100,
          validation_data=(x_val, one_hot_y_val),
          callbacks=[self.reduce_lr,self.checkpoint])

    def get_history(self):
        return self.history

    def plot_model(self):
        plot_model(self.model, to_file='model.png')
        Image(filename='model.png')



# In[44]:


class WordEmbedding():
    def __init__(self,
                vocab_size = 8000,
                embedding_size = 32,
                batch_size = 16,
                n_hidden=128,
                n_visible=3,
                reg_constant = 0.001,
                learning_rate = 0.001,
                truncated = False):
        self.model_path = "model_files/weights" + str(embedding_size) + ".{epoch:02d}-{val_loss:.2f}.hdf5"
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.n_visible = n_visible
        self.reg_constant = reg_constant
        self.truncated = truncated

    def build_model(self):
        self.model = Sequential()
        self.model.add(Embedding(self.vocab_size, self.embedding_size,  embeddings_initializer='uniform', embeddings_regularizer=l2(self.reg_constant), activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=self.n_visible))
#         self.model.add(SimpleRNN(self.n_hidden, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=l2(self.reg_constant), recurrent_regularizer=l2(self.reg_constant), bias_regularizer=l2(self.reg_constant), activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=True))
        self.model.add(Flatten())
        self.model.add(Dense(self.n_hidden, activation='tanh'))
        self.model.add(Dense(self.vocab_size, activation='softmax',use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(self.reg_constant), bias_regularizer=l2(self.reg_constant), activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

    def config_model(self, learning_rate, reduce_factor):
        optimizer = Adam(lr=learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=reduce_factor, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
        self.checkpoint = ModelCheckpoint(self.model_path, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    def train_model(self,x_train,one_hoty_train, x_val, one_hot_y_val):
        self.history = self.model.fit(x_train, one_hot_y_train,
          batch_size=self.batch_size,
          epochs=100,
          validation_data=(x_val, one_hot_y_val),
          callbacks=[self.reduce_lr,self.checkpoint])

    def get_history(self):
        return self.history

    def plot_model(self):
        plot_model(self.model, to_file='model.png')
        Image(filename='model.png')


# In[3]:


def get_sentences_list(text_file):
    sentences_list = []
    with open(text_file) as text:
        for sentence in text:
            sentence = sentence.lower()
            sentence = "START " + sentence + " END"
            sentence_list = sentence.split()
            sentences_list. append(sentence_list)
    return sentences_list


# In[4]:


def get_input_output(sentences_list, n, vocabulary_dict):
    output = []
    input = []
    for sentence in sentences_list:
        for word_index in range(len(sentence) - n -1):
            input.append([vocabulary_dict[x] if x in vocabulary_dict else vocabulary_dict['UNK'] for x in sentence[word_index:word_index+n]] )
            output.append(vocabulary_dict[sentence[word_index + n]] if sentence[word_index+n] in vocabulary_dict else vocabulary_dict['UNK'] )
    # print output
    # print input
    return np.array(input), np.array(output)


# In[6]:


vocabulary_dict = pickle.load(open('preprocessed_dict.p','rb'))
id_dict = {y:x for x,y in vocabulary_dict.iteritems()}


# In[7]:


n_visible = 3
x_train, y_train = get_input_output(get_sentences_list('train.txt'), n_visible, vocabulary_dict)
x_val, y_val = get_input_output(get_sentences_list('val.txt'), n_visible, vocabulary_dict)
one_hot_y_train = keras.utils.to_categorical(y_train, num_classes=8000)
one_hot_y_val = keras.utils.to_categorical(y_val, num_classes=8000)
print y_train.shape
print x_train.shape


# In[8]:


n_gram_id = 3
print id_dict[x_train[n_gram_id][0]],id_dict[x_train[n_gram_id][1]], id_dict[x_train[n_gram_id][2]], id_dict[y_train[n_gram_id]]


# In[22]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('output/modelloss')
plt.show()


# In[26]:

################### Q3.6.1 #########################
rnn16 = WordEmbeddingRNN(embedding_size=16)
rnn16.build_model()
for layer in rnn16.model.layers:
    print layer.weights[0].shape
rnn16.config_model(learning_rate=0.001, reduce_factor=0.5)
rnn16.train_model(x_train,one_hot_y_train, x_val, one_hot_y_val)
history = rnn16.get_history()
pickle.dump(history.history, open('history/16.p','wb'))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('output/modelloss16')
plt.show()


# In[ ]:


history16 = pickle.load(open('history/16.p','rb'))
plt.plot(np.exp(history.history16['val_loss']))
plt.title('Perplexity')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Perplexity'], loc='upper left')
plt.savefig('output/modelperplexity16')
plt.show()

#####################################################
# In[31]:

################### Q3.6.2 #########################
rnn32 = WordEmbeddingRNN(embedding_size=32)
rnn32.build_model()
rnn32.config_model(learning_rate=0.001, reduce_factor=0.5)
rnn32.train_model(x_train,one_hot_y_train, x_val, one_hot_y_val)
history = rnn32.get_history()
pickle.dump(history.history, open('history/32.p','wb'))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('output/modelloss32')
plt.show()


# In[35]:


plt.plot(np.exp(history.history['val_loss']))
plt.title('Perplexity')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Perplexity'], loc='upper left')
plt.savefig('output/modelperplexity32')
plt.show()


# In[49]:


rnn64 = WordEmbeddingRNN(embedding_size=64)
rnn64.build_model()
rnn64.config_model(learning_rate=0.001, reduce_factor=0.5)
rnn64.train_model(x_train,one_hot_y_train, x_val, one_hot_y_val)
history = rnn64.get_history()
pickle.dump(history.history, open('history/64.p','wb'))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('output/modelloss64')
plt.show()


# In[50]:


plt.plot(np.exp(history.history['val_loss']))
plt.title('Perplexity')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Perplexity'], loc='upper left')
plt.savefig('output/modelperplexity64')
plt.show()


# In[51]:


rnn128 = WordEmbeddingRNN(embedding_size=128)
rnn128.build_model()
rnn128.config_model(learning_rate=0.001, reduce_factor=0.5)
rnn128.train_model(x_train,one_hot_y_train, x_val, one_hot_y_val)
history = rnn128.get_history()
pickle.dump(history.history, open('history/128.p','wb'))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('output/modelloss128')
plt.show()


# In[52]:


plt.plot(np.exp(history.history['val_loss']))
plt.title('Perplexity')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Perplexity'], loc='upper left')
plt.savefig('output/modelperplexity128')
plt.show()

##################################################
# In[56]:


class WordEmbeddingRNNTruncated():
    def __init__(self,
                vocab_size = 8000,
                embedding_size = 32,
                batch_size = 16,
                n_hidden=128,
                n_visible=2,
                reg_constant = 0.001,
                learning_rate = 0.001,
                truncated = False):
        self.model_path = "model_files/weights" + str(embedding_size) + ".{epoch:02d}-{val_loss:.2f}.hdf5"
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.n_visible = n_visible
        self.reg_constant = reg_constant
        self.truncated = truncated

    def build_model(self):
        self.model = Sequential()
#         self.model.add(Embedding(self.vocab_size, self.embedding_size,  embeddings_initializer='uniform', embeddings_regularizer=l2(self.reg_constant), activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=self.n_visible, batch_input_shape=(self.batch_size, self.n_visible,self.embedding_size)))
        self.model.add(Embedding(self.vocab_size, self.embedding_size,  embeddings_initializer='uniform', embeddings_regularizer=l2(self.reg_constant), activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=self.n_visible))
        self.model.add(SimpleRNN(self.n_hidden, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=l2(self.reg_constant), recurrent_regularizer=l2(self.reg_constant), bias_regularizer=l2(self.reg_constant), activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=True))
        # model.add(Flatten())
        # model.add(Dense(n_hidden, activation='tanh'))
        self.model.add(Dense(self.vocab_size, activation='softmax',use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(self.reg_constant), bias_regularizer=l2(self.reg_constant), activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

    def config_model(self, learning_rate, reduce_factor):
        optimizer = Adam(lr=learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=reduce_factor, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
        self.checkpoint = ModelCheckpoint(self.model_path, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    def train_model(self,x_train,one_hoty_train, x_val, one_hot_y_val):
        self.history = self.model.fit(x_train, one_hoty_train,
          batch_size=self.batch_size,
          epochs=100,
          validation_data=(x_val, one_hot_y_val),
          callbacks=[self.reduce_lr,self.checkpoint],
          shuffle=False)

    def get_history(self):
        return self.history

    def plot_model(self):
        plot_model(self.model, to_file='model.png')
        Image(filename='model.png')


# In[57]:

##################### Q3.6.3 ########################
rnn16truncated = WordEmbeddingRNNTruncated(embedding_size=16)
rnn16truncated.build_model()
rnn16truncated.config_model(learning_rate=0.001, reduce_factor=0.5)

ten_percent = np.random.choice(len(x_train), int(len(x_train) * 0.1), replace = False)

rnn16truncated.train_model(x_train[ten_percent][:,1:3],one_hot_y_train[ten_percent], x_val[:,1:3], one_hot_y_val)
history = rnn16truncated.get_history()
pickle.dump(history.history, open('history/128.p','wb'))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('output/modelloss128')
plt.show()


# In[58]:


plt.plot(np.exp(history.history['val_loss']))
plt.title('Perplexity')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Perplexity'], loc='upper left')
plt.savefig('output/modelperplexity128')
plt.show()

#####################################################
# In[29]:


# ten_percent = np.random.choice(len(x_train), int(len(x_train) * 0.1), replace = False)


# In[55]:



# print ten_percent
# print len(x_train[ten_percent])
# print x_train[ten_percent][:,1:3]
# print one_hot_y_train[ten_percent].shape


# In[48]:


# rnn2 = WordEmbedding(embedding_size=2)
# rnn2.build_model()
# rnn2.config_model(learning_rate=0.001, reduce_factor=0.5)
# rnn2.train_model(x_train,one_hot_y_train, x_val, one_hot_y_val)
# history = rnn2.get_history()
# pickle.dump(history.history, open('history/2.p','wb'))
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('output/modelloss128')
# plt.show()
