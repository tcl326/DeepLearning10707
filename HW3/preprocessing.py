# Create a vocabulary dictionary (i.e.) you are required to create an entry for every word in the training set (make
# the data lower-cased). This will serve as a lookup table for the words and their corresponding id. Note: Splitting
# it on space should do, there is no need for any additional processing.
import numpy as np
import pickle
# import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

class WordInfo():
    def __init__(self, word, word_id, word_freq):
        self.word = word
        self.word_id = word_id
        self.word_freq = word_freq
    def __gt__(self, word_info2):
        return self.word_freq > word_info2.word_freq

    def __repr__(self):
        return repr((self.word, self.word_id, self.word_freq))

def create_vocab_dict(file_path):
    word_id = 1
    # word_info_list = []
    word_dict = {}
    with open(file_path) as text:
        for sentence in text:
            sentence = sentence.lower()
            sentence = "START " + sentence + " END"
            sentence_list = sentence.split()
            for word in sentence_list:
                # print word
                if word not in word_dict:
                    word_dict[word] = WordInfo(word, word_id, 1)
                    word_id += 1
                else:
                    word_dict[word].word_freq += 1
    print word_dict['START']
    return word_dict

def limit_vocab_size(word_info_dict, size):
    word_info_list = word_info_dict.values()
    sorted_word_info_list = sorted(word_info_list, key=lambda word_info: word_info.word_freq, reverse=True)
    resized_word_info_list = sorted_word_info_list[:size-1]
    resized_word_info_dict = {}
    word_id = 0
    for word_info in resized_word_info_list:
        # word_info.word_id = word_id
        resized_word_info_dict[word_info.word] = word_id
        word_id += 1
    resized_word_info_dict['UNK'] = word_id
    print (resized_word_info_dict['UNK'])
    return resized_word_info_dict

def n_gram_distribution(file_path, n, vocab_dict):
    with open(file_path) as text:
        n_gram_dict = {}
        for sentence in text:
            sentence = sentence.lower()
            sentence = "START " + sentence + " END"
            sentence_list = sentence.split()
            for word_index in range(len(sentence_list) - n + 1):
                n_words = sentence_list[word_index]
                if n_words not in vocab_dict:
                    n_words = "UNK"
                for i in range(1,n):
                    n_words = n_words + ' ' + sentence_list[word_index + i]
                if n_words not in n_gram_dict:
                    n_gram_dict[n_words] = 1
                else:
                    n_gram_dict[n_words] += 1
    # print n_gram_dict
    print len(n_gram_dict)
    return n_gram_dict

def plot_top_50_distribution(n_gram_dict):
    counter = 0
    top_50_dict = {}
    names = []
    values = []
    for key, value in sorted(n_gram_dict.iteritems(), key=lambda (k,v): (v,k), reverse=True):
        if counter == 50:
            break;
        names.append(key)
        values.append(value)
        counter += 1
    # names = list(top_50_dict.keys())
    # values = list(top_50_dict.values())
    print names
    print values

    # fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    plt.barh(names, values)
    # plt.xlabel("xlabe", fontsize=10)
    plt.tick_params(axis='y', labelsize=5)
    plt.tight_layout()
    plt.show()


def plot_distribution(n_gram_dict):
    counter = 0
    top_50_dict = {}
    names = []
    values = []
    for key, value in sorted(n_gram_dict.iteritems(), key=lambda (k,v): (v,k), reverse=True):
        # for i in range(value):
        #     names.append(key)
        values.append(value)
    # names = list(n_gram_dict.keys())
    # values = list(n_gram_dict.values())
    print len(names)
    # print names
    plt.plot(np.arange(len(values)), values)
    plt.fill_between( np.arange(len(values)), values)
    plt.ylabel("Frequency")
    plt.xlabel("4 Gram Sorted by Descending Frequency")
    plt.title("4 Gram Distribution Plot")
    # print values
    # plt.
    plt.show()

##################### Q3.1 #############################
# vocab_dict = create_vocab_dict("train.txt")
# final_dict = limit_vocab_size(vocab_dict, 8000)
# pickle.dump(final_dict, open('preprocessed_dict.p', 'wb'))
# final_dict = pickle.load(open('preprocessed_dict.p', 'rb'))
# n_gram_dict = n_gram_distribution('train.txt', 4, final_dict)
# pickle.dump(n_gram_dict, open('n_gram_dict.p', 'wb'))
# plot_top_50_distribution(n_gram_dict)
# plot_distribution(n_gram_dict)
########################################################
