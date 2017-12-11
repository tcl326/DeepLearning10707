import csv
import numpy as np
import matplotlib.pyplot as plt

def loadData():
    train_X, train_Y = loadDataFile('digitstrain.txt')
    test_X, test_Y = loadDataFile('digitstest.txt')
    valid_X, valid_Y = loadDataFile('digitsvalid.txt')

    return train_X, train_Y, test_X, test_Y, valid_X, valid_Y

def loadDataFile(fileName):
    data = []
    X = []
    Y = []
    with open(fileName, 'rb') as dataSet:
        dataReader = csv.reader(dataSet,delimiter = ',',quoting=csv.QUOTE_NONNUMERIC)
        for row in dataReader:
            # print row
            data.append(row)
    data = np.array(data)
    np.random.shuffle(data)
    # X.append(row[:-1])
    # Y.append(row[-1])
    X = data[:,:-1]
    print X.shape
    Y = data[:,-1]
    print Y.shape
    assert X.shape[1] == 28*28
    assert Y.shape[0] == X.shape[0]
    return X, Y

def plot(x, y):
    # print y
    image = np.reshape(x,(28,28))
    impgplot = plt.imshow(image)
    plt.show()

# train_x, train_y, test_x, test_y, valid_x, valid_y = loadData();
# plot(train_x[-1],train_y[-1])
# plot(valid_x[-1], valid_y[-1])
# print train_x[0]
