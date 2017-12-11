import numpy as np
from RBM import RBM
import helpfulFunction as hp

v = np.array([0, 1, 1, 0])
V = np.tile(v, (6,1))
h = np.array([1, 1, 0])
H = np.tile(h, (6,1))
b = np.ones(h.size)
c = np.ones(v.size)
W = np.eye(h.size,v.size)

testRBM = RBM(V, H=H, W=W, b=b, c=c, testing=True)


def test_energy():
    v = np.array([1, 2, 3, 4])
    h = np.array([4, 3, 2])
    b = np.ones(v.size)
    c = np.ones(h.size)
    W = np.eye(v.size, h.size)
    learningParam = {"W": W, "b": b, "c": c}
    e = testRBM.energy(v, h, W, b, c)
    assert (e == -35)
    print e

def test_energy_mini():
    # v = np.array([1, 2, 3, 4])
    # V = np.tile(v, (3,1))
    # print V
    # h = np.array([4, 3, 2])
    # H = np.tile(h, (3,1))
    # print H
    # b = np.ones(h.size)
    # c = np.ones(v.size)
    # W = np.eye(h.size,v.size)
    # learningParam = {"W": W, "b": b, "c": c}
    # m = V.shape[0]
    # E = []
    # for i in range(m):
    #     E.append(testRBM.energy(v, h, W, b, c))
    # E = np.array(E)
    EMini = testRBM.energy_mini()
    np.testing.assert_array_equal (EMini , np.array([-5, -5, -5, -5, -5, -5]))
    print EMini

# def
# def

# test_energy_mini();
print np.random.binomial(1, np.array([0.5, 0.6, 0.3]), (784,3)).T
# x = np.array([[2,3,4],[1,2,3],[4,4,4]])
# x = hp.shuffleXData(x)
# print x

# print np.random.normal(0, [0.1,0.6], (2,784))
