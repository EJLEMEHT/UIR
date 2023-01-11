import numpy as np


def refracted2(vector, norm, n1, n2):
    n = n2/n1
    return n*(vector - np.dot(norm, vector) * norm) + np.sqrt(1 - n ** 2 * (1 - np.dot(norm, vector) ** 2)) * norm


def refracted(vector, norm, n1, n2):
    n = n1 / n2
    cosI = -np.dot(norm, vector)
    sinT2 = n ** 2 * (1 - cosI ** 2)
    cosT = np.sqrt(1 - sinT2)
    # if cosI < 0.25:
    #     return reflected(vector, norm)
    return n * vector + (n * cosI - cosT) * norm


a = np.array([0.45, 0])
b = np.array([1, 0])
n1 = 1
n2 = 1.5
print(refracted2(a, b, n1, n2))
