#!/usr/bin/python3
# My code was written for Python 3 and may (it does) have issues with Python 2

import numpy as np
import matplotlib.pyplot as plt
import random

def prob2():
    print('Problem 2, Part 1')
    numPoints = 10
    dataX = np.random.rand(numPoints)
    dataY = np.random.rand(numPoints)
    dataZ = np.random.random_integers(0, 1, numPoints) * 2 - 1
    plt.scatter(dataX, dataY, s = 64 * np.pi, c = dataZ, alpha = 0.5)
    plt.show()
    data = list(zip(dataX, dataY, dataZ))
    print(data)
    
    print('Problem 2, Part 2')
    # Now look at a large number of possible decision lines
    verticalPixels = 512
    horizontalPixels = verticalPixels // 2
    heatmap = np.zeros((horizontalPixels, verticalPixels))
    for w1 in np.linspace(-1, 1, verticalPixels):
        for w2 in np.linspace(0, 1, horizontalPixels):
            if w1 == 0.0 and w2 == 0.0:
                # A bad decision line
                continue
            (b, cost) = optimizeBoas(data, [w1, w2])
            hPos = np.int(w2 * (horizontalPixels - 1) + 1e-4)
            vPos = np.int(((w1 + 1) * (verticalPixels - 1)) / 2 + 1e-4)
            heatmap[hPos][vPos] = cost
    plt.clf()
    extents = [0, 1, -1, 1]
    plt.imshow(heatmap.T, extent = extents)
    plt.show()

def optimizeBoas(data, decisionLine):
    # We can simplify the determination of the boas number by
    # determining the order of the points as seen by a sweep of the decision line
    # The dot product of the points positions with the vector perpendicular
    # to the decision line will give us such an order
    # This will allow us to use a dynamic programming algorithm
    # to compute the optimal boas number for this
    # decision line in O(n log(n)) time, which is equivalent to our 1D case.
    perpDir = [-decisionLine[1], decisionLine[0]]
    dataOrder = sorted([(i, data[i][0] * perpDir[0] + data[i][1] * perpDir[1])
                        for i in range(len(data))],
                       key = lambda d: d[1])
    totalPluses = sum(1 for d in data if d[2] == 1)
    totalMinuses = sum(1 for d in data if d[2] == -1)
    # The minimum index will point to the point containing the decision line
    # We will define points one the decision line as being on the minus side
    minCostIndex = -1
    minWrong = totalMinuses
    prevWrong = minWrong
    for k in range(1, len(data)):
        point = data[dataOrder[k][0]]
        curWrong = prevWrong
        if point[2] == -1:
            curWrong -= 1
        else:
            curWrong += 1
        if curWrong < minWrong:
            minCostIndex = k
            minWrong = curWrong
        prevWrong = curWrong
    # Now we have the point containing the decision line,
    # so we can compute the Boas number
    point = None
    if minCostIndex >= 0:
        point = data[minCostIndex]
    else:
        # If the minCostIndex is beyond all of the points,
        # look at the first point and determine the Boas number from that
        point = data[dataOrder[0][0]]
    b = -point[0] * decisionLine[0] - point[1] * decisionLine[1]
    # Perturb Boas number to make it automatically satisfy the minimum requirement
    epsilon = 1e-9
    if minCostIndex == -1:
        # Perturb b so the point is in the plus side
        b += epsilon
    else:
        # Perturb b so the point is in the minus side
        b -= epsilon
    return (b, minWrong)

if __name__ == '__main__':
    random.seed()
    prob2()
