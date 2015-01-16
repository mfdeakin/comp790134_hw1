#!/usr/bin/python3
# My code was written for Python 3 and may (it does) have issues with Python 2

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math

def prob2():
    print('Problem 2, Part 1')
    numPoints = 10
    dataX = np.random.rand(numPoints)
    dataY = np.random.rand(numPoints)
    dataZ = np.random.random_integers(0, 1, numPoints) * 2 - 1
    data = list(zip(dataX, dataY, dataZ))
    plt.figure()
    plt.scatter(dataX, dataY, s = 64 * np.pi, c = dataZ, alpha = 0.5)
    plt.show()
    print(data)
    
    print('Problem 2, Part 2')
    # Now look at a large number of possible decision lines
    verticalPixels = 256
    horizontalPixels = verticalPixels
    heatmap01 = np.zeros((horizontalPixels, verticalPixels))
    heatmapHinge = np.zeros((horizontalPixels, verticalPixels))
    for w1 in np.linspace(-1, 1, verticalPixels):
        for w2 in np.linspace(-1, 1, horizontalPixels):
            if w1 == 0.0 and w2 == 0.0:
                # A bad decision line
                continue
            hPos = np.int(w2 * (horizontalPixels - 1) + 1e-4)
            vPos = np.int(((w1 + 1) * (verticalPixels - 1)) / 2 + 1e-4)
            (bBias, cost01) = optimizeBias01(data, [w1, w2])
            heatmap01[hPos][vPos] = cost01
            (_, costHinge) = optimizeBiasHinge(data, [w1, w2])
            heatmapHinge[hPos][vPos] = costHinge
    plt.clf()
    extents = [-1, 1, -1, 1]
    plt.figure()
    plt.imshow(heatmap01.T, extent = extents)
    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.linspace(0, 1, horizontalPixels),
                       np.linspace(-1, 1, verticalPixels))
    Z = np.array(heatmap01).reshape(X.shape)
    ax.plot_surface(X, Y, Z)
    plt.figure()
    plt.imshow(heatmapHinge.T, extent = extents)
    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')
    Z = np.array(heatmapHinge).reshape(X.shape)
    ax.plot_surface(X, Y, Z)
    plt.show()

def optimizeBiasHinge(data, decisionLine):
    # Check the locations of change in slope for the bias number
    # These do not correspond to the point locations, unfortunately
    # I believe this can also be done in O(n log(n)) time, but am out of time
    minCost = float('inf')
    minCostBias = float('inf')
    for biasPt in data:
        b = decisionLine[0] * biasPt[0] + decisionLine[1] * biasPt[1] - 1
        bCost = 0
        for pt in data:
            bCost += max(0, 1 - (decisionLine[0] * pt[0] -
                                 decisionLine[1] * pt[1] - b) * pt[2])
        if bCost < minCost:
            minCostBias = b
            minCost = bCost
    return (minCostBias, minCost)

def optimizeBias01(data, decisionLine):
    # We can simplify the determination of the bias by
    # determining the order of the points as seen by a sweep of the decision line
    # The dot product of the points positions with the vector perpendicular
    # to the decision line will give us such an order
    # This will allow us to use a dynamic programming algorithm
    # to compute the optimal bias for this
    # decision line in O(n log(n)) time, which is equivalent to our 1D case.
    dataOrder = sorted([(data[i][0] * decisionLine[0] +
                         data[i][1] * decisionLine[1], data[i][2])
                        for i in range(len(data))],
                       key = lambda d: d[0])
    totalPluses = sum(1 for d in data if d[2] == 1)
    totalMinuses = sum(1 for d in data if d[2] == -1)
    # The minimum index will point to the point containing the decision line
    # At -1, all points are on the positive side of the decision line
    # At n, all points are on the negative side of the decision line
    # The point pointed at by the index is on the negative side of the line
    minCostIndex = -1
    # Since all points are initially on the positive side of the decision line,
    # the initial cost is the number of -1 points
    minWrong = totalMinuses
    prevWrong = minWrong
    for k in range(len(data)):
        curWrong = prevWrong
        if dataOrder[1] == -1:
            curWrong -= 1
        else:
            curWrong += 1
        if curWrong < minWrong:
            minCostIndex = k
            minWrong = curWrong
        prevWrong = curWrong
    # Now we have the point containing the decision line,
    # so we can compute the bias
    bias = 0
    # Perturb bias to make it automatically satisfy the minimum requirement
    epsilon = 1e-5
    if minCostIndex >= 0:
        # Since the points are sorted in increasing order with respect
        # to their projection on the line, the bias will decrease for each point
        # Thus, to ensure we consider the decision line which has this
        # point in the negative side of the decision line
        bias = dataOrder[minCostIndex][0] - epsilon
    else:
        # If the minCostIndex is beyond all of the points,
        # look at the first point and determine the bias from that
        bias = dataOrder[0][0] + epsilon
    return (bias, minWrong)

if __name__ == '__main__':
    random.seed()
    prob2()
