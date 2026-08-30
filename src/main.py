#!/usr/bin/env python

# Importing libraries and modules
import numpy as np
import matplotlib.pyplot as plt

# Defining constants 
nx, ny = 400, 100 # Number of nodes in each direction
tau = 0.55 

# Weights
w = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
# Directions for each node
e = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
# Oposite direction for each node
opposite = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

# Defining arrays for node velocity and density
u = np.ones((nx, ny, 2))
rho = np.ones((nx, ny, 2))

# f, distribution function for each 
f = np.zeros((nx, ny, 9))

# TO BE IMPLEMENTED



if __name__ == "__main__":
    pass


