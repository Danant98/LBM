#!/usr/bin/env python

# Importing libraries and modules
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from numba import njit, prange

np.random.seed(0)

# Defining constants 
nx, ny = 400, 100 # Number of nodes in each direction

tau = 0.55 # Collision timescale
rho0 = 0.5 # Average fluid density
u0 = 0.1 # Initial velocity in x-direction

nl = 9 # Number of lattice speed directions
N = 100 # Number of time steps

# Weights
w = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
# Directions for each node
e = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
# Oposite direction for each node
opposite = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

# Defining arrays for node velocity and density
u = np.zeros((ny, nx, 2))
u[:, :, 0] = u0  # Set initial velocity in x-direction
rho = np.ones((ny, nx)) * rho0

# f, distribution function for each node
f = np.ones((nl, ny, nx))
f += 0.01 * np.random.rand(nl, ny, nx)

# Creating a meshgrid for the simulation domain
X, Y = np.meshgrid(np.arange(nx), np.arange(ny))

# Defining a solid circular obstacle in the flow
R = 15 # Radius of the circular obstacle
solid = (X - nx // 4)**2 + (Y - ny // 2)**2 < R**2

solid[0, :] = False
solid[:, 0] = False
solid[-1, :] = False
solid[:, -1] = False

def feq(rho: np.ndarray, u: np.ndarray, e: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Function to calculate the equilibrium distribution function
    """
    cu = np.einsum('ia,yxa -> iyx', e, u)
    uu = np.sum(u**2, axis = -1)
    fe = w[:, None, None] * rho[None, :, :] * (1 + 3 * cu + 4.5 * cu ** 2 - 1.5 * uu[None, :, :])
    return fe

def macro(f: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to calculate the macroscopic variables (density and velocity)
    """
    r = np.sum(f, axis = 0)
    vel = np.einsum('ia,iyx -> yxa', c, f) / r[..., None]
    vel[solid] = 0 # Set velocity to zero in solid nodes
    return r, vel


def collide(f: np.ndarray, feq_: np.ndarray, tau: float) -> np.ndarray:
    """
    Function to perform the collision step in the Lattice Boltzmann method
    """
    return f - (f - feq_) / tau

@njit(parallel = True, fastmath = True)
def stream(f: np.ndarray, e: np.ndarray) -> np.ndarray:
    """
    Function to perform the streaming step in the Lattice Boltzmann method
    """
    for i in range(nl):
        ex = e[i, 0]
        ey = e[i, 1]
        f[i] = np.roll(f[i], shift = (ey, ex), axis = (0, 1))
    return f

@njit(parallel = True, fastmath = True)
def bounce_back(f: np.ndarray, solid: np.ndarray, opposite: np.ndarray) -> np.ndarray:
    """
    Function to apply the bounce-back boundary condition for solid nodes
    """
    f_old = f.copy()
    for i in range(nl):
        f[i, solid] = f_old[opposite[i], solid]
    return f


def main(f: np.ndarray) -> None:
    # Main function to run the Lattice Boltzmann simulation
    print("Starting simulation...")
    for t in range(N):
        # Streaming step
        stream(f, e)

        # Bounce-back boundary condition for solid nodes
        f = bounce_back(f, solid, opposite)

        # Calculate macroscopic variables
        rho, u = macro(f, e)

        # Collision step
        feq_ = feq(rho, u, e, w)
        f = collide(f, feq_, tau)

    print("Simulation completed.")



if __name__ == "__main__":
    main(f)


