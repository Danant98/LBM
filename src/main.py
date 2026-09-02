#!/usr/bin/env python

# Importing libraries and modules
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Defining constants 
nx, ny = 400, 100 # Number of nodes in each direction
tau = 0.55 # Collision timescale
rho0 = 0.5 # Average fluid density
nl = 9 # Number of lattice speed directions
N = 4000 # Number of time steps

# Weights
w = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
# Directions for each node
e = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
# Oposite direction for each node
opposite = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

# Defining arrays for node velocity and density
u = np.ones((ny, nx, 2))
rho = np.ones((ny, nx, 2))

# f, distribution function for each node
f = np.ones((nl, ny, nx))
f += 0.01 * np.random.rand(nl, ny, nx)

# Creating a meshgrid for the simulation domain
X, Y = np.meshgrid(np.arange(nx), np.arange(ny))

# Defining a solid circular obstacle in the flow
solid = (X - e[1, 0])**2 + (Y - e[1, 1])**2 < 20**2
solid[0, :] = False
solid[:, 0] = False
solid[-1, :] = False
solid[:, -1] = False

def feq(rho: np.ndarray, u: np.ndarray, e: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Function to calculate the equilibrium distribution function
    """
    cu = np.einsum('ia,xyb->xyi', e, u)
    uu = np.sum(u ** 2, axis = -1)
    fe = w[:, None, None] * rho[:, :, None] * (1 + 3 * cu + 4.5 * cu ** 2 - 1.5 * uu[:, :, None])
    return fe

def macro(f: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to calculate the macroscopic variables (density and velocity)
    """
    r = np.sum(f, axis = 0)
    vel = np.einsum('ia,xyi->xyb', c, f) / r[:, :, None]
    return r, vel


def collide(f: np.ndarray, feq_: np.ndarray, tau: float) -> np.ndarray:
    """
    Function to perform the collision step in the Lattice Boltzmann method
    """
    return f - (f - feq_) / tau

def main() -> None:
    # Main function to run the Lattice Boltzmann simulation
    for t in range(N):
        # Streaming step
        for i in range(nl):
            f[i] = np.roll(f[i], e[i], axis=(0, 1))

        # Bounce-back boundary condition for solid nodes
        for i in range(nl):
            f[i][solid] = f[opposite[i]][solid]

        # Calculate macroscopic variables
        rho[:, :], u[:, :] = macro(f, e)

        # Collision step
        feq_ = feq(rho, u, e, w)
        f = collide(f, feq_, tau)



if __name__ == "__main__":
    main()


