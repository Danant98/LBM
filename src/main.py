#!/usr/bin/env python

# Importing libraries and modules
import time, sys
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from numba import njit, prange


# Defining constants 
nx, ny = 400, 100 # Number of nodes in each direction

tau = 0.55 # Collision timescale
rho0 = 0.5 # Average fluid density
u0 = 0.1 # Initial velocity in x-direction

nl = 9 # Number of lattice speed directions
N = 1000 # Number of time steps

# Weights
w = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
# Directions for each node
e = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
# Oposite direction for each node
opposite = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

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
    fe = w[:, None, None] * rho[None, ...] * (1 + 3 * cu + 4.5 * cu ** 2 - 1.5 * uu[None, ...])
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

def stream(f: np.ndarray, e: np.ndarray) -> np.ndarray:
    """
    Function to perform the streaming step in the Lattice Boltzmann method
    """
    for i in range(nl):
        ex = e[i, 0]
        ey = e[i, 1]
        f[i] = np.roll(f[i], shift = (ey, ex), axis = (0, 1))
    return f

def bounce_back(f: np.ndarray, solid: np.ndarray, opposite: np.ndarray) -> np.ndarray:
    """
    Function to apply the bounce-back boundary condition for solid nodes
    """
    f_old = f.copy()
    for i in range(nl):
        f[i, solid] = f_old[opposite[i], solid]
    return f

def step(f: np.ndarray) -> np.ndarray:
    """
    Logic of the LBM simulation for a single time step, including streaming, collision, and boundary conditions.
    """
    # Streaming step
    f = stream(f, e)

    # Bounce-back boundary condition for solid nodes
    f = bounce_back(f, solid, opposite)

    # Calculate macroscopic variables
    rho, u = macro(f, e)

    # Collision step
    feq_ = feq(rho, u, e, w)
    f = collide(f, feq_, tau)

    return f

# Velocity field
u = np.zeros((ny, nx, 2)) 
u[..., 0] = u0 # Set initial velocity in x-direction

rho = np.full((ny, nx), rho0) # Set initial density

f = feq(rho, u, e, w) # Initialize the distribution function

def benchmark(f: np.ndarray, N: int) -> None:
    print("Starting simulation...")
    start = time.perf_counter()
    for _ in tqdm(range(N), desc = "Time steps"):
        f = step(f)

    print("Simulation completed.")
    end = time.perf_counter()
    print(f"Simulation time: {end - start:.2f} seconds")
    print(f"Average time per step: {(end - start) / N:.4f} seconds")
    print(f"Steps per second: {N / (end - start):.2f}")

    mlups = N * nx * ny / ((end - start) * 1e6)
    print(f"Performance: {mlups:.2f} MLUPS")

def visualize(f: np.ndarray, N: int, steps_per_frame: int = 10, fps: int = 30) -> None:
    """
    Function to visualize the simulation using matplotlib
    """
    # Number of frames for the animation
    num_frames = N // steps_per_frame
    interval = 1000 / fps

    # Initial macroscopic variables
    rho, u = macro(f, e)

    # Initial velocity magnitude and vorticity for visualization
    vel_mag = np.linalg.norm(u, axis = -1)
    omega = np.gradient(u[..., 1], axis = 1) - np.gradient(u[..., 0], axis = 0)
    norm = TwoSlopeNorm(vmin = -max(omega.max(), 1e-12), vcenter = 0.0, vmax = max(omega.max(), 1e-12))

    # Mask for the solid obstacle
    vel_mag = np.ma.masked_where(solid, vel_mag)
    omega = np.ma.masked_where(solid, omega)

    # Create figure and axis for animation
    fig, (ax_vel, ax2_vort) = plt.subplots(2, 1, figsize = (12, 5))

    # Plotting the velocity
    vel_plot = ax_vel.imshow(vel_mag, cmap = 'jet', aspect = 'auto', vmin=0.0, vmax = max(vel_mag.max(), 1e-12), origin = 'lower')
    ax_vel.contour(solid, colors = 'black', levels = [0.5], linewidths = 1.5)
    vel_colorbar = fig.colorbar(vel_plot, ax = ax_vel)
    vel_colorbar.set_label('Velocity Magnitude', rotation = 270, labelpad = 15)
    ax_vel.set_xlabel('x')
    ax_vel.set_ylabel('y')

    # Plotting the vorticity
    vort_plot = ax2_vort.imshow(omega, cmap = 'RdBu_r', aspect = 'auto', 
                                origin = 'lower', norm = norm)
    ax2_vort.contour(solid, colors = 'black', levels = [0.5], linewidths = 1.5)
    vort_colorbar = fig.colorbar(vort_plot, ax = ax2_vort)
    vort_colorbar.set_label('Vorticity', rotation = 270, labelpad = 15)
    ax2_vort.set_xlabel('x')
    ax2_vort.set_ylabel('y')

    def update(frame: int) -> None:
        nonlocal f
        for _ in range(steps_per_frame):
            f = step(f)

        current_step = (frame + 1) * steps_per_frame

        # Update macroscopic variables
        rho, u = macro(f, e)

        # Update velocity magnitude and vorticity for visualization
        vel_mag = np.linalg.norm(u, axis = -1)  
        vel_mag = np.ma.masked_where(solid, vel_mag)
        vel_plot.set_data(vel_mag)
        if vel_mag.max() > 0:
            vel_plot.set_clim(vmin = 0, vmax = vel_mag.max())

        omega = np.gradient(u[..., 1], axis = 1) - np.gradient(u[..., 0], axis = 0)
        omega = np.ma.masked_where(solid, omega)
        vort_plot.set_data(omega)
        if omega.max() > 0:
            vort_plot.set_clim(vmin = -omega.max(), vmax = omega.max())

        return vel_plot, vort_plot
    ani = FuncAnimation(fig, update, frames = num_frames, interval = interval, blit = True, repeat = False)
    plt.tight_layout()
    plt.show()


def main(f: np.ndarray, run: str, N: int) -> None:
    # Main function to run the Lattice Boltzmann simulation
    assert run in ["test", "visualize"], "Invalid run argument. Use 'test' or 'visualize'."
    assert N is not None and N > 0, "N must be provided as a positive integer."

    if run == "test":
        benchmark(f, N)
    elif run == "visualize":
        visualize(f, N, steps_per_frame = 2)


if __name__ == "__main__":
    # Getting the run argument from command line
    run = sys.argv[1] if len(sys.argv) > 1 else None
    N = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(f, run, N)
