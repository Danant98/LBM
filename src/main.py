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
    # Computing the fluid density at each node
    r = np.sum(f, axis = 0)

    # Checking flud density in 'bad' regions
    bad = (~solid) & (rho <= 0) | ~np.isfinite(rho)

    if np.any(bad):
        y, x = np.argwhere(bad)[0]
        raise RuntimeError(f"Bad density at ({x}, {y}): {r[y, x]}")

    # Computing fluid momentum
    momentum = np.einsum('ia,iyx -> yxa', c, f)

    # Initializing array for fluid velocity at each node
    vel = np.zeros_like(momentum)

    # Defining fluid nodes, excluding solids
    fluid = ~solid

    # Computing fluid velocity at each node
    vel[fluid] = momentum[fluid] / r[fluid, None]

    return r, vel

def collide(f: np.ndarray, feq_: np.ndarray, tau: float) -> np.ndarray:
    """
    Function to perform the collision step in the Lattice Boltzmann method
    """
    return f - (f - feq_) / tau

@njit
def stream_bounce_back(f:np.ndarray, e: np.ndarray, solid: np.ndarray, opposite: np.ndarray) -> np.ndarray:
    """
    Function to perform the streaming step and bounce-back boundary condition in the Lattice Boltzmann method
    """
    nl, ny, nx = f.shape

    f_new = np.zeros_like(f)

    for i in range(nl):
        for y in range(ny):
            for x in range(nx):

                # Skipping solid nodes
                if solid[y, x]:
                    continue
                # x and y coordinates of the neighboring node in the direction of e[i]
                xnew = x + e[i, 0]
                # Periodic boundary conditions in y-direction
                ynew = (y + e[i, 1]) % ny

                if xnew < 0 or xnew >= nx:
                  continue  # Skip if the new x-coordinate is out of bounds  

                if solid[ynew, xnew]:
                    f_new[opposite[i], y, x] += f[i, y, x] # Bounce-back for solid nodes
                else:
                    f_new[i, ynew, xnew] = f[i, y, x] # Streaming for fluid nodes

    return f_new

# def stream(f: np.ndarray, e: np.ndarray) -> np.ndarray:
#     """
#     Function to perform the streaming step in the Lattice Boltzmann method
#     """
#     nl, ny, nx = f.shape

#     f_stream = np.zeros_like(f)

#     for i in range(nl):
#         shifted = np.roll(f[i], shift = e[i, 1], axis = 0)

#         if e[i, 0] == 1:
#             f_stream[i, :, 1:] = shifted[:, :-1]
#         elif e[i, 0] == -1:
#             f_stream[i, :, :-1] = shifted[:, 1:]
#         else:
#             f_stream[i] = shifted    
#     return f_stream

# def bounce_back(f: np.ndarray, solid: np.ndarray, opposite: np.ndarray) -> np.ndarray:
#     """
#     Function to apply the bounce-back boundary condition for solid nodes
#     """
#     f_old = f.copy()
#     for i in range(nl):
#         f[i, solid] = f_old[opposite[i], solid]
#     return f

def inlet_zou_he(f: np.ndarray, u0: float) -> np.ndarray:
    """
    Function to apply the Zou-He boundary condition at the inlet (left boundary)
    """
    f0 = f[0, :, 0]
    f2 = f[2, :, 0]
    f3 = f[3, :, 0]
    f4 = f[4, :, 0]
    f6 = f[6, :, 0]
    f7 = f[7, :, 0]

    # Calculating the density at the inlet using the Zou-He boundary condition
    rho = (f0 + f2 + f4 + 2 * (f3 + f6 + f7)) / (1 - u0)

    # Reconstructing the unknown distribution functions at the inlet
    f[1, :, 0] = f3 + (2 / 3) * rho * u0

    # Reconstructing the diagonal distribution functions at the inlet
    f[5, :, 0] = f7 + 0.5 * (f4 - f2) + (1 / 6) * rho * u0
    f[8, :, 0] = f6 + 0.5 * (f2 - f4) + (1 / 6) * rho * u0

    return f
    

def outlet_zou_he(f: np.ndarray, rho_out: np.ndarray) -> np.ndarray:
    """
    Function to apply the Zou-He boundary condition at the outlet (right boundary)
    """
    f0 = f[0, :, -1]
    f1 = f[1, :, -1]
    f2 = f[2, :, -1]
    f4 = f[4, :, -1]
    f5 = f[5, :, -1]
    f8 = f[8, :, -1]

    # Calculating the velocity at the outlet using the Zou-He boundary condition
    ux = -1 + (f0 + f2 + f4 + 2 * (f1 + f5 + f8)) / rho_out

    # Reconstructing the unknown distribution functions at the outlet
    f[3, :, -1] = f1 - (2 / 3) * rho_out * ux
    f[6, :, -1] = f8 + 0.5 * (f4 - f2) - (1 / 6) * rho_out * ux
    f[7, :, -1] = f5 + 0.5 * (f2 - f4) - (1 / 6) * rho_out * ux

    return f


def step(f: np.ndarray) -> np.ndarray:
    """
    Logic of the LBM simulation for a single time step, including streaming, collision, and boundary conditions.
    """
    # Calculate macroscopic variables
    rho, u = macro(f, e)

    # Collision step
    feq_ = feq(rho, u, e, w)
    f = collide(f, feq_, tau)

    # # Streaming step
    f = stream_bounce_back(f, e, solid, opposite)

    # f = stream(f, e)

    # # Bounce-back boundary condition for solid nodes
    # f = bounce_back(f, solid, opposite)

    # Zou-He boundary condition at the inlet (left boundary)
    f = inlet_zou_he(f, u0)

    # Zou-He boundary condition at the outlet (right boundary)
    f = outlet_zou_he(f, rho0)

    return f

# Zero-mean perturbation
rng = np.random.default_rng(0)
# Velocity field
u = np.zeros((ny, nx, 2)) 
u[..., 0] = u0 # Set initial velocity in x-direction
u[..., 1] = 1e-4 * rng.standard_normal((ny, nx)) # Add small random perturbation in y-direction

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

def visualize(f: np.ndarray, N: int, steps_per_frame: int = 10, fps: int = 30, save_gif: bool = False) -> None:
    """
    Function to visualize the simulation using matplotlib
    """
    # Number of frames for the animation
    num_frames = int(np.ceil(N / steps_per_frame))
    interval = 1000 / fps

    # Initial macroscopic variables
    rho, u = macro(f, e)

    # Initial velocity magnitude and vorticity for visualization
    vel_mag = np.linalg.norm(u, axis = -1)
    omega = np.gradient(u[..., 1], axis = 1) - np.gradient(u[..., 0], axis = 0)
    norm = TwoSlopeNorm(vmin = -max(np.max(np.abs(omega)), 1e-12), vcenter = 0.0, vmax = max(np.max(np.abs(omega)), 1e-12))

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
        remaining = N - frame * steps_per_frame
        n_steps = min(steps_per_frame, remaining)
        for _ in range(n_steps):
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
            vort_plot.set_clim(vmin = -np.max(np.abs(omega)), vmax = np.max(np.abs(omega)))

        return vel_plot, vort_plot
    ani = FuncAnimation(fig, update, frames = num_frames, interval = interval, blit = True, repeat = False)
    plt.tight_layout()
    if save_gif:
        ani.save('lbm_simulation.gif', writer = 'pillow', fps = fps)
    plt.show()


def main(f: np.ndarray, run: str, N: int, save_gif: bool = False) -> None:
    # Main function to run the Lattice Boltzmann simulation
    assert run in ["test", "visualize"], "Invalid run argument. Use 'test' or 'visualize'."
    assert N is not None and N > 0, "N must be provided as a positive integer."

    if run == "test":
        benchmark(f, N)
    elif run == "visualize":
        visualize(f, N, steps_per_frame = 2, save_gif = save_gif)


if __name__ == "__main__":
    # Getting the run argument from command line
    run = sys.argv[1] if len(sys.argv) > 1 else None
    N = int(sys.argv[2]) if len(sys.argv) > 2 else None
    save_gif = len(sys.argv) > 3 and sys.argv[3] == "save_gif"
    main(f, run, N, save_gif)
