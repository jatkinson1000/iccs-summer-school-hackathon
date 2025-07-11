"""Generate observation from L96 model and save as NetCDF"""

from pathlib import Path
from typing import Tuple
import numpy.typing as npt
import numpy as np
import netCDF4 as nc
from L96_model import L96


def generate_obs(
    K: int = 8,
    J: int = 32,
    forcing: float = 18,
    dt: np.float32 = 0.01,
    timesteps: int = 20000,
    output_file: str | Path = "l96_obs.nc",
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Generate observations from L96 model and save to NetCDF file.

    Parameters:
    -----------
    K : int
        Number of X variables
    J : int
        Number of Y variables
    forcing : float
        Forcing parameter
    dt : float
        Time step
    timesteps : int
        Number of time steps
    output_file : str
        Output NetCDF file name

    Returns:
    --------
    X_true : ndarray
        True state trajectory
    observations : ndarray
        Observations (same as trajectory, no noise added)
    """

    total_time = timesteps * dt

    # Instantiate Lorenz 96 with chosen parameters
    l96_model = L96(K=K, J=J, F=forcing)

    # Generate model trajectory
    X_true, _, _, xy_true = l96_model.run(
        dt, total_time, store=True, return_coupling=True
    )

    # Change data type to float32
    X_true = X_true.astype(np.float32)
    xy_true = xy_true.astype(np.float32)

    # Generate observations along a trajectory
    nsteps = X_true.shape[0]
    dim = X_true.shape[1]  # Should be K

    # Initialize observations array (excluding first time step)
    observations = np.zeros((nsteps - 1, dim), dtype=np.float32)

    print("***************************************************")
    print("*** Generate observations from model trajectory ***")
    print("***************************************************")
    print()
    print(f"Write observations to file: {output_file}")
    print()
    print("Dimensions of experiment:")
    print(f"     state dimension      {dim}")
    print(f"           time steps      {nsteps}")
    print()
    print("Values along a trajectory")
    print()

    # Copy trajectory values for each time step (starting from step 2, index 1)
    for i in range(1, nsteps):
        print(f"Generate observations for time {i * dt:.3e} step {i + 1}")

        # Get state at current time step (no noise added)
        observations[i - 1, :] = X_true[i, :]

    # Save to NetCDF file
    save_observations_netcdf(observations, dt, timesteps, dim, 0.0, output_file)

    print("------- END -------------")
    print()

    return X_true, observations


def save_observations_netcdf(
    observations: npt.NDArray,
    dt: float,
    timesteps: int,
    dim: int,
    stderr: float,
    filename: str,
) -> None:
    """
    Save observations to NetCDF file in the same format as needed in Fortran by PDAF.
    See https://github.com/PDAF/PDAF/blob/master/models/lorenz96/tools/generate_obs.F90.

    Parameters:
    -----------
    observations : ndarray
        Observation data (nsteps-1, dim)
    dt : float
        Time step
    timesteps : int
        Total number of time steps
    dim : int
        State dimension
    stderr : float
        Standard deviation of observation error (set to 0.0 for no noise)
    filename : str
        Output file name
    """

    nsteps_obs = observations.shape[0]  # nsteps - 1

    # Create time and step arrays (starting from step 2)
    times = np.arange(1, nsteps_obs + 1) * dt  # Skip first time step
    steps = np.arange(2, nsteps_obs + 2)  # Steps 2 to nsteps_obs + 1

    # Create NetCDF file
    with nc.Dataset(filename, "w", format="NETCDF4") as ncfile:
        # Global attributes
        ncfile.title = "Synthetic observations from Lorenz96 experiment"

        # Define dimensions
        ncfile.createDimension("dim_state", dim)
        ncfile.createDimension("one", 1)
        ncfile.createDimension("timesteps", nsteps_obs)

        # Define variables
        stderr_var = ncfile.createVariable("stderr", "f8", ("one",))
        step_var = ncfile.createVariable("step", "i4", ("timesteps",))
        time_var = ncfile.createVariable("time", "f8", ("timesteps",))
        obs_var = ncfile.createVariable("obs", "f8", ("dim_state", "timesteps"))

        # Write data
        stderr_var[:] = stderr
        step_var[:] = steps
        time_var[:] = times
        obs_var[:, :] = observations.T  # Transpose for Fortran-style ordering

        # Add variable attributes if needed
        stderr_var.long_name = "Standard deviation of observation error"
        step_var.long_name = "Time step number"
        time_var.long_name = "Time"
        obs_var.long_name = "Observations"


def get_root_path() -> Path:
    """Find the root directory of the project.

    Returns
    -------
    Path
        Path object pointing to the project root directory
    """
    current = Path.cwd()
    for directory in [current, *current.parents]:
        if any((directory / marker).exists() for marker in ["pyproject.toml", ".git"]):
            return directory
    return current


root_path = get_root_path()
if __name__ == "__main__":
    data_path = root_path / "data" / "l96_obs.nc"
    # Generate observations with default parameters
    # This will create obs.nc file with trajectory values (no noise)
    x_true, observations = generate_obs(
        K=8, J=32, forcing=18, dt=0.01, timesteps=20000, output_file=data_path
    )

    print(f"Generated observations shape: {observations.shape}")
    print(f"True trajectory shape: {x_true.shape}")
