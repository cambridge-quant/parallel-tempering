"""
Parallel tempering algorithm implemented for the Gaussian spin glass model
Generalised to n dimensions
James Cummins
"""

import time
import numpy as np
from numba import jit
from itertools import product
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

# Initialise the seed
np.random.seed(0)

def initial_state(random_binary):
    """
    Produce the initial state

    Parameters
    ----------
    random_binary : NP.ARRAY
        Array of random binary (0, 1) values

    Returns
    -------
    state : NP.ARRAY
        Array of random Ising spins (-1, 1)

    """
    state = 2*random_binary - 1
    return state.astype(np.float64)

def positions(lattice_width, n_dimension):
    """
    Obtain coordinates of spins in the lattice

    Parameters
    ----------
    lattice_width : INT
        Width of lattice (that forms a n_dimension hypercube)
    n_dimension : INT
        Number of lattice dimensions

    Returns
    -------
    spin_positions : LIST
        List of the coordinates of spins

    """
    spin_positions = list(product(list(range(lattice_width)), repeat=n_dimension))
    return spin_positions

def nn_check(p1, p2, lattice_width, n_dimension):
    """
    Check if two positions in the lattice are nearest neighbours

    Parameters
    ----------
    p1 : NP.ARRAY
        Position 1
    p2 : NP.ARRAY
        Position 2
    lattice_width : INT
        Width of lattice
    n_dimension : INT
        Number of lattice dimensions

    Returns
    -------
    bool
        True or false depending on if positions are nearest neighbours

    """
    same_pos_check = sum(1 for i in range(n_dimension) if p1[i] == p2[i])
    if same_pos_check == n_dimension - 1:
        for j in range(n_dimension):
            if p1[j] != p2[j]:
                index = j
        if p2[index] == (p1[index]+1)%lattice_width or p2[index] == (p1[index]-1)%lattice_width:
            return True
    else:
        return False

def j_matrix(lattice_width, lattice_size, n_dimension, gaussian_numbers):
    """
    Produce the quadratic interaction (coupling) matrix J

    Parameters
    ----------
    lattice_width : INT
        Width of lattice
    lattice_size : INT
        Total size of lattice,
        equivalent to number of spins
    n_dimension : INT
        Number of lattice dimensions
    gaussian_numbers : NP.ARRAY
        Array of random numbers from a Gaussian (normal) distribution

    Returns
    -------
    j_mat : NP.ARRAY
        The coupling matrix J

    """
    j_mat = np.zeros((lattice_size, lattice_size), dtype=np.float64)
    position_list = positions(lattice_width, n_dimension)
    for i in range(lattice_size):
        for j in range(lattice_size):
            if i < j and nn_check(position_list[i], position_list[j], lattice_width, n_dimension):
                j_mat[i, j] = gaussian_numbers[i, j]
    return j_mat

def get_effective_fields(lattice_size, all_spins, sym_j_matrix, n_replicas):
    """
    Function for producing the effective linear fields

    Parameters
    ----------
    lattice_size : INT
        Size of the lattice
    all_spins : NP.ARRAY
        Array of Ising spins
    sym_j_matrix : NP.ARRAY
        Symmetric interaction matrix
    n_replicas : INT
        Number of replicas in the parallel tempering algorithm

    Returns
    -------
    field_array : NP.ARRAY
        The effective fields as an array

    """
    field_array = np.empty([n_replicas, lattice_size])
    for m_index in range(n_replicas):
        for spin_index in range(lattice_size):
            field_array[m_index, spin_index] = np.dot(sym_j_matrix[spin_index], all_spins[m_index])
    return field_array

@jit(nopython=True)
def sparse_mat_product(lattice_size, mat_data, mat_cols, mat_ind_ptr, spin_vector):
    """
    Function for quick matrix-vector multiplication
    Works well for sparse matrices

    Parameters
    ----------
    lattice_size : INT
        Size of lattice
    mat_data : NP.ARRAY
        Matrix element data
    mat_cols : NP.ARRAY
        Matrix column data
    mat_ind_ptr : NP.ARRAY
        Matrix index pointer array
    spin_vector : NP.ARRAY
        The vector

    Returns
    -------
    vector_result : NP.ARRAY
        The vector result

    """
    vector_result = np.zeros(lattice_size)
    for i in range(lattice_size):
        for row_index in range(mat_ind_ptr[i], mat_ind_ptr[i+1]):
            vector_result[i] += mat_data[row_index]*spin_vector[mat_cols[row_index]]
    return vector_result

@jit(nopython=True)
def energy_calculation(spin_array, efield_array):
    """
    Calculate the energy

    Parameters
    ----------
    spin_array : NP.ARRAY
        Spin vector
    efield_array : NP.ARRAY
        Effective field array

    Returns
    -------
    FLOAT
        The energy value

    """
    return - np.dot(spin_array, efield_array) * 0.5

@jit(nopython=True)
def magnetisation_calculation(spin_array):
    """
    Calculate the magnetisation of the system

    Parameters
    ----------
    spin_array : NP.ARRAY
        The vector of spins

    Returns
    -------
    FLOAT
        The magnetisation

    """
    return np.sum(spin_array)

@jit(nopython=True)
def update_efields(efields, all_spins, smat_data, smat_cols, smat_ind_ptr, m_index, spin_index):
    """
    Update the effective fields in one step of the PT algorithm

    Parameters
    ----------
    efields : NP.ARRAY
        The effective fields
    all_spins : NP.ARRAY
        All the spin vectors
    smat_data : NP.ARRAY
        Symmetric matrix entry data
    smat_cols : NP.ARRAY
        Symmetric matrix column data
    smat_ind_ptr : NP.ARRAY
        Symmetric matrix index pointer
    m_index : INT
        Replica index
    spin_index : INT
        Spin index

    Returns
    -------
    efields : NP.ARRAY
        The updated effective fields

    """
    data_for_row = smat_data[smat_ind_ptr[spin_index]:smat_ind_ptr[spin_index+1]]
    col_indices_for_row = smat_cols[smat_ind_ptr[spin_index]:smat_ind_ptr[spin_index+1]]
    data_index = 0
    for col_index in col_indices_for_row:
        efields[m_index, col_index] += 2*all_spins[m_index, spin_index]*data_for_row[data_index]
        data_index += 1
    return efields

@jit(nopython=True)
def metropolis_algorithm(lattice_size, all_spins, inv_t, efields, random_integers, random_floats, m_index, smat_data, smat_cols, smat_ind_ptr):
    """
    Perform the Metropolis algorithm

    Parameters
    ----------
    lattice_size : INT
        Size of lattice
    all_spins : NP.ARRAY
        All the spin vectors
    inv_t : NP.ARRAY
        Inverse temperature of replicas
    efields : NP.ARRAY
        Effective fields
    random_integers : NP.ARRAY
        Array of random integers
    random_floats : NP.ARRAY
        Array of random floats
    m_index : INT
        Replica index
    smat_data : NP.ARRAY
        Symmetric matrix entry data
    smat_cols : NP.ARRAY
        Symmetric matrix column data
    smat_ind_ptr : NP.ARRAY
        Symmetric matrix index pointer

    Returns
    -------
    all_spins : NP.ARRAY
        Updated array of all spin vectors

    """
    for spin_choice in range(lattice_size):
        spin_index = random_integers[m_index, spin_choice]
        dE = 2*all_spins[m_index, spin_index]*efields[m_index, spin_index]
        if random_floats[m_index, spin_choice] < np.exp(-dE*inv_t[m_index]):
            all_spins[m_index, spin_index] *= -1
            update_efields(efields, all_spins, smat_data, smat_cols, smat_ind_ptr, m_index, spin_index)
    return all_spins

#@jit(nopython=True)
def parallel_tempering(replica_indices, energy_array, inv_T, n_replicas, random_floats, n_sweep):
    """
    Parallel tempering algorithm

    Parameters
    ----------
    replica_indices : NP.ARRAY
        Indices of the replicas
    energy_array : NP.ARRAY
        The energy array
    inv_T : NP.ARRAY
        Array of inverse temperatures
    n_replicas : INT
        Number of replicas
    random_floats : NP.ARRAY
        Array of random floats
    n_sweep : INT
        Sweep number

    Returns
    -------
    acceptance_counter : INT
        Number of swaps made
    replica_indices : NP.ARRAY
        Updated index array

    """
    acceptance_counter = 0
    for i in range(n_replicas - 1):
        delta = (energy_array[replica_indices[i+1]] - energy_array[replica_indices[i]]) * (inv_T[i+1] - inv_T[i])
        if random_floats[n_sweep, i] < np.exp(delta):
            replica_indices[i+1], replica_indices[i] = replica_indices[i], replica_indices[i+1]
            acceptance_counter += 1
    return acceptance_counter, replica_indices

#@jit(nopython=True, parallel=True)
def metropolis_sweep(n_sweep, n_replicas, lattice_size, all_spins, invT, efields, random_integers, random_floats, replica_indices, energy_array, smat_data, smat_cols, smat_ind_ptr):
    """
    Perform one sweep of the Metropolis algorithm

    Parameters
    ----------
    n_sweep : INT
        Sweep number
    n_replicas : INT
        Number of replicas
    lattice_size : INT
        Size of the lattice
    all_spins : NP.ARRAY
        Array of all spin vectors
    invT : NP.ARRAY
        Array of inverse temperatures
    efields : NP.ARRAY
        The effective fields

    Returns
    -------
    all_spins : NP.ARRAY
        The updated array of all spins
    energy_array : NP.ARRAY
        The updated energy array

    """
    for replica_number in range(n_replicas): #prange(n_replicas):
        metropolis_algorithm(lattice_size, all_spins, invT, efields, random_integers[n_sweep], random_floats[n_sweep], replica_indices[replica_number], smat_data, smat_cols, smat_ind_ptr)
        energy_array[replica_indices[replica_number]] = energy_calculation(all_spins[replica_indices[replica_number]], efields[replica_indices[replica_number]])
    return all_spins, energy_array

#@jit(nopython=True)
def main_function(n_sweeps, n_replicas, lattice_size, all_spins, invT, efields, random_integers, random_floats, random_pt_numbers, replica_indices, energy_array, smat_data, smat_cols, smat_ind_ptr, mag_evo_array, ene_evo_array):
    """
    The main function to run to enact the parallel tempering algorithm

    Parameters
    ----------
    n_sweeps : INT
        Number of sweeps
    n_replicas : INT
        Number of replicas
    lattice_size : INT
        The lattice size
    all_spins : NP.ARRAY
        The array of all spins
    invT : NP.ARRAY
        Array of inverse temperatures
    efields : NP.ARRAY
        The effective fields
    replica_indices : NP.ARRAY
        The replica indices
    energy_array : NP.ARRAY
        The energy array

    Returns
    -------
    pt_acceptance_rate : FLOAT
        The total parallel tempering acceptance rates for replica swaps
    mag_evo_array : NP.ARRAY
        Evolution of the magnetisation
    ene_evo_array : NP.ARRAY
        Evolution of the energy
    all_spins : NP.ARRAY
        Final state of all the spins
    energy_array : NP.ARRAY
        The energy array

    """
    total_acceptance_counter = 0
    for count in range(n_sweeps):
        metropolis_sweep(count, n_replicas, lattice_size, all_spins, invT, efields, random_integers, random_floats, replica_indices, energy_array, smat_data, smat_cols, smat_ind_ptr)
        total_acceptance_counter += parallel_tempering(replica_indices, energy_array, invT, n_replicas, random_pt_numbers, count)[0]
        mag_evo_array[count + 1] = magnetisation_calculation(all_spins[replica_indices[0]])
        ene_evo_array[count + 1] = energy_array[replica_indices[0]]
        if (count + 1) % 10 == 0:
            print(count + 1, "sweeps completed.")
    pt_acceptance_rate = total_acceptance_counter / (n_sweeps * (n_replicas - 1))
    return pt_acceptance_rate, mag_evo_array, ene_evo_array, all_spins, energy_array

# Global variables
d      = 3    # Number of dimensions
L      = 6    # Width/length of simple d-dimensional lattice
N      = L**d # Total number of spins
M      = 16   # Number of replicas
SWEEPS = 100  # Length of time in units of MCS
TMIN   = 0.2  # Target temperature
TMAX   = 2.8  # Maximum temperature

# Empty arrays
MagEvo = np.empty(SWEEPS + 1)
EneEvo = np.empty(SWEEPS + 1)
Energies = np.empty(M)

# Random number arrays
Mean, StdDev = 0, 1
RandJMatArray = np.random.normal(Mean, StdDev, size=(N, N))
RandIntsArray = np.random.randint(0, N, size=(SWEEPS, M, N))
RandNumbArray = np.random.random(size=(SWEEPS, M, N))
RandPaTeArray = np.random.random(size=(SWEEPS, M-1))
RandSpinArray = np.random.randint(2, size=(M, N))

# Variable arrays
T = np.geomspace(TMIN, TMAX, M)
Beta = 1/T
RIndices = np.arange(M)

# Coulpling matrix
Interactions = j_matrix(L, N, d, RandJMatArray)
SymmetricInteractions = Interactions.transpose() + Interactions
CSRSymmetricInteractions = csr_matrix(SymmetricInteractions)
SMData = CSRSymmetricInteractions.data
SMCols = CSRSymmetricInteractions.indices
SMIndPtr = CSRSymmetricInteractions.indptr

# Initialise the state
SpinSystems = initial_state(RandSpinArray)
EffectiveFields = get_effective_fields(N, SpinSystems, SymmetricInteractions, M)
MagEvo[0] = magnetisation_calculation(SpinSystems[RIndices[0]])
EneEvo[0] = energy_calculation(SpinSystems[RIndices[0]], EffectiveFields[RIndices[0]])

# Main code
StartTime = time.perf_counter()
AcceptanceRate = main_function(SWEEPS, M, N, SpinSystems, Beta, EffectiveFields, RandIntsArray, RandNumbArray, RandPaTeArray, RIndices, Energies, SMData, SMCols, SMIndPtr, MagEvo, EneEvo)[0]
EndTime = time.perf_counter()
print("Code Time = {:.3f} seconds".format(EndTime - StartTime))
print("Parallel Tempering Acceptance Rate = {:.3f}".format(AcceptanceRate))

# Plot data
fig, ax1 = plt.subplots()
ax1.set_xlabel(r"$t$ $( \mathrm{MCS} )$", fontsize=20)
ax1.set_ylabel(r"$M (t) / N$", color="darkblue", fontsize=20)
ax1.plot(MagEvo/N, 'x-', color="darkblue", zorder=10)
ax1.tick_params(axis='y', labelcolor="darkblue")
ax2 = ax1.twinx()
ax2.set_ylabel(r"$E (t) / {}N$".format(d), color="crimson", fontsize=20)
ax2.plot(EneEvo/(d*N), '.-', color="crimson", zorder=0)
ax2.tick_params(axis='y', labelcolor="crimson")
fig.tight_layout()
plt.title("Gaussian Ising Spin Glass Model", y=1.1, fontsize=20)
plt.title("L = {}, T = {:.1f}, M = {}, PT Acceptance = {:.2f}".format(L, TMIN, M, AcceptanceRate), loc="left")
plt.savefig("L{}M{}_Evolution_EneMag.png".format(L, M), dpi=600, bbox_inches='tight')
plt.show()
plt.clf()