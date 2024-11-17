"""
So far, just a place to test modules.
"""

from scipy.interpolate import interp1d
import time

import jax.numpy as jnp
from jax import config, jit, random, vmap, grad

from difflexmm.dynamics import setup_dynamic_solver
from difflexmm.energy import build_strain_energy, ligament_energy, build_contact_energy, combine_block_energies
from difflexmm.kinematics import _block_to_node_displacement
from difflexmm.geometry import QuadGeometry, compute_inertia, DOFsInfo
from difflexmm.utils import (ControlParams, GeometricalParams,
                             LigamentParams, MechanicalParams, ContactParams,
                             SolutionData, save_data, load_data)
from difflexmm.plotting import generate_animation, plot_geometry, generate_frames
import os
import subprocess
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
config.update("jax_enable_x64", True)  # enable float64 type

###############################################################################
# Convenience functions


def rc_layout():
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['font.size'] = 9
    plt.rcParams['legend.edgecolor'] = '1'
    plt.rcParams['pdf.fonttype'] = 42


###############################################################################
# user variables
rc_layout()
print('simulation started at ', time.ctime())
# sys.exit()

hingetype = 'T1r'
foldersuffix = 'disp_cycle_v3'
my_shifts = load_data(r'Maha_morpher_shifts_quad10mm.pkl')
shifts = [ms/1000. for ms in my_shifts]  # units: m
block_side_length = 10./1000  # mm, hardcoded!
bondlengthfactor = 0.05
bond_length = bondlengthfactor*block_side_length
Nx = 8
Ny = 8
density = 1200  # Zhermack Elite Double 32 mass density 1.2 g/cm^3 = 1200 kg/m^3
# actuation point between node center and vertex
torque_arm_adjustment_factor = 0.6
rotfactor = 1.0
# rotfactor = 0.63
shearfactor = 0.63
stretchfactor = 0.63
make_animation = True
dampingprefactor = 1.
init_angle_overshoot = -0.1
final_angle_overshoot = 0.1
# stiffness values
if hingetype == 'T1r':
    # [k_stretch, k_shear, k_rot] = [75000, 25000, 0.015e-3] #axial, shear, bending; N and Nm, for 32-A rubber with nylon organza hinge. avg highest values.
    # axial, shear, bending; N and Nm, for 32-A rubber with nylon organza hinge. intermediate values. v1 #2200 s rzuntime
    [k_stretch, k_shear, k_rot] = [10000, 2500, 0.015e-3]
    # axial, shear, bending; N and Nm, for 32-A rubber with nylon organza hinge. intermediate values. v2 #4000 s runtime
    [k_stretch, k_shear, k_rot] = [30000, 2500, 0.015e-3]
    # axial, shear, bending; N and Nm, for 32-A rubber with nylon organza hinge. intermediate values. v3 # 2323s runtime
    [k_stretch, k_shear, k_rot] = [10000, 5000, 0.015e-3]
    # [k_stretch, k_shear, k_rot] = [5000, 1000, 0.015e-3] #axial, shear, bending; N and Nm, for 32-A rubber with nylon organza hinge. lowest values.
    k_shear *= shearfactor
    k_stretch *= stretchfactor
    k_rot *= rotfactor
    amplitude_1 = -2/torque_arm_adjustment_factor*0.9
    amplitude_1 = -7.3
    amplitude_2 = 9
    dampingprefactor = 1.
if hingetype == 'T1':
    # axial, shear, bending; N and Nm, for 32-A rubber with nylon organza hinge.
    [k_stretch, k_shear, k_rot] = [2400, 400, 0.05e-3]
    k_shear *= shearfactor
    k_stretch *= stretchfactor
    k_rot *= rotfactor
    amplitude_1 = -2.2
    amplitude_2 = 1
elif hingetype == 'L2':
    # axial, shear, bending; N and Nm, for 32-A rubber with living hinge L2
    [k_stretch, k_shear, k_rot] = [1400, 350, 0.6e-3]
    k_shear *= shearfactor
    k_stretch *= stretchfactor
    k_rot *= rotfactor
    amplitude_1 = -4.5
    amplitude_2 = 1

# contact parameters
k_contact = 100*k_rot
min_angle = 0./180*jnp.pi
cutoff_angle = 5./180*jnp.pi

# loading params
loading_rate = 0.01
simulation_time = loading_rate**-1 * \
    (jnp.abs(amplitude_1) + jnp.abs(amplitude_2-amplitude_1) + jnp.abs(amplitude_2))
n_timepoints = 200

datafolder = "./data_hinge-{:s}_arm-{:.2f}_shf-{:.2f}_strf-{:.2f}_rotf-{:.2f}_bl-{:.2f}_{:s}".format(
    hingetype, torque_arm_adjustment_factor, shearfactor, stretchfactor, rotfactor, bondlengthfactor, foldersuffix)
if not os.path.exists(datafolder):
    os.mkdir(datafolder)

###############################################################################
# Define geometry
squares_zero = QuadGeometry(n1_blocks=Nx, n2_blocks=Ny,
                            spacing=block_side_length*jnp.sqrt(2), bond_length=0.)
block_centroids_zero, centroid_node_vectors_zero, bond_connectivity, reference_bond_vectors_zero = squares_zero.get_parametrization()
squares = QuadGeometry(n1_blocks=Nx, n2_blocks=Ny,
                       spacing=block_side_length*jnp.sqrt(2), bond_length=bond_length*jnp.sqrt(2))
block_centroids, centroid_node_vectors, bond_connectivity, reference_bond_vectors = squares.get_parametrization()
plot_geometry(block_centroids(*shifts),
              centroid_node_vectors(*shifts), bond_connectivity())

# Compute inertia of the blocks
inertia = compute_inertia(
    vertices=centroid_node_vectors(*shifts), density=density)

# Construct energy
strain_energy = build_strain_energy(
    bond_connectivity=bond_connectivity(), bond_energy_fn=ligament_energy)
contact_energy = build_contact_energy(bond_connectivity=bond_connectivity())
potential_energy = combine_block_energies(strain_energy, contact_energy)

###############################################################################
# construct actuation with force control as per Gio
# define clamping
clamped_block_id = 28  # Vertex id to clamp where the applied force is directed to
clamped_block_centroid = block_centroids(*shifts)[clamped_block_id]
clamped_centroid_node_vector_id = 2
loaded_block_id = 35  # Block id to load
loaded_centroid_node_vector_id = 0  # Vertex id to load
loaded_block_centroid = block_centroids(*shifts)[loaded_block_id]

constrained_block_DOF_pairs = jnp.array([
    [clamped_block_id, 0],  # Clamped unit
    [clamped_block_id, 1],  # Clamped unit
    [clamped_block_id, 2],  # Clamped unit
    [loaded_block_id, 0],
    [loaded_block_id, 1],
    [loaded_block_id, 2]
    # [loaded_block_id, 2],
])

timepoints = jnp.linspace(0, simulation_time, n_timepoints)


def loaded_block_displacement(t):  # t varies from 0 to total_time,
    # construct actuation with displacement control as per mechanism limit

    pos_cl = block_centroids_zero(*shifts)[loaded_block_id]
    vec_vl0 = centroid_node_vectors_zero(*shifts)[loaded_block_id, 0]
    pos_nl0 = pos_cl + vec_vl0

    pos_cc = block_centroids_zero(*shifts)[clamped_block_id]
    vec_vc1 = centroid_node_vectors_zero(*shifts)[clamped_block_id, 1]
    vec_vc2 = centroid_node_vectors_zero(*shifts)[clamped_block_id, 2]
    pos_nc1 = pos_cc + vec_vc1
    pos_nc2 = pos_cc + vec_vc2

    uvec_initdirection = (pos_nl0 - pos_nc1)/jnp.linalg.norm(pos_nl0 - pos_nc1)
    uvec_finaldirection = (pos_nc2 - pos_nc1) / \
        jnp.linalg.norm(pos_nc2 - pos_nc1)
    init_angle = jnp.arctan2(uvec_initdirection[1], uvec_initdirection[0])
    final_angle = 2*jnp.pi + \
        jnp.arctan2(uvec_finaldirection[1], uvec_finaldirection[0])
    edgelength = jnp.linalg.norm(pos_nc1 - pos_nl0)

    current_angle = init_angle*(1-t/simulation_time) + \
        final_angle*t/simulation_time  # linear angle ramp

    def angle_cycle(t, amplitude_1, amplitude_2, loading_rate):
        # Cycle: 0 → amplitude_1 → amplitude_2 → 0
        return jnp.where(
            t < loading_rate**-1*jnp.abs(amplitude_1),
            t * loading_rate*jnp.sign(amplitude_1),
            jnp.where(
                t < loading_rate**-1 *
                (jnp.abs(amplitude_1)+jnp.abs(amplitude_2-amplitude_1)),
                amplitude_1 + (t*loading_rate-jnp.abs(amplitude_1)
                               )*jnp.sign(amplitude_2-amplitude_1),
                jnp.where(
                    t < loading_rate**-1 *
                    (jnp.abs(amplitude_1)+jnp.abs(amplitude_2 -
                                                  amplitude_1)+jnp.abs(amplitude_2)),
                    amplitude_2 + (t*loading_rate-(jnp.abs(amplitude_1)+jnp.abs(
                        amplitude_2-amplitude_1)))*jnp.sign(amplitude_1-amplitude_2),
                    0.,
                )
            )
        )

    # cyclic angle ramp

    max_angle = final_angle + final_angle_overshoot
    min_angle = init_angle + init_angle_overshoot
    total_angle_range = jnp.abs(max_angle - min_angle)
    angle_slope = total_angle_range*2/simulation_time

    # t_switch_1 = simulation_time/2 * jnp.abs(init_angle_overshoot)/total_angle_range
    # t_switch_2 = simulation_time/2 + t_switch_1
    # # print(t_switch_1, t_switch_2)
    # if t < t_switch_1:
    #     current_angle = -angle_slope*t + init_angle
    # elif t < t_switch_2:
    #     current_angle = min_angle + angle_slope*(t-t_switch_1)
    # else:
    #     current_angle = max_angle - angle_slope*(t-t_switch_2)
    current_angle = init_angle + \
        angle_cycle(t, min_angle-init_angle, max_angle-init_angle, angle_slope)

    displacement_cl_time = pos_cc + vec_vc1 - vec_vl0 + edgelength * \
        jnp.array([jnp.cos(current_angle), jnp.sin(current_angle)]) - pos_cl
    all_disps = jnp.concatenate(
        [jnp.array([0, 0, 0]), displacement_cl_time, jnp.array([0])])
    return all_disps


ax = plt.gca()
disps = jnp.array([loaded_block_displacement(t) for t in timepoints])
ax.scatter(loaded_block_centroid[0] + disps[:, 3],
           loaded_block_centroid[1] + disps[:, 4], cmap='magma', c=timepoints)
max_dist = jnp.sqrt(disps[-1][3]**2 + disps[-1][4]**2)*1000
# print(max_dist)
# sys.exit()

# damping term, see SI
avmass = density*block_side_length**2
avstiffness = k_stretch
dampingfactor = jnp.sqrt(avmass*avstiffness)
damped_blocks = jnp.arange(0, squares.n_blocks)
damping = dampingprefactor*0.001 * dampingfactor * \
    jnp.full((len(damped_blocks), 3), jnp.array([1, 1, block_side_length]))

# Setup the solver
solve_dynamics = setup_dynamic_solver(
    geometry=squares,
    energy_fn=potential_energy,
    # loaded_block_DOF_pairs=loaded_block_DOF_pairs,
    # loading_fn=loading_fn,
    constrained_block_DOF_pairs=constrained_block_DOF_pairs,
    constrained_DOFs_fn=loaded_block_displacement,
    damped_blocks=damped_blocks,
)

# Initial condition
state0 = jnp.array([
    # Initial position
    0 * random.uniform(random.PRNGKey(0), (squares.n_blocks, 3)),
    # Initial velocity
    0 * random.uniform(random.PRNGKey(1), (squares.n_blocks, 3))
])

# Control parameters
control_params = ControlParams(
    geometrical_params=GeometricalParams(
        block_centroids=block_centroids(*shifts),
        centroid_node_vectors=centroid_node_vectors(*shifts),
    ),
    mechanical_params=MechanicalParams(
        bond_params=LigamentParams(
            k_stretch=k_stretch,
            k_shear=k_shear,
            k_rot=k_rot,
            reference_vector=reference_bond_vectors(),
        ),
        density=density,
        inertia=inertia,  # If omitted, inertia is computed from the geometry and density
        contact_params=ContactParams(
            k_contact=k_contact,
            min_angle=min_angle,
            cutoff_angle=cutoff_angle,
        ),
        damping=damping
    ),
)

# Solve the dynamics
solve_dynamics_jitted = jit(solve_dynamics)
t0 = time.perf_counter()
solution = solve_dynamics_jitted(
    state0=state0,
    timepoints=timepoints,
    control_params=control_params,
)
print(
    f"Solution time (second call, i.e. using jitted solver): {time.perf_counter() - t0:.2f} s")

# Save solution
solutionData = SolutionData(
    block_centroids=block_centroids(*shifts),
    centroid_node_vectors=centroid_node_vectors(*shifts),
    bond_connectivity=bond_connectivity(),
    timepoints=timepoints,
    fields=solution
)
filename = "_".join([
    "rotated_squares",
    "k_springs", f"{k_shear:.2f}", f"{k_rot:.4f}",
    "n1xn2", f"{squares.n1_blocks}x{squares.n2_blocks}",
    "time", f"{simulation_time:.0f}"
])
save_data(os.path.join(datafolder, filename) + "_anne.pkl", solutionData)

###############################################################################
# do secondary analysis of loads and displacements

print('Calculating forces and displacements...')

elastic_forces = grad(potential_energy)


def force_history(solution_data: SolutionData, control_params: ControlParams):
    """
    Compute force-displacement data for the given solution data and control params.
    """
    displacement_history = solution_data.fields[:, 0]
    # reaction_block_DOF_pairs = jnp.array([    [loaded_block_id, 0],
    #     [loaded_block_id, 1],]) # TODO
    reaction_block_DOF_pairs = jnp.array([[clamped_block_id, 0],
                                          # TODO
                                          [clamped_block_id, 1], [clamped_block_id, 2],])
    force_history = vmap(
        lambda u: elastic_forces(u, control_params)[
            reaction_block_DOF_pairs[:, 0],
            reaction_block_DOF_pairs[:, 1],
            # reaction_block_DOF_pairs[:, 2],
        ]
    )(displacement_history)
    return force_history


def energy_history(solution_data: SolutionData, control_params: ControlParams):
    """
    Compute force-displacement data for the given solution data and control params.
    """
    displacement_history = solution_data.fields[:, 0]
    return vmap(lambda u: potential_energy(u, control_params))(displacement_history)


my_force_history = force_history(solutionData, control_params)
my_energy_history = energy_history(solutionData, control_params)*1000  # Nmm

fig2, axs2 = plt.subplots(1, 3, figsize=[3*3, 3*1])
axs2[0].set_xlabel(r't')
axs2[0].set_ylabel(r'loads [N]/torques [mNm]')
axs2[1].set_xlabel(r't')
axs2[1].set_ylabel(r'elastic energy [mNm]')
axs2[1].plot(timepoints, my_energy_history*1000)
axs2[0].plot(timepoints, my_force_history[:, 0], label='fx')
axs2[0].plot(timepoints, my_force_history[:, 1], label='fy')
axs2[0].plot(timepoints, my_force_history[:, 2]*1000, label='m')
axs2[0].legend()

# check direction of loads and torques

actuated_vertex_separation = []
measured_load_magnitudes = []

clamped_centroid_node_vector = centroid_node_vectors(*shifts)[
    clamped_block_id, clamped_centroid_node_vector_id]
clamped_centroid_actuationpoint = clamped_block_centroid + \
    torque_arm_adjustment_factor*clamped_centroid_node_vector
loaded_centroid_node_vector = centroid_node_vectors(*shifts)[
    loaded_block_id, loaded_centroid_node_vector_id]
loaded_centroid_actuationpoint = loaded_block_centroid + \
    torque_arm_adjustment_factor*loaded_centroid_node_vector

# visualizen_t
plot_geometry(block_centroids(*shifts),
              centroid_node_vectors(*shifts), bond_connectivity())
geo_ax = plt.gca()
plot_geometry(block_centroids(*shifts),
              centroid_node_vectors(*shifts), bond_connectivity())
geo_ax2 = plt.gca()
geo_colors = cm.magma(jnp.linspace(0.1, 0.9, n_timepoints+1))

for frame in range(len(solutionData.fields)):
    # free_DOF_ids, _, all_DOF_ids = DOFsInfo(
    # squares.n_blocks, constrained_block_DOF_pairs)
    # all_displacement = jnp.zeros(squares.n_blocks*3)
    applied_load_vector = my_force_history[frame]
    free_displacement = solutionData.fields[frame, 0, :, :]
    # Current displacement of all blocks (n_blocks, 3)
    displacement = free_displacement.reshape(-1, 3)
    displacement_loaded_block = displacement[loaded_block_id]  # (x, y, theta)
    # Current position of the loaded block centroid
    current_loaded_centroid = displacement_loaded_block[:2] + \
        loaded_block_centroid
    current_loaded_vertex = _block_to_node_displacement(displacement_loaded_block, loaded_centroid_node_vector,)[
        # Current position of the loaded vertex
        :2] + loaded_block_centroid + loaded_centroid_node_vector
    vec_vl_parallel = (current_loaded_vertex -
                       current_loaded_centroid)*torque_arm_adjustment_factor
    current_loaded_point = current_loaded_centroid + vec_vl_parallel

    displacement_clamped_block = displacement[clamped_block_id]
    current_clamped_centroid = displacement_clamped_block[:2] + \
        clamped_block_centroid
    current_clamped_vertex = _block_to_node_displacement(displacement_clamped_block, clamped_centroid_node_vector,)[
        # Current position of the clamped vertex
        :2] + clamped_block_centroid + clamped_centroid_node_vector
    vec_vc_parallel = (current_clamped_vertex -
                       current_clamped_centroid)*torque_arm_adjustment_factor
    current_clamped_point = current_clamped_centroid + vec_vc_parallel

    geo_ax.scatter(current_loaded_centroid[0], current_loaded_centroid[1],
                   c=jnp.atleast_2d(geo_colors[frame]))
    geo_ax.plot([current_loaded_centroid[0], current_loaded_centroid[0] + applied_load_vector[0]/100.],
                [current_loaded_centroid[1], current_loaded_centroid[1] +
                    applied_load_vector[1]/100.],
                c=geo_colors[frame])
    geo_ax2.plot([current_loaded_centroid[0], current_loaded_centroid[0] + applied_load_vector[2]],
                 [current_loaded_centroid[1], current_loaded_centroid[1]],
                 c=geo_colors[frame])

    geo_ax.plot([current_loaded_point[0], current_clamped_point[0]],
                [current_loaded_point[1], current_clamped_point[1]],
                c=geo_colors[frame], linestyle=':', dash_capstyle='round',
                marker='o')

    force_at_current_loaded_point = applied_load_vector[:-1]
    force_unitvector = force_at_current_loaded_point / \
        jnp.linalg.norm(force_at_current_loaded_point)
    arm = jnp.sqrt(jnp.dot(vec_vl_parallel, vec_vl_parallel) -
                   jnp.dot(vec_vl_parallel, force_unitvector)**2)
    moment_at_current_loaded_point = applied_load_vector[-1] - arm*jnp.linalg.norm(
        force_at_current_loaded_point)
    load_measurement_direction = (current_loaded_point - current_clamped_point) / \
        jnp.linalg.norm(current_loaded_point - current_clamped_point)

    geo_ax.plot([current_loaded_point[0], current_loaded_point[0] + force_at_current_loaded_point[0]/100.],
                [current_loaded_point[1], current_loaded_point[1] +
                    force_at_current_loaded_point[1]/100.],
                c=geo_colors[frame])
    geo_ax2.plot([current_loaded_point[0], current_loaded_point[0] + moment_at_current_loaded_point],
                 [current_loaded_point[1], current_loaded_point[1]],
                 c=geo_colors[frame])

    geo_ax.plot([current_loaded_centroid[0], current_loaded_centroid[0] + applied_load_vector[0]/100.],
                [current_loaded_centroid[1], current_loaded_centroid[1] +
                    applied_load_vector[1]/100.],
                c=geo_colors[frame])
    measured_load_magnitude = jnp.dot(
        force_at_current_loaded_point, load_measurement_direction)
    measured_load_magnitudes.append(measured_load_magnitude)

    actuated_vertex_separation.append(jnp.linalg.norm(
        current_clamped_point-current_loaded_point)*1000)  # millimeters
actuated_vertex_separation = jnp.array(actuated_vertex_separation)
measured_load_magnitudes = jnp.array(measured_load_magnitudes)

# vertex_forcing = applied_force_cycle(timepoints, amplitude_1, amplitude_2, loading_rate)

my_dvals = -(actuated_vertex_separation -
             actuated_vertex_separation[0])
interp_dvals = jnp.linspace(my_dvals.min(), my_dvals.max(), len(my_dvals))

interp_energies = jnp.interp(interp_dvals, my_dvals, my_energy_history)
numeric_force = jnp.diff(interp_energies)/jnp.diff(interp_dvals)
numeric_force = jnp.concatenate([jnp.array([0]), numeric_force])
measured_interp_force = jnp.interp(
    interp_dvals, my_dvals, measured_load_magnitudes)


interp_energies = interp1d(my_dvals, my_energy_history)(interp_dvals)
numeric_force = jnp.diff(interp_energies)/jnp.diff(interp_dvals)
numeric_force = jnp.concatenate([jnp.array([0]), numeric_force])
measured_interp_force = interp1d(
    my_dvals, measured_load_magnitudes)(interp_dvals)

# plot results
fig, [ax, ax2, ax3, ax4] = plt.subplots(1, 4, figsize=[4*3, 3])
ax.set_xlabel(r'$t$  [s]')
ax.set_ylabel(r'$d$  [mm]')

ax2.set_xlabel(r'$t$  [s]')
ax2.set_ylabel(r'$F$  [N]')

ax3.set_xlabel(r'$d$  [mm]')
ax3.set_ylabel(r'$F$  [N]')

ax4.set_xlabel(r'$d$  [mm]')
ax4.set_ylabel(r'$E$  [mNm]')

ax.plot(timepoints, my_dvals)
ax2.plot(timepoints, measured_load_magnitudes)
ax3.plot(my_dvals, measured_load_magnitudes)
ax3.plot(interp_dvals, numeric_force)
ax3.plot(interp_dvals, measured_interp_force)
ax4.plot(my_dvals, my_energy_history)
# ax4.plot(actuated_vertex_separation, elastic_energy_history)
fig.savefig(os.path.join(datafolder, r'time_disp_force.png'))
fig.savefig(os.path.join(datafolder, r'time_disp_force.pdf'))


history_dict = {'displacement_history': my_dvals,
                'force_history': measured_load_magnitudes,
                'force_history_derived': numeric_force,
                'displacement_history_derived': interp_dvals,
                }
save_data(os.path.join(
    datafolder,   "simulation_cycle_quantities.pkl"), history_dict)

if make_animation:
    xlim, ylim = squares.get_xy_limits(
        *shifts) + 0.5*squares.spacing * jnp.array([-1, 1])
    print("plotting frames")
    generate_animation(solutionData, field="u",
                       out_filename=f"{datafolder}/animation", deformed=True, figsize=(10, 5), xlim=xlim, ylim=ylim, dpi=100)
