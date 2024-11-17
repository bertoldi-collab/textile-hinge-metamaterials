import matplotlib.cm as cm
import os
import subprocess
from scipy.integrate import cumtrapz
import pickle
import matplotlib.pyplot as plt
from difflexmm.utils import SolutionData, save_data, load_data, OptimizationData
from difflexmm.plotting import plot_geometry, generate_frames
from difflexmm.geometry import RotatedSquareGeometry, compute_inertia, QuadGeometry
from difflexmm.energy import strain_energy_bond, build_potential_energy, ligament_energy, build_contact_energy, build_contact_energy_mod
from difflexmm.dynamics import setup_dynamic_solver
from jax import random, grad, vmap
import jax.numpy as jnp
import jax
import sys
sys.path.append(r'../')
sys.path.append(r'../../')
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)  # enable float64 type


def unpickle(filename):
    inputfile = open(filename, 'rb')
    pickled = pickle.load(inputfile)
    inputfile.close()
    return pickled


def nupickle(data, filename):
    outputfile = open(filename, 'wb')
    pickle.dump(data, outputfile, protocol=pickle.HIGHEST_PROTOCOL)
    outputfile.close()


def rc_layout():
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['font.size'] = 9
    plt.rcParams['legend.edgecolor'] = '1'
    plt.rcParams['pdf.fonttype'] = 42


def get_B_shifts():
    optimization_data = load_data(
        r"./simulated_data/shape_shifting/opt_shape_shifting_B_pulled_top_bottom.pkl",
    )
    horizontal_shifts, vertical_shifts = optimization_data.design_values[-1]
    # print('ok')
    horizontal_shifts, vertical_shifts = optimization_data.design_values[-1]
    # my_block_cols = [2,3,4]
    # my_block_rows = [8,9,10]
    my_horz_shifts = horizontal_shifts[2:6, 8:12, :]*0.95  # for irregular v1
    my_vert_shifts = vertical_shifts[2:6, 8:12, :]*0.95

    my_horz_shifts = horizontal_shifts[6:9, 0:4, :]*0.95  # trying new shapes
    my_vert_shifts = vertical_shifts[6:9, 0:4, :]*0.95

    return my_horz_shifts, my_vert_shifts


# def simulate_composite_3x3(datafolder, kvalues, biaxfile, syncfile,
#                        slowness=3, dampingprefactor=0.01,
#                        plot_frames=False, init_angle=0,
#                        exp_start_time = 0., disp_exp_offset=0.,
#                        k_contact=0.01, cutoff_angle=-10.*jnp.pi/180, min_angle=-30.*jnp.pi/180,
#                        irregular=False, exp_shift=0.):
def simulate_composite(datafolder, k_shear, k_stretch, k_rot, myconfig,
                       slowness=3, dampingprefactor=0.01,
                       plot_frames=True, init_angle=0,
                       min_disp=0, max_disp=0, Nx=9, Ny=15,
                       density=1200, blocklength=0.01, bondlength=0.001,
                       irregular=False):

    if not os.path.exists(datafolder):
        os.mkdir(datafolder)
    print('Starting simulation.')

    # set material parameters. All units in m, kg, s.
    density = 1200  # Zhermack Elite Double 32 mass density 1.2 g/cm^3 = 1200 kg/m^3
    # k_stretch, k_shear, k_rot = kvalues #stretch, shear, and bending stiffness
    k_shear_rand = k_shear
    k_rot_rand = k_rot
    k_stretch_rand = k_stretch

    # define geometric parameters
    if irregular:
        # blocks have a set side length of 0.01 metre.
        block_side_length = 0.01*0.95
        # bond length is set at 0.001 metre for computation.
        bond_length = 0.1*block_side_length*0.95
    else:
        block_side_length = 0.01*jnp.cos(init_angle)
        bond_length = 0.1*block_side_length*jnp.cos(init_angle)

    # construct geometry
    print('\t Building geometry...')
    geometry = QuadGeometry(n1_blocks=3, n2_blocks=3,
                            spacing=block_side_length*jnp.sqrt(2), bond_length=bond_length*jnp.sqrt(2))
    block_centroids, centroid_node_vectors, bond_connectivity, reference_bond_vectors = geometry.get_parametrization()

    # define geometric parameters
    if irregular:
        shifts = get_B_shifts()
    else:
        shifts = geometry.get_design_from_rotated_square(init_angle)
        shifts = geometry.get_design_from_rotated_rectangle_(
            block_side_length/5, block_side_length/5)
        print(r'ok')

    # construct actuation at top and bottom middle square
    constrained_blocks = jnp.array([1, 7])
    constrained_block_DOF_pairs = jnp.array([[1, 0], [1, 1],
                                             [7, 0], [7, 1]])
    move_direction_vector = jnp.diff(
        block_centroids(*shifts)[constrained_blocks], axis=0)
    move_direction_unitvector = move_direction_vector / \
        jnp.linalg.norm(move_direction_vector)

    # check geometry and actuation directions
    plot_geometry(block_centroids(*shifts),
                  centroid_node_vectors(*shifts), bond_connectivity())
    centers = block_centroids(*shifts)
    c1 = centers[1]
    c2 = centers[7]
    restlength = jnp.linalg.norm(c1-c2)*1000
    print('\t Rest length (mm): ', restlength)
    # sys.exit()
    ax = plt.gca()
    ax.quiver(c1[0], c1[1], -move_direction_vector[0]
              [0], -move_direction_vector[0][1], color='r')
    ax.quiver(c2[0], c2[1], move_direction_vector[1]
              [0], move_direction_vector[1][1])
    # sys.exit()

    # get experimental protocol to copy
    print('\t Building actuation...')

    # get synced data
    # syncdatadict = unpickle(syncfile)
    # # disp_speed = syncdatadict['biax_speed_mmps']*1e-3 #mm per sec to m per sec
    # disp_sim = syncdatadict['disp_sim']/1000. #units in m
    # times_sim = syncdatadict['times_sim']
    # times_exp = syncdatadict['times_exp']
    # disp_exp = syncdatadict['disp_exp']/1000. #units in m

    # #get shifted data
    # disp_sim = syncdatadict['disp_sim_shifted']/1000. #units in m
    # disp_exp = syncdatadict['disp_exp_shifted']/1000. + disp_exp_offset #units in m
    # times_sim = syncdatadict['times_sim_shifted']

    # #get force data
    # force_exp_data = jnp.load(biaxfile)
    # force_exp = jnp.array(force_exp_data['force_history_NS'])
    # energy_exp = cumtrapz(force_exp, disp_exp+exp_shift, initial=0.)
    # # timepoints = disp_time[jnp.where(disp_time >= exp_start_time)[0][0]:]
    # # simulation_time = timepoints[-1]
    # # n_timepoints = len(timepoints)
    # max_disp = 0.
    # min_disp = 0.

    inertia = compute_inertia(
        vertices=centroid_node_vectors(*shifts), density=density)

    # Indicate constraints for extending two (asymmetric) blocks.
    pullidx_bottom = int(Nx/2)
    pullidx_top = int(Nx*Ny)-1 - pullidx_bottom
    # print(pullidx_top, pullidx_bottom)
    constrained_block_DOF_pairs = jnp.array([[pullidx_bottom, 0], [pullidx_bottom, 1],  # blocks 4 and 95, x and y
                                             [pullidx_top, 0], [pullidx_top, 1]])
    constrained_blocks = jnp.array([pullidx_bottom, pullidx_top])

    block_DOF_pairs = jnp.array(
        [[pullidx_bottom, 0], [pullidx_bottom, 1]])  # for force

    move_direction_vector = jnp.diff(
        block_centroids(*shifts)[constrained_blocks], axis=0)
    move_direction_unitvector = move_direction_vector / \
        jnp.linalg.norm(move_direction_vector)

    blockside = geometry.spacing/jnp.sqrt(2)
    disp_speed = 0.05*blockside  # m/s

    print('wait... ')
    disp_exp = jnp.concatenate([jnp.linspace(0, max_disp, 10), jnp.linspace(
        max_disp, min_disp, 10), jnp.linspace(min_disp, 0, 10)])/1000
    disp_time = jnp.cumsum(jnp.abs(jnp.diff(disp_exp, prepend=0)))/disp_speed
    timepoints = disp_time
    simulation_time = timepoints[-1]
    n_timepoints = len(timepoints)

    def dispfun_4x(
        t): return -move_direction_unitvector[0][0]*jnp.interp(t, disp_time, disp_exp/2)

    def dispfun_4y(
        t): return -move_direction_unitvector[0][1]*jnp.interp(t, disp_time, disp_exp/2)

    def dispfun_95x(
        t): return move_direction_unitvector[0][0]*jnp.interp(t, disp_time, disp_exp/2)

    def dispfun_95y(
        t): return move_direction_unitvector[0][1]*jnp.interp(t, disp_time, disp_exp/2)

    def displace_to(t):
        displace_functions = jnp.array([
            dispfun_4x(t),
            dispfun_4y(t),
            dispfun_95x(t),
            dispfun_95y(t)
        ])
        return displace_functions

    # Initial conditions
    state0 = jnp.array([
        # Initial position
        0 * random.uniform(random.PRNGKey(0), (geometry.n_blocks, 3)),
        # Initial velocity
        0 * random.uniform(random.PRNGKey(1), (geometry.n_blocks, 3))
    ])

    # Construct damping term to aid convergence
    avmass = density*blockside**2
    avstiffness = k_rot
    dampingfactor = jnp.sqrt(avmass*avstiffness)
    damped_blocks = jnp.arange(0, geometry.n_blocks)
    damping = dampingprefactor*dampingfactor * \
        jnp.full((len(damped_blocks), 3), jnp.array([0, 0, 1]))

    # Construct spring energy
    springs_energy = strain_energy_bond(bond_connectivity=bond_connectivity(
    ), bond_energy_fn=ligament_energy)  # nonlinear
    strain_energy = build_potential_energy(springs_energy)
    contact_energy = build_contact_energy(bond_connectivity())
    contact_kwargs = dict(k_contact=1, min_angle=0 *
                          jnp.pi/180, cutoff_angle=0.1*jnp.pi/180)
    potential_energy = lambda *args, **bond_kwargs: strain_energy(
        *args, **bond_kwargs) + contact_energy(*args, **contact_kwargs)

    # Setup solver
    print('setting up solver')
    solve_dynamics = setup_dynamic_solver(
        geometry=geometry,
        energy_fn=potential_energy,
        # loading_fn=loading,
        constrained_block_DOF_pairs=constrained_block_DOF_pairs,
        constrained_DOFs_fn=displace_to,
        damped_blocks=damped_blocks,
        damping_values=damping)

    # Solve dynamics
    print('solving')
    solution = solve_dynamics(
        state0=state0,
        timepoints=timepoints,
        centroid_node_vectors=centroid_node_vectors(*shifts),
        inertia=inertia,
        # bond params
        k_stretch=k_stretch_rand,
        k_shear=k_shear_rand,
        k_rot=k_rot_rand,
        reference_vector=reference_bond_vectors())

    print('solution')
    solution_data = SolutionData(
        block_centroids=block_centroids(*shifts),
        centroid_node_vectors=centroid_node_vectors(*shifts),
        bond_connectivity=bond_connectivity(),
        timepoints=timepoints,
        fields=solution)

    print('calculating elastic energy')
    elastic_energy_history = vmap(
        lambda u: potential_energy(u,
                                   centroid_node_vectors(*shifts),
                                   k_stretch=k_stretch_rand,
                                   k_shear=k_shear_rand,
                                   k_rot=k_rot_rand,
                                   reference_vector=reference_bond_vectors(),))(solution_data.fields[:, 0])

    print('calculating forces')

    def reaction_force_history(elastic_forces, displacement_history, block_DOF_pairs, *args, **kwargs):
        return vmap(
            lambda u: jnp.linalg.norm(elastic_forces(
                u, *args, **kwargs)[block_DOF_pairs[:, 0], block_DOF_pairs[:, 1]])
        )(displacement_history)

    elastic_forces = grad(
        lambda u, *args, **kwargs: potential_energy(u, *args, **kwargs))

    force_history = reaction_force_history(
        elastic_forces,
        solution_data.fields[:, 0],
        block_DOF_pairs,
        centroid_node_vectors(*shifts),
        # bond params
        k_stretch=k_stretch_rand,
        k_shear=k_shear_rand,
        k_rot=k_rot_rand,
        reference_vector=reference_bond_vectors(),
    )
    # print(force_history)

    plot_force = jnp.where(
        disp_exp[-len(timepoints):] < 0, - force_history, force_history)
    fig, [ax, ax2, ax3] = plt.subplots(1, 3, figsize=[3*3, 3])
    ax.set_xlabel(r'$t$  [s]')
    ax.set_ylabel(r'$d$  [mm]')

    ax2.set_xlabel(r'$t$  [s]')
    ax2.set_ylabel(r'$F$  [N]')

    ax3.set_xlabel(r'$d$  [mm]')
    ax3.set_ylabel(r'$F$  [N]')

    ax.plot(timepoints, 1000*disp_exp[-len(timepoints):])
    ax2.plot(timepoints, plot_force)
    ax3.plot(1000*disp_exp[-len(timepoints):], plot_force)
    fig.savefig(os.path.join(datafolder, r'time_disp_force.png'))
    fig.savefig(os.path.join(datafolder, r'time_disp_force.pdf'))
    plt.close(fig)
    # sys.exit()

    filename = "_".join([
        "rotated_squares",
        "angle", f"{init_angle:.2f}",
        "k_springs", f"{k_shear:.2f}", f"{k_rot:.4f}",
        "n1xn2", f"{geometry.n1_blocks}x{geometry.n2_blocks}",
        "time", f"{simulation_time:.0f}"])

    save_data(os.path.join(datafolder, filename + ".pkl"), solution_data)

    history_dict = {'displacement_history': disp_exp[-len(timepoints):],
                    'force_history': force_history,
                    'energy_history': elastic_energy_history}
    save_data(os.path.join(
        datafolder,   "simulation_cycle_quantities.pkl"), history_dict)

    if plot_frames:

        xlim, ylim = geometry.get_xy_limits(
            *shifts) + 0.5*geometry.spacing * jnp.array([-1, 1])
        print("plotting frames")
        generate_frames(solution_data, field="v",
                        out_dir=datafolder, deformed=True, figsize=(10, 5), xlim=xlim*1.5, ylim=ylim*1.5, dpi=100)

        cwd = os.getcwd()
        os.chdir(datafolder)
        print("generating movie")
        subprocess.call(['ffmpeg', '-y', '-r', str(int(n_timepoints /
                        simulation_time)), '-i', '%04d.png', 'output.mov'])
        os.chdir(cwd)


# get deformed geometry from .dxf
patch_dict = jnp.load('ordered_blockpatch_dict.npy',
                      allow_pickle='TRUE').item()
patch_centroids = jnp.array([patch_dict[i]['centroid']
                            for i in patch_dict.keys()])
patch_vertices = jnp.array([patch_dict[i]['vertices']
                           for i in patch_dict.keys()])

# get reference diamond geometry from .dxf
ref_patch_dict = jnp.load(
    'ordered_reference_blockpatch_dict.npy', allow_pickle='TRUE').item()
ref_patch_centroids = jnp.array(
    [patch_dict[i]['centroid'] for i in patch_dict.keys()])
ref_patch_vertices = jnp.array(
    [patch_dict[i]['vertices'] for i in patch_dict.keys()])
block_side_length = 10.  # mm, hardcoded!
bond_length = 0.0*block_side_length
Nx = 8
Ny = 8

figc, axc = plt.subplots(1, 1)

# calculate shift vectors for each patch (ordered CCW starting from rightmost diamond point)
mycolors = cm.jet(jnp.linspace(0.1, 0.9, len(patch_dict.keys())))
for patch_idx, dd in patch_dict.items():
    mycolor = mycolors[patch_idx]
    # if patch_idx != 1: continue
    ref_dd = ref_patch_dict[patch_idx]
    patch_verts = dd['vertices']
    ref_patch_verts = ref_dd['vertices']
    shifts = patch_verts - ref_patch_verts
    ref_patch_dict[patch_idx].update({'shifts': shifts})
    # axc.scatter(ref_patch_verts.T[0], ref_patch_verts.T[1], c=jnp.atleast_2d(mycolor), alpha=0.1)
    axc.plot(ref_patch_verts.T[0], ref_patch_verts.T[1], c=mycolor, alpha=0.4)
    axc.quiver(ref_patch_verts.T[0], ref_patch_verts.T[1], shifts.T[0], shifts.T[1], color=mycolor,
               angles='xy', scale_units='xy', scale=1, alpha=0.1)
    axc.plot(patch_verts.T[0], patch_verts.T[1], c=mycolor)
# sys.exit()
    # axc.plot()


# order shift vectors according to convention
horz_shifts = jnp.zeros((2*(Nx+1), Ny, 2))
vert_shifts = jnp.zeros((Nx, Ny+1, 2))
geometry = QuadGeometry(n1_blocks=Nx, n2_blocks=Ny,
                        spacing=block_side_length*jnp.sqrt(2), bond_length=bond_length*jnp.sqrt(2))
block_centroids, centroid_node_vectors, bond_connectivity, reference_bond_vectors = geometry.get_parametrization()
plot_geometry(block_centroids(*[horz_shifts, vert_shifts]),
              centroid_node_vectors(*[horz_shifts, vert_shifts]), bond_connectivity())
ax = plt.gca()
ax.set_xlim(-100, 200)
# ax.set_xlim(-100,100)
for nx in range(Nx):
    for ny in range(Ny):
        patch_idx = ny*Nx + nx
        patch_idx = nx*Ny + ny
        patch_shifts = ref_patch_dict[patch_idx]['shifts']
        ref_patch_verts = ref_patch_dict[patch_idx]['vertices']
        # horz_idx_1 =
        # horz_idx_2 =
        # vert_idx_1 = nx,ny
        # vert_idx_2 = nx, ny+1
        horz_shifts = horz_shifts.at[nx, ny, :].set(patch_shifts[2])
        horz_shifts = horz_shifts.at[nx+1, ny].set(patch_shifts[0])
        vert_shifts = vert_shifts.at[nx, ny].set(patch_shifts[3])
        vert_shifts = vert_shifts.at[nx, ny+1].set(patch_shifts[1])

        # axc.quiver(ref_patch_verts[2].T[0], ref_patch_verts[2].T[1], patch_shifts[2].T[0], patch_shifts[2].T[1], color='g',
        #            angles='xy', scale_units='xy', scale=1)
        # break
        # sys.exit()
    # if ny>0: break
shifts = [horz_shifts, vert_shifts]
save_data(r'Maha_morpher_shifts_quad10mm.pkl', shifts)

# initialize diamond reference geometry

# shift geometry and check if it works
plot_geometry(block_centroids(*shifts),
              centroid_node_vectors(*shifts), bond_connectivity())
ax = plt.gca()
ax.set_xlim(-100, 100)
ax.axis('equal')


sys.exit()

# define geometric parameters
# if irregular:
#     shifts = get_B_shifts()
# else:
#     shifts = geometry.get_design_from_rotated_square(init_angle)
#     shifts = geometry.get_design_from_rotated_rectangle_(block_side_length/5, block_side_length/5)
#     print(r'ok')

# construct actuation at top and bottom middle square
# constrained_blocks = jnp.array([1,7])
# constrained_block_DOF_pairs = jnp.array([[1,0], [1,1],
#                                          [7,0], [7,1]])
# move_direction_vector = jnp.diff(block_centroids(*shifts)[constrained_blocks], axis=0)
# move_direction_unitvector = move_direction_vector/jnp.linalg.norm(move_direction_vector)


sys.exit()


kvaluesets = [
    [2400, 400, 0.05e-3],
    [2400, 400, 0.05e-3],
    [2400*0.7, 400, 0.05e-3],
]
irregulars = [True, False, False]
biaxfiles = [
    r'./3x3_tests/3x3_organza_irregular/exp_3x3_irregular_organza_3.npz',
    r'./3x3_tests/3x3_organza_open/exp_3x3_regular_organza_4.npz',
    r'./3x3_tests/3x3_organza_closed/exp_3x3_collapsed_organza_5.npz'
]
syncfiles = [
    r'./3x3_tests/3x3_organza_irregular/sync_data.pkl',
    r'./3x3_tests/3x3_organza_open/sync_data.pkl',
    r'./3x3_tests/3x3_organza_closed/sync_data.pkl'
]

datafolderstrings = [
    r'./special_sim_data/sim_3x3_irregular_v2_small',
    r'./special_sim_data/sim_3x3_open_rectangular_small',
    r'./special_sim_data/sim_3x3_closed_small',
]
init_angles = [0.,
               1e-3,
               -2*jnp.pi/12.,
               ]
k_contacts = [0.01,
              0.1,
              0.1]
cutoff_angles = [-10*jnp.pi/180.,
                 -5*jnp.pi/180.,
                 -5*jnp.pi/180.]

min_angles = [-30*jnp.pi/180.,
              -30*jnp.pi/180.,
              -30*jnp.pi/180.
              ]

exp_shifts = jnp.array([-1.5, -0.3, -1.5])/1000.  # in m
disp_exp_offsets = jnp.array([0, 0, -0])/1000.  # in m

for ki, kvalues in enumerate(kvaluesets):
    if ki != 1:
        continue
    datafolder = datafolderstrings[ki] + \
        "_k=[{:.2e},{:.2e},{:.2e}]".format(*kvalues)
    k_stretch, k_shear, k_rot = kvalues
    myconfig = [0]*12
    # init_angle = init_angles[ki]
    init_angle = 0.001
    Nx = 3
    Ny = 3
    density = 1200  # Zhermack Elite Double 32 mass density 1.2 g/cm^3 = 1200 kg/m^3
    block_side_length = 0.01  # m
    bond_length = 0.1*block_side_length  # m
    # simulate_composite_3x3(datafolder=datafolder, kvalues=kvalues, biaxfile=biaxfiles[ki],
    #                        syncfile=syncfiles[ki],
    #                    slowness=3, dampingprefactor=1, plot_frames=True,
    #                    init_angle=init_angles[ki], irregular=irregulars[ki],
    #                    k_contact=k_contacts[ki], cutoff_angle=cutoff_angles[ki], min_angle=min_angles[ki],
    #                    exp_shift = exp_shifts[ki],
    #                    disp_exp_offset=disp_exp_offsets[ki])
    # datafolder = r'./simulated_data/sim_{:d}x{:d}_smallblocks_dmin-{:.1f}_dmax-{:.1f}_{:s}_{:d}'.format(Nx, Ny, dmin, dmax, patternstring, mi)
    dmin = 5*3-28.3
    dmax = 5
    simulate_composite(datafolder=datafolder, k_shear=k_shear, k_stretch=k_stretch, k_rot=k_rot, myconfig=myconfig,
                       slowness=3, dampingprefactor=0.05, plot_frames=True,
                       init_angle=init_angle, min_disp=dmin, max_disp=dmax, Nx=Nx, Ny=Ny,
                       density=density, blocklength=block_side_length, bondlength=bond_length)
