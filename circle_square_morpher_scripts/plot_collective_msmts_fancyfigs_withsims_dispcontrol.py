import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.signal import argrelextrema
import sys
sys.path.append(r'..')
import pickle
from scipy.signal import savgol_filter

def rc_layout():    
    plt.rcParams['figure.autolayout']=True
    plt.rcParams['font.size']=14
    plt.rcParams['legend.edgecolor']='1'
    plt.rcParams['pdf.fonttype']=42
    
def unpickle(filename):
    inputfile = open(filename,'rb')
    pickled = pickle.load(inputfile)
    inputfile.close()
    return pickled

def nupickle(data,filename):
    outputfile = open(filename,'wb')
    pickle.dump(data,outputfile,protocol=pickle.HIGHEST_PROTOCOL)
    outputfile.close()
    
def lighten(color, strength=0.5):
    white = np.ones_like(color)
    newcolor = strength*white + (1-strength)*color
    if len(newcolor)>3:
        newcolor[-1] = color[-1]
    return newcolor
    
rc_layout()
###############################################################################
#prep exp files
# test_T1_plastic_rotclamp_opentoclose_stiffclamps_test4_-1.30_8.50.npz: counterrotating BC
# test_T1_plastic_rotclamp_opentoclose_stiffclamps_test4_flipsample_-1.30_8.50.npz ibid
fnames = [
        # r'./rigid_plastic_blocks_T1/morpher_maha_biaxdata/test_T1_rubber_centerclamp_opentoclose_test5_freefloating_-1.50_11.00.npz',
        #    r'./rigid_plastic_blocks_T1/morpher_maha_biaxdata/test_T1_plastic_centerclamp_opentoclose_test5_freefloating_-1.50_11.00.npz',
        #    r'./rigid_plastic_blocks_T1/morpher_maha_biaxdata/test_L2_rubber_centerclamp_opentoclose_test5_freefloating_-1.50_11.00.npz',
           r'./rigid_plastic_blocks_T1/morpher_maha_biaxdata/test_T1_rubber_centerclamp_opentoclose_test4_-1.50_11.00.npz',
            # r'./rigid_plastic_blocks_T1/morpher_maha_biaxdata/test_T1_plastic_centerclamp_opentoclose_test4_-1.50_11.00.npz', #flexible clamps
           r'./rigid_plastic_blocks_T1/morpher_maha_biaxdata/test_T1_plastic_centerclamp_opentoclose_stiffclamps_test4_-1.50_10.00.npz',
           # r'./rigid_plastic_blocks_T1/morpher_maha_biaxdata/test_T1_plastic_centerclamp_opentoclose_stiffclamps_test5_flipsample_-1.50_10.00.npz',
           r'./rigid_plastic_blocks_T1/morpher_maha_biaxdata/test_L2_rubber_centerclamp_opentoclose_test5_-1.50_11.00.npz',
          ]
labels = [ 'T1', 'T1-r', 'T1-r', 'T1-r','L2']
colors = [ np.array([1,.3,.8]), np.array([.2,.2,.2]), np.array([.6,.6,.6]), np.array([.9,.9,.9]),np.array([.2,1,.5])]
labels = [ 'T1', 'T1-r','L2']
colors = [ np.array([1,.3,.8]), np.array([.6,.6,.6]), np.array([.2,1,.5])]
num_zeros = 9

###############################################################################
#prep sim files
fnames_sim = [
            # r'./simulations/data_hinge-T1_arm-0.60_shearf-0.63_stretchf-0.63_rotf-1.00/simulation_cycle_quantities.pkl',
            # r'./simulations/data_hinge-T1_arm-0.60_shf-0.63_strf-0.63_rotf-0.63_bl-0.05_disp_norot_v1/simulation_cycle_quantities.pkl',
            # r'./simulations/data_hinge-T1_arm-0.80_shf-0.63_strf-0.63_rotf-0.63_bl-0.05_disp_norot_v1/simulation_cycle_quantities.pkl',
            r'./simulations/data_hinge-T1_arm-0.60_shf-0.63_strf-0.63_rotf-1.00_bl-0.05_disp_cycle_v1/simulation_cycle_quantities.pkl',
            r'./simulations/data_hinge-L2_arm-0.60_shf-0.63_strf-0.63_rotf-0.63_bl-0.05_disp_cycle_v1/simulation_cycle_quantities.pkl',
            # r'./simulations/data_hinge-L2_arm-0.60_shearf-0.63_stretchf-0.63_rotf-0.63_bl-0.05_v10/simulation_cycle_quantities.pkl',
            r'./simulations/data_hinge-T1r_arm-0.60_shf-0.63_strf-0.63_rotf-1.00_bl-0.05_disp_cycle_v3/simulation_cycle_quantities.pkl'    
            ]
# labels_sim = [ 'T1', 'T1-2', 'T1-3', 'T1-4', 'T1-5', 'L2']
# colors_sim = [ np.array([1,.3,.8]), np.array([.9,.4,.7]), np.array([.8,.5,.6]), np.array([.7,.6,.5]), np.array([.6,.7,.4]),np.array([.2,1,.5]), ]
labels_sim = [ 'T1',  'L2', 'T1-r']
colors_sim = [ np.array([.8,.1,.6]),np.array([.0,.8,.3]), np.array([.6,.6,.6])]
num_zeros_sim = 3

###############################################################################
#prep figure
fig, axs = plt.subplots(1,7, figsize=[7*3,3])
axs[5].set_xlabel(r'$t$  [s]')
axs[5].set_ylabel(r'$d$  mm')
axs[0].set_xlabel(r'$t$  [s]')
axs[0].set_ylabel(r'$F$  [N]')
axs[1].set_xlabel(r'$d$  [mm]')
axs[1].set_ylabel(r'$F$  [N]')
axs[6].set_xlabel(r'$d$  [mm]')
axs[6].set_ylabel(r'$F$  [N]')
axs[1].set_xlabel(r'$d$  [mm]')
axs[2].set_xlabel(r'$d$  [mm]')
axs[2].set_ylabel(r'$\mathcal{E}$  [mNm]')
axs[3].set_xlabel(r'$d$  [mm]')
axs[3].set_ylabel(r'$\mathcal{E}_{fric}$  [mNm]')
axs[4].set_xlabel(r'$d$  [mm]')
axs[4].set_ylabel(r'$\mathcal{E}$  [mNm]')
axs[1].set_xticks([0,5,10])
axs[1].set_xlim([-1,11])
axs[1].set_ylim([-8,11])
# axs[1].set_ylim(-2, 7)
# axs[1].set_yticks([-2,0, 2, 4, 6])
# axs[2].set_ylim(-1,20)
# axs[1].set_xlim(-2,33)
# axs[2].set_xlim(-2,33)

###############################################################################
#plot sim data
for fi, fname in enumerate(fnames_sim):
    myres = unpickle(fname)

    print(fname)
    d_meas = myres['displacement_history']   # d -= d[0]
    f_meas = myres['force_history']
    
    d_derived = myres['displacement_history_derived']
    # d -= d[0]
    f_derived = myres['force_history_derived'] 
    
    f_meas = savgol_filter(f_meas, 21, 3)
    f_derived = savgol_filter(f_derived, 21, 3)
    
    from scipy.interpolate import interp1d
    
    f_meas_interp = interp1d(d_meas, f_meas)(d_derived)
    
    axs[1].fill_between(d_derived, f_derived, f_meas_interp,label=labels_sim[fi], color=colors_sim[fi], alpha=0.1, lw=2)
    axs[1].plot(d_meas, f_meas, label=labels_sim[fi], color=colors_sim[fi], alpha=0.1, lw=2)
    axs[1].plot(d_derived, f_derived,label=labels_sim[fi], color=colors_sim[fi], alpha=0.1, lw=2)
    axs[1].plot(d_derived, (f_derived+f_meas_interp)/2,label=labels_sim[fi], color=colors_sim[fi], alpha=1, lw=2)
    axs[6].fill_between(d_derived, f_derived, f_meas_interp,label=labels_sim[fi], color=colors_sim[fi], alpha=0.1, lw=2)
    axs[6].plot(d_meas, f_meas, label=labels_sim[fi], color=colors_sim[fi], alpha=0.1, lw=2)
    axs[6].plot(d_derived, f_derived,label=labels_sim[fi], color=colors_sim[fi], alpha=0.1, lw=2)
    axs[6].plot(d_derived, (f_derived+f_meas_interp)/2,label=labels_sim[fi], color=colors_sim[fi], alpha=1, lw=2)
    # axs[1].plot(d_full,label=labels_sim[fi], color=colors_sim[fi], alpha=1, lw=2)
    continue
# sys.exit()
    # sys.exit()
    # t = myres['time_history']
    # d -= d[0]
    # d = -d
    # f = -f
    # e = cumtrapz(f, d, initial=0)
    
###############################################################################
#plot exp data
for fi, fname in enumerate(fnames):
    myres = np.load(fname)
    print(fname)
    d = -myres['displacement_history_NS']
    f = myres['force_history_NS'] 
    t = myres['time_history']
    d -= d[0]
    d = -d
    f = -f
    e = cumtrapz(f, d, initial=0)

    #find up/down cycles
    # d_min_idx = np.concatenate([[0], argrelextrema(d, np.less)[0]])
    d_min_idx = argrelextrema(d, np.less)[0]
    d_max_idx = argrelextrema(d, np.greater)[0]
    
    #plot results
    axs[5].plot(t, d, label=labels[fi], color=colors[fi])
    axs[0].plot(t, f, label=labels[fi], color=colors[fi])
    axs[0].scatter(t[d_min_idx], f[d_min_idx], marker='v', c=np.atleast_2d(colors[fi]))
    axs[0].scatter(t[d_max_idx], f[d_max_idx], marker='^', c=np.atleast_2d(colors[fi]))
    axs[1].plot(d[d_max_idx[1]:], f[d_max_idx[1]:],label=labels[fi], color=colors[fi], alpha=0.3, lw=2)
    axs[4].plot(d, e, color=colors[fi])
    
    #ignore first cycle
    d_min_idx = d_min_idx[1:]
    d_max_idx = d_max_idx[1:]
    
    #calculate average force up and down for integration
    d_interp = np.linspace(d.min(), d.max(), 200)
    f_avgs_up = []
    f_avgs_down = []
    d_downs = ([d[d_min_idx[i]:d_max_idx[i]] for i in range(len(d_max_idx))])
    d_ups = ([d[d_max_idx[i]:d_min_idx[i+1]][::-1] for i in range(len(d_max_idx))])
    f_downs = ([f[d_min_idx[i]:d_max_idx[i]] for i in range(len(d_max_idx))])
    f_ups = ([f[d_max_idx[i]:d_min_idx[i+1]][::-1] for i in range(len(d_max_idx))])
    for fui, f_up in enumerate(f_ups):
        f_interp = np.interp(d_interp, d_ups[fui], f_up)
        f_avgs_up.append(f_interp)
    for fui, f_down in enumerate(f_downs):
        f_interp = np.interp(d_interp, d_downs[fui], f_down)
        f_avgs_down.append(f_interp)
    f_avg_down = np.mean(np.array(f_avgs_up), axis=0)
    f_avg_up = np.mean(np.array(f_avgs_down), axis=0)
    
    axs[1].plot(d_interp,f_avg_up,label=labels[fi], linestyle=':', dash_capstyle ='round', 
                color=colors[fi], zorder=2000, lw=2)
    axs[1].plot(d_interp,f_avg_down,label=labels[fi], linestyle='--', dash_capstyle ='round', 
                color=colors[fi], zorder=3000, alpha=1., lw=2)
    
    #calculate work done on way up
    e_avg_up = cumtrapz(f_avg_up, d_interp, initial=0)
    e_avg_down = cumtrapz(f_avg_down, d_interp, initial=0)
    d_interp_start_idx = np.argmin(np.abs(d_interp))
    
    e_avg_up -= e_avg_up[d_interp_start_idx]
    e_avg_down -= e_avg_down[-1] - e_avg_up[-1]
    
    axs[2].plot(d_interp,e_avg_up,label=labels[fi], linestyle=':', dash_capstyle ='round', color=colors[fi], lw=3)
    axs[2].plot(d_interp,e_avg_down,label=labels[fi], linestyle='--', dash_capstyle ='round', color=colors[fi], lw=3)


axs[0].legend()
# axs[1].legend()
axs[1].plot([-2,12], [0,0], c='k', linestyle=':', dash_capstyle ='round')

fig.savefig(r'Maha_bistables_force_energy_supported.pdf')
fig.savefig(r'Maha_bistables_force_energy_supported.png', dpi=300)
