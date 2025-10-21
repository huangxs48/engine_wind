import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc 
import matplotlib.lines as mlines
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
np.set_printoptions(edgeitems=50)

import pandas as pd

rc('text',usetex=False)
rc('font',family='serif',size=15)

from scipy.optimize import curve_fit


def straightline(x,x0,slope):
    return slope*(x-x0)

m_p = 1.6726e-24
k_B = 1.3807e-16
Lsun = 3.85e33
Msun = 1.9891e33
Rsun = 6.955e10
year = 3.155815e7
pc = 3.0857e18
c_light = 2.99792458e10
G = 6.67259e-8
AU = 1.496e13
a_Rad = 7.57e-15 
ev = 1.16e4 #ev to kelvin


def plot_profile_data(name, ax=None, flag='rho', c='darkslategrey', ls='-',\
                      plot_rmin=None, plot_rmax=None, **kwargs):

  data = pd.read_csv(name, delim_whitespace=True)

  rad = data['r']
  mass = data['Mr']
  rho = data['rho']
  vel = data['vel']
  temp = data['temp']
  gas_gamma = 5.0/3.0

  drad = np.gradient(rad)
  cellvol = 4.0 * np.pi * rad * rad * drad

  #get initial total energy
  ke_tot = np.cumsum(0.5*rho*vel*vel*cellvol)
  ie_tot = np.cumsum(gas_gamma*rho*temp/(gas_gamma-1.0) * cellvol)
  egrav_tot = -G*mass*(rho*cellvol) / rad 

  if ax is not None:
    ax.set_xscale('log')
    #ax.set_xlim(3.0e13, 3.0e14)

  if flag=='ke':
    if ax is not None:
      ax.plot(rad, ke_tot, c=c, ls=ls, **kwargs)
      ax.set_ylabel(r"$KE$")
      ax.set_yscale('log')
    var = ke_tot  
  if flag=='ie':
    if ax is not None:
      ax.plot(rad, ie_tot, c=c, ls=ls, **kwargs)
      ax.set_ylabel(r"$IE$")
      ax.set_yscale('log')
    var = ie_tot  
  if flag=='eg':
    if ax is not None:
      ax.plot(rad, -egrav_tot, c=c, ls=ls, **kwargs)
      ax.set_ylabel(r"$E_{\rm g}$")
      ax.set_yscale('log')
    var = egrav_tot 
  if flag=='etot':
    etot = ke_tot + ie_tot + egrav_tot
    if ax is not None:
      ax.plot(rad, etot, c=c, ls=ls, **kwargs)
      ax.set_ylabel(r"$KE+IE+E_{\rm g}$")
      ax.set_yscale('symlog', base=10, linthresh=1.0e40)
    var = etot


  if flag=='rho':
    if ax is not None:
      ax.plot(rad, rho, c=c, ls=ls, **kwargs)
      ax.set_ylabel(r"$\rho(\rm g~cm^{-3})$")
      ax.set_yscale('log')
      #ax.set_ylim(1.0e-16, 1.0e-6)
      #ax.set_xscale('log')
    var = rho 
  elif flag=='vel':
    if ax is not None:
      ax.plot(rad, vel/1.0e5, c=c, ls=ls, **kwargs)
      ax.set_ylabel(r"$v_{\rm r}(\rm km~s^{-1})$")
      #ax.set_yscale('log')
      #ax.set_xscale('log')
    var = vel
  elif flag=='temp':
    if ax is not None:
      ax.plot(rad, temp, c=c, ls=ls, **kwargs)
      ax.set_ylabel(r"$T(K)$")
      ax.set_yscale('log')
      ax.set_ylim(1.0e3, 1.0e6)
      #ax.set_xscale('log')
    var = temp


  return rad, var

fig = plt.figure(figsize=(10, 10))
spec = gridspec.GridSpec(ncols=1,nrows=3, left=0.15, right=0.95, height_ratios=[1,1,1])

ax1 = fig.add_subplot(spec[0,0])
ax2 = fig.add_subplot(spec[1,0])
ax3 = fig.add_subplot(spec[2,0])

from athena_read import athdf, vtk

#scaling units
from wind_unit import code_units
u = code_units(rho_unit=1.0e-10, temp_unit=1.0e5, l_unit=1.0e12, mmw=0.6)
temp_unit       = u.temp_unit #1.0e5
l_unit          = u.l_unit #1.0e12
rho_unit        = u.rho_unit #1.0e-10
prat = u.prat #5.499119e+02
crat = u.crat #8.082444e+03

mmw = u.mmw#0.6
R_ideal = k_B/mmw/m_p
vel_unit = pow(temp_unit*R_ideal, 0.5) 
time_unit = l_unit/vel_unit
mass_unit = rho_unit*pow(l_unit, 3)
energy_density_unit = rho_unit*pow(vel_unit, 2)
energy_unit = energy_density_unit*pow(l_unit, 3)

#for mass injection
final_energy = 1.0e48
inj_tstart = 0.0
inj_tend = 77579.8626
inj_mass_spread = 1.9891e32 #0.1msun
inj_start_point = 1

inj_mass_spread_code = inj_mass_spread/mass_unit

#read in HDF5 out2
def plot_hdf5_out2(frame, ax, flag='cellvol', c='darkslategrey', ls='-', alpha=1.0, lw=1.0,\
              plot_rmin=None, plot_rmax=None, **kwargs):
  print(frame['Time']*time_unit/3600/24)

  radius_cgs = frame['x1v'][:]*l_unit
  if plot_rmin is None:
    idx_min = 0 
  else:
    idx_min = np.argmin(np.abs(radius_cgs - plot_rmin))
  if plot_rmax is None:
    idx_max = len(frame['x1v'])
  else:
    idx_max = np.argmin(np.abs(radius_cgs - plot_rmax))

  x1v = frame['x1v'][idx_min:idx_max]
  x1f = frame['x1f'][idx_min:idx_max+1]
  dx1f = x1f[1:]-x1f[:-1]
  cellvol = 4.0 * np.pi * x1v * x1v * dx1f

  cellvol = frame['cellvol'][0,0,idx_min:idx_max]
  dmass = frame['dmass_r'][0,0,idx_min:idx_max]
  mass_en= frame['mass_enclose_r'][0,0,idx_min:idx_max]
  rad_index = frame['r_index'][0,0,idx_min:idx_max]
  mass_coord = frame['mass_coord_r'][0,0,idx_min:idx_max]

  radius = frame['x1v'][idx_min:idx_max]*l_unit

  massinj_flag = frame['inject_flag'][0,0,idx_min:idx_max]
  einj = frame['einj'][0,0,idx_min:idx_max]

  if flag=='cellvol':
    ax.plot(radius, cellvol, c=c, ls=ls, alpha=alpha, lw=lw, **kwargs)
    ax.set_ylabel(r"$\Delta V$")
    ax.set_yscale('log')
  elif flag=='dmass':
    ax.plot(radius, dmass, c=c, ls=ls, alpha=alpha, lw=lw, **kwargs)
    #ax.scatter(radius, vel1_prim*vel_unit, c=c, alpha=alpha)
    ax.set_ylabel(r"$m(r)$")
    ax.set_yscale('log')
  elif flag=='rindex':
    ax.plot(radius, rad_index, c=c, ls=ls, alpha=alpha, lw=lw, **kwargs)
    ax.set_ylabel(r"$i_{R}$")
    #ax.set_yscale('log')
  elif flag=='massenclose':
    ax.plot(radius, mass_en*mass_unit, c=c, ls=ls, alpha=alpha, lw=lw, **kwargs)
    #ax.scatter(radius, mass_en*mass_unit, c=c, alpha=alpha)
    ax.plot(radius, np.cumsum(dmass)*mass_unit, c='k', ls=':', alpha=alpha, lw=lw, **kwargs)
    ax.set_ylabel(r"$\int_{R_{0}}^{r}\rho dV$")
    ax.set_yscale('log')
  elif flag=='injectflag':
    ax.plot(radius, massinj_flag, c=c, ls=ls, alpha=alpha, lw=lw, **kwargs)
    ax.set_ylabel(flag)
  elif flag=='masscoord':
    ax.plot(radius, mass_coord*mass_unit, c=c, ls=ls, alpha=alpha, lw=lw, **kwargs)
    ax.set_yscale('log')
    ax.set_ylabel(r"$M(r)$")
  elif flag=='massfactor':
    a_coef = np.log(100)/inj_mass_spread_code
    mass_coord *= massinj_flag
    sum_factor = np.cumsum(np.exp(-a_coef*mass_coord) * dmass)
    b_coef = np.exp(-a_coef*mass_coord) 

    index_flag = np.argmin(np.abs(np.cumsum(dmass)-inj_mass_spread_code))
    total_inj_mass = np.cumsum(dmass*massinj_flag)[index_flag]
    exp_mass = sum_factor[index_flag]
    print("exp mass:", exp_mass, "acoef:", a_coef)
    ax.plot(radius, b_coef*dmass*massinj_flag/exp_mass, c=c, ls=ls, alpha=alpha, lw=lw, **kwargs)
    ax.plot(radius, dmass*massinj_flag/total_inj_mass,  c=c, ls=':', alpha=alpha, lw=lw, **kwargs)
    

    # #uniform injection case
    # index_uniform = np.argmin(np.abs(np.cumsum(dmass)-inj_mass_spread_code))
    # uniform_mass = np.cumsum(dmass*massinj_flag)[index_uniform]
    # print("uniform_mass: ", uniform_mass)
    # ax.plot(radius, np.cumsum(dmass*massinj_flag/uniform_mass))

    # #non-uniform injection
    # exp_mass = sum_factor[index_uniform]
    # print("exp mass:", exp_mass)
    # ax.plot(radius, np.cumsum(b_coef*dmass*massinj_flag/exp_mass))
    # ax.set_yscale('log')

  elif flag=='einj':
    print("time", frame2['Time'], "total injected energy", np.sum(einj))
    ax.plot(radius, einj, c=c, ls=ls, alpha=alpha, lw=lw, **kwargs)
    #ax.set_yscale('log')
    ax.set_ylabel(r"$E_{\rm in}$")

  ax.set_xscale('log')

#read in HDF5 out
def plot_hdf5(frame, ax, flag='rho', c='darkslategrey', ls='-', alpha=1.0, lw=1.0,\
              plot_rmin=None, plot_rmax=None, **kwargs):

  print(frame['Time']*time_unit/3600/24)
  radius_cgs = frame['x1v'][:]*l_unit
  if plot_rmin is None:
    idx_min = 0 
  else:
    idx_min = np.argmin(np.abs(radius_cgs - plot_rmin))
  if plot_rmax is None:
    idx_max = len(frame['x1v'])
  else:
    idx_max = np.argmin(np.abs(radius_cgs - plot_rmax))

  x1v = frame['x1v'][idx_min:idx_max]
  x1f = frame['x1f'][idx_min:idx_max+1]
  dx1f = x1f[1:]-x1f[:-1]
  cellvol = 4.0 * np.pi * x1v * x1v * dx1f

  rho_prim = frame['rho'][0,0,idx_min:idx_max]
  vel1_prim = frame['vel1'][0,0,idx_min:idx_max]
  press_prim = frame['press'][0,0,idx_min:idx_max]
  tgas_prim = press_prim/rho_prim 

  radius = frame['x1v'][idx_min:idx_max]*l_unit
  vol_ghost = cellvol[0]
  vol_ghost_cgs = 2*vol_ghost * pow(l_unit, 3)
  einj_cgs = 1.0e48/vol_ghost_cgs
  einj_code = einj_cgs/energy_density_unit
  print("energy injection cgs:", einj_cgs, "vol:", cellvol[0], "code:", einj_code/(5.0/3.0-1.0))

  if flag=='rho':
    ax.plot(radius, rho_prim*rho_unit, c=c, ls=ls, alpha=alpha, lw=lw, **kwargs)
    #ax.scatter(radius, rho_prim*rho_unit, c=c, alpha=alpha)
    ax.set_ylabel(r"$\rho(\rm g~cm^{-3})$")
    ax.set_yscale('log')
    #ax.set_ylim(1.0e-16, 1.0e-6)
  elif flag=='vel':
    ax.plot(radius, vel1_prim*vel_unit/1.0e5, c=c, ls=ls, alpha=alpha, lw=lw, **kwargs)
    #ax.scatter(radius, vel1_prim*vel_unit, c=c, alpha=alpha)
    ax.set_ylabel(r"$v_{\rm r}(\rm km~s^{-1})$")
    #ax.set_yscale('symlog', linthresh=100)
  elif flag=='ein':
    ein = press_prim/(5.0/3.0-1.0)*cellvol
    #print(press_prim/(5.0/3.0-1.0)*cellvol, 1.0e48/(rho_unit*pow(vel_unit, 2)*pow(l_unit, 3)))
    ax.plot(radius, ein * (rho_unit*pow(vel_unit, 2)*pow(l_unit, 3)), c=c, ls=ls, alpha=alpha, lw=lw, **kwargs)
    #ax.scatter(radius, vel1_prim*vel_unit, c=c, alpha=alpha)
    ax.set_ylabel(r"$PV/(\gamma-1)(\rm erg)$")
    ax.set_yscale('log')
  elif flag=='tgas':
    ax.plot(radius, tgas_prim*temp_unit, c=c, ls=ls, alpha=alpha, lw=lw, **kwargs)
    ax.set_ylabel(r"$T(K)$")
    ax.set_yscale('log')
    #ax.set_ylim(1.0e3, 1.0e6)
  elif flag=='dmass':
    dmass = cellvol * rho_prim 
    ax.plot(radius, dmass, c=c, ls=ls, alpha=alpha, lw=lw, **kwargs)
    ax.set_ylabel(r"$T(K)$")
    ax.set_yscale('log')
    #ax.set_ylim(1.0e3, 1.0e6)

  elif flag=='massenclose':
    dmass = cellvol * rho_prim 
    ax.plot(radius, np.cumsum(dmass)*mass_unit, c=c, ls=ls, alpha=alpha, lw=lw, **kwargs)
    ax.set_ylabel(r"$T(K)$")
    ax.set_yscale('log')
    #ax.set_ylim(1.0e3, 1.0e6)

  ax.set_xscale('log')

plot_xmin_, plot_xmax_ = 5.1e13, 5.3e13
hdf5_name = '../csm.out1.00005.athdf'
frame = athdf(hdf5_name, quantities=['rho', 'vel1', 'press', 'Er', 'Fr1'])
t_now = frame['Time']*time_unit/3600/24
tlabel = "t="+'{:.1f}'.format(t_now)+'day'
plot_hdf5(frame, ax1, flag='rho', c='tab:blue', label=tlabel)#,plot_rmin=plot_xmin_, plot_rmax=plot_xmax_)
plot_hdf5(frame, ax2, flag='tgas', c='tab:blue')#,plot_rmin=plot_xmin_, plot_rmax=plot_xmax_)
plot_hdf5(frame, ax3, flag='vel', c='tab:blue')#,plot_rmin=plot_xmin_, plot_rmax=plot_xmax_)

ax3.set_xscale('log')
#ax3.set_yscale('log')

# # #fname = "../../snec_profile/profiles_15Msun_1d48erg_vrat009_0.000e+00day_trimmedouter.txt"
# # fname = "../../../profiles_for_xiaoshan/profiles_15Msun_1d48erg_vrat009_5.707e+00day.txt"
# fname = "../../../profiles_for_xiaoshan/profiles_15Msun_1d48erg_vrat009_3.621e+01day.txt"
# # fname = "../../../profiles_for_xiaoshan/profiles_15Msun_1d48erg_vrat009_5.017e+01day.txt"
# rad, rho = plot_profile_data(name=fname, ax=ax1, flag='rho')
# rad, temp = plot_profile_data(name=fname, ax=ax2, flag='temp')
# rad, vel = plot_profile_data(name=fname, ax=ax3, flag='vel')

# rad, rho = plot_profile_data(name=fname, ax=ax1, flag='ke')
# rad, temp = plot_profile_data(name=fname, ax=ax2, flag='eg')
# rad, vel = plot_profile_data(name=fname, ax=ax3, flag='etot')

# # for ax in [ax1, ax2, ax3]:
# #   ax.set_xlim(1.5e12, 3.0e12)

ax1.legend()
plt.show()




