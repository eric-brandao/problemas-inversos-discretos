# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 10:28:00 2026

@author: Win11
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import regu_test_problems as tp
import lcurve_functions as lc

#%%
heat = tp.HeatInfRod(L=1.0, alpha = 1.0, tmax = 10, fs = 20)
heat.heat_forward_matrix()
heat.heat_square_pulse(tinit = 1, tend = 3, Tsource = 200, Tbase = 0)
heat.heat_random(mean_temp = 300, std_temp = 20)
#heat.heat_hann_pulse(tinit = 1, tend = 4, Tsource = 200, Tbase = 100)
heat.compute_temp()
heat.add_gaussian_noise(snr = 30, seed = 0)

# Figure
fig, axs = plt.subplots(1, 2, figsize = (9,3))
axs[0].plot(heat.time, heat.u, 'k')
axs[0].grid(linestyle = '--')
axs[0].set_xlim((0, heat.time[-1]))
axs[0].set_ylim((0, 1.5*heat.u.max()))
axs[0].set_xlabel(r'time [s]')
axs[0].set_ylabel(r'$u(t)$ [K]')
axs[0].set_title(r'Source field (what we want to recover)')
axs[1].plot(heat.time, heat.temp_meas, label = r'$L = ${} [m]'.format(heat.L), color = 'b')
axs[1].grid(linestyle = '--')
axs[1].set_xlim((0, heat.time[-1]))
axs[1].set_ylim((0, 1.5*heat.u.max()))
axs[1].legend(loc = 'upper right')
axs[1].grid(linestyle = '--')
axs[1].set_xlabel(r'time [s]')
axs[1].set_ylabel(r'$T(L,t)$ [K]')
axs[1].set_title(r'Measured temperature (no noise)')
plt.tight_layout()

#%%
[U,s,V] = lc.csvd(heat.G)
lc.plot_picard(U, s, heat.temp_meas, figsize = (6,4))
#%%
#lam = lc.l_curve_new(U, s, heat.temp_meas, plotit = True, smin_ratio = 0.001)
lam = lc.gcv_lambda(U, s, heat.temp_meas, plot_gcvfun=True, smin_ratio = 0.001)
h_lc = lc.tikhonov(U, s, V, heat.temp_meas, lam)

# Figure
fig, axs = plt.subplots(1, 2, figsize = (9,3))
axs[0].plot(heat.time, heat.u, 'k')
axs[0].plot(heat.time, h_lc, 'b')
axs[0].grid(linestyle = '--')
axs[0].set_xlim((0, heat.time[-1]))
axs[0].set_ylim((0, 1.5*heat.u.max()))
axs[0].set_xlabel(r'time [s]')
axs[0].set_ylabel(r'$u(t)$ [K]')
axs[0].set_title(r'Source field (what we want to recover)')
axs[1].plot(heat.time, heat.temp_meas, label = r'$L = ${} [m]'.format(heat.L), color = 'b')
axs[1].grid(linestyle = '--')
axs[1].set_xlim((0, heat.time[-1]))
axs[1].set_ylim((0, 1.5*heat.u.max()))
axs[1].legend(loc = 'upper right')
axs[1].grid(linestyle = '--')
axs[1].set_xlabel(r'time [s]')
axs[1].set_ylabel(r'$T(L,t)$ [K]')
axs[1].set_title(r'Measured temperature (no noise)')
plt.tight_layout()

#%%
sky = tp.Sky(size_x=30, size_y=30)
sky.add_stars(pos = [(5, 5), (10,10), (12, 16)], br = [1, 0.8, 0.6])
sky.plot_true_sky()
sky.blur_sky_im(psf_radius = 3)
#sky.add_gaussian_noise(snr = 10)
sky.add_poisson_noise(snr = 10)
sky.plot_blurred_sky()
sky.pixel_sum()
sky.get_blur_matrix()
sky.check_sens_mtx()

#%%
[U,s,V] = lc.csvd(sky.A.toarray())
lc.plot_picard(U, s, sky.sky_blurred.flatten(), figsize = (6,4))


#%%
#lam = lc.l_curve_new(U, s, sky.sky_blurred.flatten(), plotit = True)
lam = lc.gcv_lambda(U, s, sky.sky_blurred.flatten(), plot_gcvfun=True, smin_ratio = 0.000001)
s_lc = lc.tikhonov(U, s, V, sky.sky_blurred.flatten(), lam)
s_lc = np.reshape(s_lc, sky.sky.shape)
#%%
plt.figure(figsize = (4,4))
plt.imshow(s_lc.T, cmap='grey', origin='lower')
plt.colorbar(shrink = 0.5, label = 'brightness')
plt.axis("off") 
plt.tight_layout()

#%%
s_1 = lc.cvx_tikhonov(sky.A.toarray(), sky.sky_blurred.flatten(), 0.02, l_norm = 1)
s_1 = np.reshape(s_1, sky.sky.shape)

plt.figure(figsize = (4,4))
plt.imshow(np.real(s_1.T), cmap='grey', origin='lower')
plt.colorbar(shrink = 0.5, label = 'brightness')
plt.axis("off") 
plt.tight_layout()

#%% Sound field 2D
sf = tp.SoundField2D()
sf.add_mic_array(n_x = 25, n_z =25)
sf.compute_pres(theta_deg = [45, 180], amps = [1, 0.5])
sf.add_noise(snr = 30)
sf.plot_measured_field()
#%%
sf.get_sens_mtx(nwaves = 180)
[U,s,V] = lc.csvd(sf.A)
lc.plot_picard(U, s, sf.pres, figsize = (6,4))
#%%
lam = lc.l_curve_new(U, s, sf.pres, plotit = True)
x_lc = lc.tikhonov(U, s, V, sf.pres, lam)
#%%
x_1 = lc.cvx_tikhonov(sf.A,  sf.pres, 0.02, l_norm = 1)
sf.plot_sol_xy(x_1)
#%%
sf.plot_sol_field(x_1)

#%%
dec = tp.MultiDecay(fs = 500, tmax = 1.2, amps = [1, 0.25, 0.0001],
                    t_decs = [0.5, 1.0])
dec.compose_decay()
dec.add_gaussian_noise(snr = 20)
dec.plot_edc()

#%%
A = dec.sensing_mtx(t_dec_range = [0.05, 3], order = 30)
[U,s,V] = lc.csvd(A)
lc.plot_picard(U, s, dec.edc, figsize = (6,4))
#%%
lam = lc.l_curve_new(U, s, dec.edc, plotit = True)
x_lc = lc.tikhonov(U, s, V, dec.edc, lam)
#%%
x_lc = lc.cvx_tikhonov(A,  dec.edc, 0.3, l_norm = 1)
#%%

plt.figure(figsize = (6,3))
plt.plot(np.linspace(0.05, 3, 30), x_lc[:-1], 'k')
plt.grid(linestyle = '--')
plt.xlabel("Decay Time")
plt.ylabel("Amplitude")
plt.title("Noise level {}".format(x_lc[-1]))
#%%
edc_recon = A @ x_lc
plt.figure(figsize = (6,3))
plt.plot(dec.time, 10*np.log10(dec.edc/np.amax(dec.edc)), '--k')
plt.plot(dec.time, 10*np.log10(edc_recon/np.amax(edc_recon)), 'b')

plt.grid(linestyle = '--')
plt.xlabel("Decay Time")
plt.ylabel("Amplitude") 
