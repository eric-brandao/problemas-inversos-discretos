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
sky = tp.Sky(size_x=30, size_y=20)
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

#%%
