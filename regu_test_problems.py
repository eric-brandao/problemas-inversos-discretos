# -*- coding: utf-8 -*-
"""
some test problems from regu

@author: Admin
"""
import numpy as np
import scipy
import warnings
import matplotlib.pyplot as plt

def gravity_model(n,a,b,d):
    """ Defines the square matrix A by mid-point sampling
    
    Parameters:
    ------------
    n : int
        order of the model
    a : float
        bottom limit of the Fredholm integral
    b : float
        upper limit of the Fredholm integral
    d : float
        vertical distance from the source line (x') to measurement line (x)

    Returns:
    ------------
    A : numpy ndArray
        n x n real sensing matrix
    """
    dx = 1/n
    dxl = (b-a)/n
    x = dx * (np.arange(n) + 0.5) 
    xl = a + dxl * (np.arange(n) + 0.5) 
    X,XL = np.meshgrid(x,xl)
    A = dx * (d/(d**2 + (X-XL)**2)**(3/2)) 
    return A, x

def density_piecewise(n):
    """ Piecewise source function
    
    Defines a function of order n for which the first 1/3 of the source points are 2, and the rest are 1.
    
    Parameters:
    ------------
    n : int
        order of the model

    Returns:
    ------------
    rhox : numpy 1dArray
        pieciwise source term
    """
    nn = int(n/3)
    rhox = np.ones(n)
    rhox[:nn] = 2
    return rhox

def density_sin(x, fp):
    """ Sinusoidal source function
    
    Defines a function or order len(x) for a sinusoidal source function.
    
    Parameters:
    -----------
    x : 1dArray
        measured coordinates
    fp : float
        spatial frequency of the phenomenon
    Returns:
    ------------
    rhox : numpy 1dArray
        sinusoidal source term
    """
    rhox = np.sin(2*np.pi*fp*x)
    return rhox

def density_rechann(x, hann_x = 0.5):
    """ Rectangular + hanning density function
    
    Parameters:
    ------------
    n : int
        order of the model

    Returns:
    ------------
    rhox : numpy 1dArray
        pieciwise source term
    """
    n = len(x)
    nh = len(x[x>hann_x])
    hw = scipy.signal.windows.hann(2*nh)
    rhox = 2*np.ones(n)
    rhox[n-nh:] = hw[nh:]+1
    return rhox

class Sky(object):
    """ Blurred Sky toy problem
    """
    def __init__(self, size_x = 20, size_y = 20):
        # self.size_x, self.size_y = size_x, size_y
        if size_x < 20 or size_y <20:
            warnings.warn("Size of sky should be at least 100 x 100 pixels")
            size_x, size_y = 20, 20
        # instance of empty sky
        self.sky = np.zeros((size_x, size_y))
        
    def add_star(self, pos = (10, 10), br = 1):
        """ adds a star at position "pos" with brightness br
        """
        self.sky[pos] = br
        
    def add_stars(self, pos = [(5, 5), (10,10)], br = [1, 0.8]):
        """ add several stars at once
        """
        for jp, p in enumerate(pos):
            self.add_star(pos = p, br = br[jp])

    def circular_psf(self, psf_radius = 1):
        """Generates a normalized circular (pillbox) PSF kernel.
        
        """
        self.psf_radius = psf_radius
        size = int(2*self.psf_radius + 1)
        y, x = np.ogrid[-self.psf_radius:self.psf_radius+1, -self.psf_radius:self.psf_radius+1]
        # Create a binary mask where distance from center is <= radius
        mask = (x**2 + y**2 <= self.psf_radius**2)
        # Initialize kernel and normalize so all elements sum up to 1.0
        self.kernel = np.zeros((size, size))
        self.kernel[mask] = 1.0
        self.kernel /= self.kernel.sum()
        
    def blur_sky_im(self, psf_radius = 1):
        """ blur the sky image
        """
        self.circular_psf(psf_radius = psf_radius)
        self.sky_blurred = scipy.signal.convolve2d(self.sky, self.kernel, mode='same')
        
    def add_gaussian_noise(self, snr = 20, seed = 0):
        """ Add gaussian noise to blurred sky
        """
        np.random.seed(seed)
        n = np.random.normal(loc = 0.0, scale = 1, size = len(self.sky_blurred.flatten()))
        n /= np.linalg.norm(n)
        n *= (10**(-snr/10))*np.linalg.norm(self.sky_blurred.flatten())
        self.sky_blurred += np.reshape(n, self.sky.shape)
        
    def add_poisson_noise(self, snr = 20, seed = 0):
        """
        Scales a blurred image to hit a target peak SNR under Poisson noise,
        applies the noise, and returns the image scaled back to original units.
        """
        # Enforce non-negativity
        clean_img = np.clip(self.sky_blurred, 0, None)
        # Find the current maximum pixel value in the blurred image
        current_peak = np.max(clean_img)
        # Calculate the required target peak photon count
        # Since SNR = sqrt(Intensity), Target_Intensity = SNR^2
        target_peak_value = snr ** 2
        # Calculate the scaling (exposure) factor
        scale_factor = target_peak_value / current_peak
        # Scale image to photon counts
        photon_image = clean_img * scale_factor
        # Apply Poisson noise
        noisy_photon_image = np.random.poisson(photon_image)
        # 7. Scale back to original intensity units
        self.sky_blurred = noisy_photon_image / scale_factor
            
    def get_blur_matrix(self,):
        """ Get sensing matrix
        
        Creates an explicit sparse blur matrix H for a given 2D PSF kernel.
        Assumes 'same' boundary conditions with zero padding.
        """
        H_rows, H_cols = self.sky.shape
        N = H_rows * H_cols  # Total number of pixels
        # Get coordinates of the PSF relative to its center
        kh, kw = self.kernel.shape
        kh2, kw2 = kh // 2, kw // 2
        
        # We will build the sparse matrix using the 'LIL' (List of Lists) format for speed
        self.A = scipy.sparse.lil_matrix((N, N))
        
        # Loop through every pixel in the 2D image
        for r in range(H_rows):
            for c in range(H_cols):
                row_idx = r * H_cols + c  # Flattened 1D index of the current pixel
                # Place the PSF weights on the neighboring pixels
                for kr in range(kh):
                    for kc in range(kw):
                        weight = self.kernel[kr, kc]
                        if weight == 0:
                            continue
                        # Target pixel coordinates in the image
                        target_r = r + (kr - kh2)
                        target_c = c + (kc - kw2)
                        # Check boundary conditions (Zero Padding)
                        if 0 <= target_r < H_rows and 0 <= target_c < H_cols:
                            col_idx = target_r * H_cols + target_c
                            self.A[row_idx, col_idx] = weight     
        self.A.tocsr() # Convert to CSR format for fast math operations
        # return H.tocsr() # Convert to CSR format for fast math operations
    
    def pixel_sum(self,):
        """ print pixel sum
        """
        print(r"True sky brightness sum: B = {:.3f}".format(np.sum(self.sky)))
        print(r"Blurred sky brightness sum: B = {:.3f}".format(np.sum(self.sky_blurred)))
        
    def check_sens_mtx(self,):
        """ Check sensing matrix forward problem vs. conv
        """
        sb = self.A @ self.sky.flatten() # forward problem
        sb = np.reshape(sb, self.sky.shape)
        norm = np.linalg.norm(sb-self.sky_blurred)
        print("Norm difference (forward - conv) is {:.16f}".format(norm))

    def plot_true_sky(self):
        """ plots true sky
        """
        self.plot_sky(self.sky)
    
    def plot_blurred_sky(self):
        """ plots true sky
        """
        self.plot_sky(self.sky_blurred)

    def plot_sky(self, sky):
        """ plots true sky
        """
        plt.figure(figsize = (4,4))
        plt.imshow(sky.T, cmap='grey', origin='lower')
        plt.colorbar(shrink = 0.5, label = 'brightness')
        plt.axis("off") 
        plt.tight_layout()
