from skimage import filters
from skimage import restoration
from skimage.morphology import square
import numpy as np
#import pybind_bm3d as m
from settings import FilterSettings
from scipy import ndimage
from skimage.restoration import denoise_tv_bregman
from skimage.restoration import denoise_wavelet
from skimage.exposure import rescale_intensity
import tensorflow as tf

def iou(y_true, y_pred):
     def f(y_true, y_pred):
         intersection = (y_true * y_pred).sum()
         union = y_true.sum() + y_pred.sum() - intersection
         x = (intersection + 1e-15) / (union + 1e-15)
         x = x.astype(np.float32)
         return x
     return tf.numpy_function(f, [y_true, y_pred], tf.float32)

model_dir = 'models/unet_v4'
model = tf.keras.models.load_model(model_dir, custom_objects={'iou': iou})

class DenoisingFilters:

    def __init__(self, image_input):
        self._image_input = image_input

    def standardize(self):
        filter_settings = FilterSettings()
        up = filter_settings.sup
        low = filter_settings.inf
        satured_image = self._image_input
        satured_image[satured_image >= up] = up
        satured_image[satured_image <= low] = low
        delta = up - low
        image_norm = (satured_image - low) / delta
        return image_norm, delta, low

    def gaussian_filter(self, window_size):
        sigma = window_size/3
        return filters.gaussian(self._image_input, sigma=sigma)

    def mean_filter(self, window_size):
        mask = np.ones((window_size, window_size))/(window_size**2)
        return ndimage.filters.convolve(self._image_input, mask)

    def median_filter(self, window_size):
        selem = square(window_size)
        return filters.median(self._image_input, selem=selem)

    def wiener_filter(self, const_psf):
        psf = np.ones((const_psf, const_psf)) / (const_psf * const_psf)
        image_norm, delta, low = self.standardize()
        filtered_image = restoration.unsupervised_wiener(image_norm, psf)[0]
        filtered_image = (filtered_image * delta) - low
        filtered_image = filtered_image - filtered_image.mean()
        return filtered_image

    def bilateral_filter(self, win_size, sigma_r, sigma_d):
        image_norm, delta, low = self.standardize()
        filtered_image = restoration.denoise_bilateral(image_norm,
                                                       win_size=win_size,
                                                       sigma_color=sigma_r,
                                                       sigma_spatial=sigma_d)
        filtered_image = (filtered_image * delta) - low
        filtered_image = filtered_image - filtered_image.mean()
        return filtered_image

    def nlmeans_filter(self, patch_size, patch_distance):
        image_norm, delta, low = self.standardize()
        filtered_image = restoration.denoise_nl_means(image_norm,
                                                      patch_size=patch_size,
                                                      patch_distance=patch_distance)
        filtered_image = (filtered_image * delta) - low
        filtered_image = filtered_image - filtered_image.mean()
        return filtered_image

    def bm3D_filter(self, sigma):
        filter_settings = FilterSettings()
        img_satured = 99 + np.clip(self._image_input, filter_settings.inf, filter_settings.sup)
        filtered_img = m.bm3d(img_satured, sigma) - 99
        return filtered_img

    def tv_filter(self, w):
        image_norm, delta, low = self.standardize()
        filtered_image = denoise_tv_bregman(image_norm, w)
        filtered_image = (filtered_image * delta) - low
        filtered_image = filtered_image - filtered_image.mean()
        return filtered_image

    def wavelets_filter(self, param):
        image_norm, delta, low = self.standardize()
        filtered_image = denoise_wavelet(image_norm)
        filtered_image = (filtered_image * delta) - low
        filtered_image = filtered_image - filtered_image.mean()
        return filtered_image

    
    def unet_filter(self, param):
        im_no_ped_clipped = np.clip(self._image_input, -26, 16)
        im_no_ped_clipped_rescaled = rescale_intensity(im_no_ped_clipped, in_range=(0, 1.))
        return model(im_no_ped_clipped_rescaled.reshape(1, 2048, 2048, 1)).numpy()[0,:,:,0]




