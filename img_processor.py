#! /usr/bin/env python
# -*- coding: utf_8 -*-

import os, sys, time
import numpy as np
from PIL        import Image
from scipy      import ndimage
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib import patheffects
from fit_exp_2d import Exp2D as fit

######################
#  Global Variables  #
######################

rp = os.path.dirname(os.path.realpath(sys.argv[0]))
calibration_data = os.path.join(rp, "zscale_2014-01-15.dat")

############################
#  Image Processing Class  #
############################

class IM(object):
  _filename = ""
  _data = None
  _mask = None
  _scale = 1

  def __init__(self, filename, min_flake_size=1000, max_flake_size=100000,
      threshold=100, channel=-1, backgroundpicture='background.tif',
      darkcount=None):
    """init"""
    self._filename = filename
    self.open(filename, backgroundpicture, darkcount, channel)
    self.mask = threshold
    #self.label_im = threshold #(threshold, 0)
    self.zscale = self.z_calibration()
    #self.zscale = lambda x: x
    self.minsize = min_flake_size
    self.maxsize = max_flake_size

  @property
  def scale(self):
    """scale - scaling factor for size calibration"""
    return self._scale
  @scale.setter
  def scale(self, value=1):
    self._scale = value
  @property
  def data(self):
    """image data values as a numpy array."""
    return self._data
  @data.setter
  def data(self, data):
    """image data values as a numpy array."""
    self._data = data
  @property
  def mask(self):
    """image mask as a numpy array."""
    return self._mask
  @mask.setter
  def mask(self, threshold):
    self._mask = self.data > threshold
  @property
  def masked_image(self):
    return np.ma.masked_array(self.data, self.mask)

  def open(self, flakes_image, background_image,
                darkcount_image, channel=None):
    '''
    Open flake and background image, normalizes it to the specified range and
    type (uint8(0..255), float32, etc.) and writes the resulting array into
    self.data
    '''
    # check for grayscale picture
    if channel == -1:
      slicer = ()
    # else take the specified color channel
    else:
      slicer = (slice(None), slice(None), channel)
    # is there a dark count image?
    #if darkcount_image:
      #pictures = (flakes_image, background_image, darkcount_image)
      #imgs = fl_im, bg_im, dc_im
    #else:
      #pictures = (flakes_image, background_image)
      #imgs = fl_im, bg_im
    pictures = (flakes_image, background_image, darkcount_image)
    fl_im, bg_im, dc_im = (plt.imread(pic)[slicer].astype(np.float32,
                    copy=False) for pic in pictures)

    """
    normalize to 0..1
    x_S = data-min / max-min
    height is proportional to negative transmission
    x_N = 1 - x_S
    """
    norm_im = 1 - ((fl_im - dc_im) / (bg_im - dc_im))

    self.data = norm_im
    print 'Data: \n'
    print self.data

    fig = plt.figure(1, (12,10))
    fig.clf()
    ax = fig.add_subplot(111)
    im = ax.imshow(self.data)
    fig.colorbar(im)
    plt.savefig(flakes_image[:-4] + '_normalized.png')

  def label(self):
    """
    * labels image parts based on `mask`
      every flake gets assigned a mask with a number
    * then for each flake it masks again, removing other flakes
      and append it to the `flakes` dict
    """
    # previously here was `np.invert(self.mask)` this now leads to wrong
    # detection
    label_image, num_labels = ndimage.label((self.mask))
    flakes = {}
    for label in range(num_labels +1): # TODO: better use label key/value pairs
      data = self.data
      mask = np.ma.masked_not_equal(label_image, label).mask
      flake = self.Flake(data, mask)
      if flake.size > self.minsize and flake.size < self.maxsize:
        flakes[label] = flake
        flakes[label].zscale = self.zscale
    return flakes

  def z_calibration(self, SaveAs=False):
    """calibrate
    load z calibration data from file and return lookup table
    """
    z = np.genfromtxt(calibration_data, names=True, skiprows=1)

    #fit = exp2d.Exp2D()
    p, best, err = fit().fit(np.array(z['mm']), np.array(z['value']), y0=0)
    fdata = fit().fitdata(np.arange(15,200), best, interpolate=True)
    #print fit.fitfunc(best, 50.3)

    if SaveAs:
      fig = plt.figure(1, (12,10))
      fig.clf()
      ax = fig.add_subplot(111)
      ax.scatter(z['mm'], z['value'], label='samples')
      ax.plot(fdata[0], fdata[1], 'y-', label='fit')
      ax.set_xlabel('height [nm]')
      ax.set_ylabel('intensity [a.u.]')
      plt.savefig(SaveAs)
    return lambda x: fit.fitfunc(best, x)

  class Flake(object):
    _data = None
    _scale = 1

    @property
    def scale(self):
      """scale - scaling factor for size calibration"""
      return self._scale
    @scale.setter
    def scale(self, value=1):
      self._scale = value

    @property
    def size(self):
      size = np.sum(np.ma.negative(self.data.mask))
      return size*self.scale**2

    @property
    def height(self):
      #heigt = ndimage.mean(self.data, self.label_im, self.labels)
      #h_min, h_max, pos_min, pos_max = ndimage.extrema(self.data, self.label_im, self.labels)
      # imshow(ndimage.binary_dilation(a.data.mask, iterations=10))
      #error = h_max - h_min
      area = 10
      x,y = self.center_of_mass
      x = int(x)
      y = int(y)
      d = self.data[x-area:x+area, y-area:y+area]
      #height = d.mean()
      height = self.data.mean()
      return height
      #return self.zscale(height)
      #return height*self.zscale #, error*self.zscale

    @property
    def rawheight(self):
      #heigt = ndimage.mean(self.data, self.label_im, self.labels)
      #h_min, h_max, pos_min, pos_max = ndimage.extrema(self.data, self.label_im, self.labels)
      # imshow(ndimage.binary_dilation(a.data.mask, iterations=10))
      #error = h_max - h_min
      area = 3
      x,y = self.center_of_mass
      x = int(x)
      y = int(y)
      d = self.data[x-area:x+area, y-area:y+area]
      height = d.mean()
      #height = self.data.mean()
      return height

    @property
    def center_of_mass(self):
      return ndimage.center_of_mass(self.data) #self.mask, self.label_im, self.labels)

    @property
    def data(self):
      """image data values as a numpy array."""
      return self._data
    @data.setter
    def data(self, data):
      """image data values as a numpy array."""
      self._data = data

    def __init__(self, data, mask):
      self.data = np.ma.masked_array(data, mask)

    def annotate(self, ax, text=""):
      x,y = self.center_of_mass
      s = self.size
      h = self.height
      #h_err = self.heights[1][i]
      h_err = 0
      effects=[patheffects.withStroke(linewidth=0.9, foreground='w')]
      ax.annotate('%04.0f,%04.0f\n%04.0f\n%04.0f+-%0.4f\n'%(x,y,s,h,h_err) + text, (y,x), path_effects=effects)
