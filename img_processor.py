#! /usr/bin/env python
# -*- coding: utf_8 -*-

import os, sys, time
import scipy.ndimage as ndimage
import numpy as np
#from numpy import *
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects
from fit_exp_2d import Exp2D as fit
from PIL import Image

######################
#  Global Variables  #
######################

calibration_data = "140115_zscale.dat"
#min_flake_size = 1000

############################
#  Image Processing Class  #
############################


class IM(object):
  _filename = ""
  _data = None
  _mask = None
  _scale = 1
  #_zscale = lambda x: x

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
    '''
    masked_image():
    Returns a masked image array
    '''
    return np.ma.masked_array(self.data, self.mask)

  def __init__(self, filename, min_flake_size=1000, max_flake_size=100000,
      threshold=1500,  ChannelNumber=1, backgroundpicture='Img_018b_bg.JPG'):
    """init"""
    self._filename = filename
    self.open(filename,  ChannelNumber,  backgroundpicture)
    self.mask = threshold
    #self.label_im = threshold #(threshold, 0)
    self.zscale = self.z_calibration()
    #self.zscale = lambda x: x
    self.minsize = min_flake_size
    self.maxsize = max_flake_size

  def open(self, flakes_image,  ChannelNumber,  background_image):
    '''
    open(filename):
    Opens an image with given filename.
    '''

    #bg_im = Image.open(backgroundpicture)
    #pim = Image.open(filename)
    #fl_im = Image.open(filename)
    #im = np.fromstring(pim.tostring(), dtype=np.uint16)
    #im.shape = pim.size[::-1]

    #x = ChannelNumber
    #self.data = np.asarray(pim)[:, :, 0] # - np.asarray(pim.convert("L", rgb2xyzG))

    #im = np.asarray(fl_im)[:, :, x]
    #im = array(im ,  dtype = float32)

    #imb = np.asarray(bg_im)[:, :, x]
    #imb = array(imb ,  dtype = float32)

    #imd = clip(255 - (imb - im),  0,  255)
    #imdiv = clip(im / imb * 255,  0,  255)

    channel = ChannelNumber

    #fl_im, bg_im =  (plt.imread(pic)[:, :, channel].astype(np.float32,
                    #copy=False) for pic in (flakes_image, background_image))
                    # channel selection is now done at tiff conversion
    fl_im, bg_im =  (plt.imread(pic).astype(np.float32,
                    copy=False) for pic in (flakes_image, background_image))

    #self.data = norm_im = np.clip(fl_im / bg_im, 0, 1)
    #cal_im = (fl_im - bg_im)
    #self.data = (cal_im - cal_im.min()) / (cal_im.max() - cal_im.min())
    self.data = bg_im / fl_im


    #print 'Bild'
    #print im
    #print 'Minus'
    #print imd
    #print 'Dividiert'
    #print imdiv
    #print '1Channel'
    #print np.asarray(fl_im)[:, :, 0]
    #print '2Channel'
    #print np.asarray(fl_im)[:, :, 1]
    #print '3Channel'
    #print np.asarray(fl_im)[:, :, 2]
    #self.data = array(imdiv,  dtype=uint8 )


    print 'Data: \n'
    print self.data


    #a = Image.open('Tv58.png')
    #print np.asarray(a)
    #self.data = np.asarray(a)
    #print 'break'
    #print 'break'
    #test = self.data[:][:][0]
    #print test
    #pim.save('Test.png')
    #BrainR.save('TestR.png')
    #BrainG.save('TestG.png')
    #BrainB.save('TestB.png')

    fig = plt.figure(1, (12,10))
    fig.clf()
    ax = fig.add_subplot(111)
    #im = ax.imshow(array(imdiv,  dtype=uint8 ))
    im = ax.imshow(self.data)
    fig.colorbar(im)
    #plt.savefig(flakes_image[:-4] + '_DataUINTGreenDivided.png')
    plt.savefig(flakes_image[:-4] + '_normalized.png')




    #bgim = Image.open('Img_005.JPG')
    #bg = bgim.convert("RGB", rgb2xyzR)
    #bg.save('Testbg.png')

  def label(self):
    """
    * labels image parts based on `mask`
      every flake gets assigned a mask with a number
    * then for each flake it masks again, removing other flakes
      and append it to the `flakes` dict
    """
    label_image, num_labels = ndimage.label(np.invert(self.mask))
    flakes = {}
    for label in range(num_labels +1): # TODO: better use label key/value pairs
      data = self.data
      mask = np.ma.masked_not_equal(label_image, label).mask
      flake = self.Flake(data, mask)
      if flake.size > self.minsize and flake.size < self.maxsize:
        #print("FLAKE::::: IN IMGPROC", flake)
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
      #size = ndimage.sum(self.mask, self.label_im, self.labels)
      size = np.sum(np.ma.negative(self.data.mask))
      #print("SIZE OF size: ---------- ", sys.getsizeof(size))
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
      effects=[PathEffects.withStroke(linewidth=0.9, foreground='w')]
      ax.annotate('%04.0f,%04.0f\n%04.0f\n%04.0f+-%0.4f\n'%(x,y,s,h,h_err) + text, (y,x), path_effects=effects)
