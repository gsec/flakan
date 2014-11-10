#! /usr/bin/env python
# -*- coding: utf_8 -*-

from __future__ import print_function, division
import os, sys, time, pickle, getopt, shutil

module_path = os.path.join('C:', 'Data', 'Python_Modules')
sys.path.append(module_path)

import numpy as np
import img_processor as imgp
from collections  import namedtuple
from matplotlib   import pyplot as plt
from itertools    import chain

################
#  parameters  #
################
"""
The channel is specified by the named tuple, detaching the color from the actual
number of the channel.

TODO: implement ufraw batch command line into script
$ ufraw-batch --out-path=converted --out-type=tif --out-depth=16 *
.CR2 --grayscale=mixer --grayscale-mixer=0,1,0

"""
Color           = namedtuple('Color', ['grey', 'red', 'green', 'blue'])
channel         = Color(-1, 0, 1, 2)

# Global CONSTANTS:
# 500 um equals 300 Pixel in 4xMag for the OlympusMicroscope, OLD
DEFAULT_PATH    = os.curdir
DEFAULT_TAG     = '.tif'
AREA_SCALE      = 5.0 / 3.0
NM_TO_UM        = 0.001
SAMPLE_INFO     = 'TestSampleForCalibration'
FOLDERTAG       = 'single_flake_analysis'

###############
#  functions  #
###############

def IMAnalysis(_fname=None, _path=None, _min_flake_size=None, _darkcount=None,
    _max_flake_size=None, _threshold=None, _channel=None, _background=None):
  '''
  Analysis for the picture `_filename`. Channel sets the color channel that is
  chosen (Red = 0, Green = 1, Blue = 2).  `_background` is the filename for the
  background callibration image
  '''
  fig = plt.figure(1, (12,10))
  fig.clf()
  axs = fig.add_subplot(111)
  im  = imgp.IM(_fname, _min_flake_size, _max_flake_size, _threshold,
                _channel, _background, _darkcount)
  flakes = im.label()
  out_dir = os.path.splitext(_fname)[0] + '_' + FOLDERTAG
  try:
    os.mkdir(out_dir)
  except OSError:
    print(out_dir, " already exists")

  im.z_calibration(os.path.join('.', 'z-calibration.png'))

  print('='*50)
  print("Processing: \t", _fname)
  all_flakes = []

  for flake_key in flakes:
    flake = flakes[flake_key]
    fig.clf()
    ax = fig.add_subplot(111)
    ax.imshow(flake.data)
    flake.annotate(ax, str(flake_key))
    plt.savefig(os.path.join(out_dir,'flake-%03d.png'%flake_key))

    print('_'*50)
    print("Chosen Flake Number  : ", flake_key)
    print("Heigth               : ", flake.height)
    print("Size                 : ", flake.size)
    print("Rawheigth            : ", flake.rawheight)
    all_flakes.append((flake.height,flake.size,flake.rawheight))
  return all_flakes

def wfd(var=None, wfd=None):
  """
  Generates the WholeFlakeData wfd. If not specified it will take the
  wfd_iterator and return the corresponding list. `var` is the argument you want
  to select, if None all arguments are returned concatenated as list elements.
  """
  if wfd is None:
    wfd       = wfd_iter
  # lists (needed multiple times):
  height      = [field[0] for field in wfd]
  area        = [field[1] * AREA_SCALE  for field in wfd]
  sqrt_area   = [_area**0.5 for _area in area]
  # generators (needed only once per function call):
  raw_height  = (field[2] for field in wfd)
  volume      = (_area * (_height*NM_TO_UM) for (_area, _height) in
                zip(area, height))
  ratio       = (_sqarea / (_height*NM_TO_UM) for (_sqarea, _height) in
                zip(sqrt_area, height))
  if var is None:
    return [height, area, sqrt_area, list(volume), list(ratio)]
  elif var == 'height':
    return height
  elif var == 'area':
    return area
  elif var == 'sqrt_area':
    return sqrt_area
  elif var == 'raw_height':
    return list(raw_height)
  elif var == 'volume':
    return list(volume)
  elif var == 'ratio':
    return list(ratio)
  else:
    raise NameError("Specified `Flake variable` is not valid")

def clean(_path):
  """ Removes all analysis files from the folder.  """
  ff = os.listdir(_path)
  files = (os.path.join(_path, f) for f in ff)
  print( "DELETING:")
  for f in files:
    if f.endswith('.png') or (SAMPLE_INFO in f and f.endswith('dat')):
      print(f)
      os.remove(f)
    elif FOLDERTAG in f:
      print(f)
      shutil.rmtree(f)

def usage():
  print()
  print("FlakAn a flake analyzer script")
  print("-"*50)
  print("Usage: ", sys.argv[0], "[option] <folder>")
  print("""
        -h  --help      : Print this help
        -t  --tag       : Specify file ending (default is '.tif' )
        -p  --path      : Specify path with image files. Default is current
                          directory.
        --clean         : Clean up target path. This deletes all '.png' files
                          and folders containing specified 'FOLDERTAG' in their
                          name.
        """)

##########
#  main  #
##########

def main(path=None, tag=None):

  # paths and file tags:
  if path is None:
    path = os.curdir
  os.chdir(os.path.abspath(path))

  if tag is None:
    tag = '.tif'

  # Flake parameters
  flake_params = {'_min_flake_size' : 1.5e3,
                  '_max_flake_size' : 2e5,
                  '_threshold'      : 0.2,
                  '_channel'        : channel.grey,
                  '_path'           : path,
                  '_background'     : 'background.tiff',
                  '_darkcount'      : 'darkcount.tiff',
                  }

  #raw_files = [os.path.abspath(os.path.join(path,f)) for f in os.listdir(path)
  raw_files = [f for f in os.listdir('.') if f.endswith(tag)]
  raw_files.sort()
  print("Files List:")
  print('\n'.join(raw_files))
  """
  per-file-list of per-flake-lists, each containing 3-tuple (height, size,
  rawheight)
  """
  wfd_perfile = []
  for filepath in raw_files:
    #os.chdir(os.path.dirname(filepath))
    print("-"*50)
    print(filepath + ' is being analyzed')
    # only raw_files with endfiletag analyzed -> no png files etc.
    img_analysis = IMAnalysis(filepath, **flake_params)
    wfd_perfile.append(img_analysis)

  """
  Flattened version, iterator of 3 tuples, each flake: (h,s,rh)
  """
  global wfd_iter
  wfd_iter = list(chain.from_iterable(wfd_perfile))

  fig = plt.figure(1, (12,10))
  fig.clf()
  ax = fig.add_subplot(111)
  #ax.scatter(label='samples')
  ax.scatter(wfd('area'), wfd('height'), label='fit')
  #ax.scatter(wfd_Area, wfd_Heigth, label='fit', color='k', alpha=.01)
  ax.set_xlabel('Area [um^2]')
  ax.set_ylabel('Heigth [nm]')
  ax.set_xlim(0, 60000)
  ax.set_ylim(10, 160)
  plt.savefig(os.path.join(SAMPLE_INFO + '_Results.png'))

  fig = plt.figure(1, (12,10))
  fig.clf()
  ax = fig.add_subplot(111)
  #ax.scatter(label='samples')

  ax.scatter(wfd('sqrt_area'), wfd('height'), label='fit')
  ax.set_xlabel('Typical Length [um]')
  ax.set_ylabel('Heigth [nm]')
  ax.set_xlim(0, 120)
  ax.set_ylim(10, 160)
  plt.savefig(os.path.join(SAMPLE_INFO + '_Results_Sqr.png'))

  np.savetxt(SAMPLE_INFO + '_Results.dat', wfd(), fmt='%f', delimiter='\t')

  print( 'Fertig!!!')

# ==============================================================================
if __name__ == '__main__':
  try:
    # Short option syntax: "hv:"
    # Long option syntax: "help" or "verbose="
    opts, args = getopt.getopt(sys.argv[1:], "hp:t:",
        ["help", "path=", "tag=", "clean="])
    arg_tag = arg_path = None

  except getopt.GetoptError, err:
    # Print debug info
    print( str(err))
    #error_action
    sys.exit(2)

  for option, argument in opts:
    if option in ("-h", "--help"):
      usage()
      sys.exit()

    elif option in ("-p", "--path"):
      arg_path = argument

    elif option in ("-t", "--tag"):
      arg_tag = argument

    elif option in ("--clean"):
      clean(argument)
      print("\nThe folder '" + os.path.abspath(argument) +
          "' has been cleaned up.")
      sys.exit()

  main(arg_path, arg_tag)
