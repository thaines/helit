# Copyright 2012 Tom SF Haines

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.



import os.path
import math
import numpy

from utils.make import make_mod
from video_node import *



# The C version - we require that this be built, even if not used...
make_mod('backsub_dp_c', os.path.dirname(__file__), 'backsub_dp_c.c')
import backsub_dp_c



# The OpenCL version - try to compile it, and use it if at all possible...
try:
  make_mod('backsub_dp_cl', os.path.dirname(__file__), ['manager_cl.h', 'open_cl_help.h', 'backsub_dp_cl.c'], openCL=True)
  import backsub_dp_cl
except:
  backsub_dp_cl = None



# The python wrapper around BackSubCore, to give it the right interface etc.
class BackSubDP(VideoNode):
  """A background subtraction algorithm, implimented as a video reader interface that eats another video reader. Uses a per-pixel mixture model, specifically a Dirichlet process on Gaussian distributions - it uses Gibbs sampling with the Gaussians collapsed out. It is an implimentation of the paper 'Background Subtraction with Dirichlet Processes' by Tom SF Haines & Tao Xiang, ECCV 2012."""
  def __init__(self, cl = None):
    """Initialises the background subtraction algoithm to eat video (with the optional channel), which it then runs background subtraction on to output a mask, plus the probability map and the video again if wanted. components can be also set, to set the maximum number of components for each pixels mixture model. cl is the return value from getCL() on the manager object - if not provided OpenCL will not be used."""
    self.video = None
    self.channel = 0
    self.colour = None
    self.colourChannel = 0

    self.core = None

    self.auto_prior = 1.0

    self.param_prior_count = 1.0
    self.param_prior_mu = numpy.array((0.5,0.5,0.5))
    self.param_prior_sigma2 = numpy.array((0.25,0.25,0.25))

    self.param_components = 8
    self.param_concentration_ps = 3.0 # Concentration per second - divided by the frame rate.
    self.param_weight_ps = 30.0 # Weight multiplier per second.
    self.param_cap = 512.0

    self.param_smooth = (0.0**2.0) / (255.0**2.0)
    self.param_minWeight_ps = 1.5 # Min weight for each second.

    self.param_threshold = 0.5
    self.param_cert_limit_pr = 0.0001 # For 320x240 - adjusted to other resolutions.
    self.param_change_limit_pr = 0.0001 # For 320x240 - adjusted to other resolutions.
    self.param_min_same_prob = 0.975
    self.param_change_mult_pr = 1.9 # For 320x240 - adjusted to other resolutions.
    self.param_half_life = 0.8
    self.param_iterations = 16
    
    self.param_minSize = 64
    self.param_maxLayers = 4
    self.param_itersPerLevel = 2

    self.param_con_comp_min = 0

    self.cl = cl


  def setAutoPrior(self, mult = 1.0):
    """Sets the automatic prior, where it updates the prior based on the distribution of the current frame. mult is how much to multiply the variance by, to soften the distribution a bit. Call it with mult set to None to disable it - by default it is on with a value of 1."""
    self.auto_prior = mult

  def setPrior(self, weight = 1.0, mean = numpy.array((0.5,0.5,0.5)), sd = numpy.array((0.5,0.5,0.5))):
    """Sets the parameters for the student-t distribution prior for the pixel values."""
    self.param_prior_count = weight
    self.param_prior_mu = mean
    self.param_prior_sigma2 = sd*sd

  def setDP(self, comp = 8, conc = 0.1, cap = 512.0, weight = 1.0):
    """Sets the parameters for the DP, specifically the number of components, the concentration parameter and the certainty cap, which limits how much weight can be found in the DP. Also a multiplier for the weight of a sample when combined. Because the concentration is frame rate dependent it is actually set assuming 30fps, and converted to whatever the video actually is. Same for the weight parameter."""
    self.param_components = comp
    self.param_concentration_ps = conc * 30.0
    self.param_cap = cap
    self.param_weight_ps = weight *30.0

  def setHackDP(self, smooth = (0.0**2.0)/(255.0**2.0), min_weight = 0.05):
    """Sets some parameters that hack the DP, to help maintain stability. Specifically smooth is an assumption about noise in each sample, used to stop the distributions from ever getting too narrow, whilst min_weight is a minimum influence that a sample can have on the DP, to inhibit overconfidence. This last one is subject to frame rate adjustments - it is set under the assumption of 30 frames per second."""
    self.param_smooth = smooth
    self.param_minWeight_ps = min_weight * 30.0

  def setBP(self, threshold = 0.5, half_life = 0.8, iters = 16):
    """Sets the main parameters for the belief propagation step, the fist of which is the threshold of probability before it considers it to be a foreground pixel. Note that it is converted into a prior, and that due to the regularisation terms this is anything but hard. The half_life is used to indicate the colourmetric distance that equates to the probability of two pixels being different reaching 50:50, whilst iters is how many iterations to run, and is used only for controlling the computational cost. This BP post processing step can be switched off by setting iters to 0, though the threshold will still be used to binarise the probabilities."""
    self.param_threshold = threshold
    self.param_half_life = half_life
    self.param_iterations = iters

  def setExtraBP(self, cert_limit = 0.0001, change_limit = 0.0001, min_same_prob = 0.975, change_mult = 1.9):
    """Sets minor BP parameters that you are unlikelly to want to touch - specifically limits on how certain it can be with regards to it certainty that a pixel is background/foreground and its certainty that two pixels are the same/different, parameter to influence distance scaling to make sure probabilities never drop below a certain value, plus a term to reweight their relative strengths. All except for min_same_prob are set assuming a video resolution of 320x240, and adjusted to whatever the resolution actually is."""
    self.param_cert_limit_pr = cert_limit
    self.param_change_limit_pr = change_limit
    self.param_min_same_prob = min_same_prob
    self.param_change_mult_r = change_mult
  
  def setOnlyCL(self, minSize = 64, maxLayers = 4, itersPerLevel = 2):
    """Sets parameters that only affect the OpenCL version - minSize of a layer of the BP hierachy, for both dimensions; maxLayers is the maximum number of layers of the BP hierachy allowed; itersPerLevel is how many iterations are done at each level of the BP hierachy, except for the last which is done iterations times."""
    self.param_minSize = minSize
    self.param_maxLayers = maxLayers
    self.param_itersPerLevel = itersPerLevel

  def setConComp(self, threshold):
    """Allows you to run connected components after the BP step. You provide the number of pixels below which a foreground segment is terminated. By default it is set to 0, i.e. off."""
    self.param_con_comp_min = threshold


  def width(self):
    return self.video.width()

  def height(self):
    return self.video.height()

  def fps(self):
    return self.video.fps()

  def frameCount(self):
    return self.video.frameCount()


  def inputCount(self):
    return 2

  def inputMode(self, channel=0):
    if channel==0: return MODE_RGB
    else: return MODE_MATRIX

  def inputName(self, channel=0):
    if channel==0: return 'video containing foreground objects moving over stationary background'
    else: return 'Optional colour correction matrix to update the background model - only applies the multiplicative diagonal. Used to correct for lighting changes.'

  def source(self, toChannel, video, videoChannel=0):
    if toChannel==0:
      self.video = video
      self.channel = videoChannel
    else:
      self.colour = video
      self.colourChannel = videoChannel


  def dependencies(self):
    return [self.video]


  def nextFrame(self):
    # Intial setup...
    if self.core==None:
      didCL = False
      if backsub_dp_cl!=None and self.cl!=None:
        try:
          self.core = backsub_dp_cl.BackSubCoreDP()
          self.core.setup(self.cl, self.width(), self.height(), self.param_components, os.path.abspath(os.path.dirname(__file__)))
          didCL = True
        except:
          self.core = None

      if self.core==None:
        if self.cl!=None:
          print 'Warning: Did not use OpenCL implimentation, falling back to slow c implimentation.' ############################### Need better error mech.
        self.core = backsub_dp_c.BackSubCoreDP()
        self.core.setup(self.width(), self.height(), self.param_components)

      self.core.prior_count = self.param_prior_count
      self.core.set_prior_mu(self.param_prior_mu[0], self.param_prior_mu[1], self.param_prior_mu[2])
      self.core.set_prior_sigma2(self.param_prior_sigma2[0], self.param_prior_sigma2[1], self.param_prior_sigma2[2])

      self.core.cap = self.param_cap
      self.core.smooth = self.param_smooth
      
      pixel_ratio = float(320 * 240) / float(self.video.width()*self.video.height())

      self.core.threshold = self.param_threshold
      self.core.cert_limit = math.pow(self.param_cert_limit_pr, pixel_ratio)
      self.core.change_limit = math.pow(self.param_change_limit_pr, pixel_ratio)
      self.core.min_same_prob = self.param_min_same_prob
      self.core.change_mult = self.param_change_mult_pr * math.sqrt(pixel_ratio)
      self.core.half_life = self.param_half_life
      self.core.iterations = self.param_iterations

      self.core.con_comp_min = self.param_con_comp_min
      
      if didCL:
        self.core.minSize = self.param_minSize
        self.maxLayers = self.param_maxLayers
        self.itersPerLevel = self.param_itersPerLevel


      self.lastFrame = 0
      self.prob = numpy.zeros((self.height(),self.width()), dtype=numpy.float32)
      self.mask = numpy.zeros((self.height(),self.width()), dtype=numpy.uint8)
      self.bg = numpy.zeros((self.height(),self.width(),3), dtype=numpy.float32)
    
    # Update temporarlly dependent parameters every frame, incase the frame rate is varying (e.g. webcam)...
    fps = float(self.video.fps())
    self.core.concentration = self.param_concentration_ps / fps
    self.core.weight = self.param_weight_ps / fps
    self.core.minWeight = self.param_minWeight_ps / fps

    # Check if lighting correction infomation is being supplied - if so use it...
    if self.colour:
      ccm = self.colour.fetch(self.colourChannel)
      self.core.light_update(ccm[0,0], ccm[1,1], ccm[2,2])

    # Get the frame...
    frame = self.video.fetch(self.channel)
    if frame==None:
      return False

    # Check if we are automatically screwing with the prior - if so screw...
    if self.auto_prior!=None:
      mean = frame.mean(axis=0).mean(axis=0)
      var = numpy.square(frame).mean(axis=0).mean(axis=0) - mean**2.0
      var *= self.auto_prior

      self.core.prior_update(mean[0], mean[1], mean[2], var[0], var[1], var[2])

    # Do the work...
    self.core.process(frame, self.prob)
    self.core.make_mask(frame, self.prob, self.mask)

    self.bgDirty = True

    return True


  def outputCount(self):
    return 3

  def outputMode(self, channel=0):
    if channel==0: return MODE_MASK
    elif channel==1: return MODE_FLOAT
    else: return MODE_RGB

  def outputName(self, channel=0):
    if channel==0: return 'mask indicating which areas are foreground'
    elif channel==1: return 'foreground probability map'
    elif channel==2: return 'background image - the mode, sort of.'

  def fetch(self, channel = 0):
    if channel==0:
      return self.mask
    elif channel==1:
      return self.prob
    elif channel==2:
      if self.core==None: return None
      if self.bgDirty:
        self.core.background(self.bg)
        self.bgDirty = False
      return self.bg
