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
import time
from collections import defaultdict

import cv

from utils.make import make_mod
from video_node import VideoNode



# Arrange for the Python module that does iniail setup for the OpenCL content to be avaliable. incase we want some real speed...
try:
  make_mod('manager_cl', os.path.dirname(__file__), ['manager_cl.h', 'manager_cl.c'], openCL=True)
  import manager_cl
except:
  manager_cl = None



class Manager:
  """Simple class that manages a bunch of objects of type VideoNode - is given a bunch of these objects and then provides a nextFrame method. This method calls the nextFrame method for each object, but does so in an order that satisfies the dependencies. For conveniance it also provides a run method for use with the ViewVideo objects - it calls the cv.WaitKey function as well as nextFrame, and optionally keeps the framerate correct - makes simple visualisations constructed with ReadVideo objects easy to do. It also manages the OpenCL context and queue, in the event that you are optimising the video processing as such, so that frames can be passed between nodes without leaving the graphics card - the useCL parameter allows OpenCL optimisation to be switched off."""
  def __init__(self, useCL = True):
    self.videos = []
    self.dirty = True
    self.profile = defaultdict(float)
    self.profileRuncount = defaultdict(int)

    self.cl = None
    if useCL and manager_cl!=None:
      try:
        self.cl = manager_cl.ManagerCL()
      except: pass

    self.fps = 0.0


  def haveCL(self):
    """Returns True if OpenCL is avaliable, False otherwise."""
    return self.cl!=None

  def getCL(self):
    """Returns the object that needs to be passed to OpenCL supporting nodes if you want them to dance quickly."""
    return self.cl


  def add(self, videos):
    """Videos can be a list-like object of ReadVideo things or an actual ReadVideo object. Adds them to the manager."""
    if isinstance(videos,VideoNode):
      self.videos.append(videos)
    else:
      for vid in videos:
        self.videos.append(vid)
    self.dirty = True


  def __sort_vids(self):
    """Goes through and sorts the videos into an order that is dependency safe, throws an error if this is not possible."""
    unsorted = dict()
    for vid in self.videos: unsorted[id(vid)] = vid # Culls duplicates.
    self.videos = []

    while len(unsorted)!=0:
      didOne = False

      for key in unsorted.keys():
        safe = True
        for dep in unsorted[key].dependencies():
          if id(dep) in unsorted: safe = False

        if safe:
          self.videos.append(unsorted[key])
          del unsorted[key]
          didOne = True

      if not didOne:
        raise Exception('Video dependencies are impossible to satisfy')

    self.dirty = False


  def nextFrame(self):
    """Calls nextFrame for all contained video, in a dependency satisfying order, returning True only if all calls return true."""
    if self.dirty: self.__sort_vids()
    result = True
    for vid in self.videos:
      start = time.clock()
      res = vid.nextFrame()
      end = time.clock()
      self.profile[vid.__class__.__name__] += end - start
      self.profileRuncount[vid.__class__.__name__] += 1
      #if res!=True: print '%s returned False/None on nextFrame'%str(vid)
      result = result and res
    return result


  def run(self, real_time = True, quiet = False, callback = None, profile = False):
    """Helper method that runs the node system of this Manager object until one of the Nodes says to stop. real_time - If True it trys to run in real time - should typically be True if you are visualising the output, otherwise False to go as fast as possible. quiet - If True it doesn't print any status output to the console. callback - A function that can be called every frame; is given no parameters. profile - If True a file profile.csv will be saved to disk, containing a simple profile of how much time was used by each node in the graph."""
    try:
      timePerFrame = 1.0/self.videos[-1].fps()
    except ZeroDivisionError:
      raise Exception('Frame rate obtained from video file is dodgy (fps=%.2f).'%self.videos[-1].fps())
    lastTime = time.clock()
    i = 0
    delay = 0

    if not quiet and self.cl!=None:
      print 'OpenCL is enabled'

    while True:
      if self.nextFrame()==False: break
      i += 1

      now = time.clock()
      self.fps = self.fps*0.9 + 0.1/max(now-lastTime,1e-6)
      if not quiet: print 'Frame %i, time %.3f, sync delay %.3f, fps %.2f'%(i, now-lastTime, max(delay,3)/1000.0, self.fps)
      if callback!=None: callback()

      if real_time:
        delay = int(1000.0*(timePerFrame-(now-lastTime)))
      else:
        delay = 0

      lastTime = now

      key = cv.WaitKey(max(delay,3))
      if key!=-1:
        break

    # If requested write out a file containing profiling information...
    if profile!=False and len(self.profile)>0:
      fn = '%s-profile.csv'%profile if isinstance(profile,str) else 'profile.csv'
      f = open(fn, 'w')
      f.write('Class name, Total run time, Percentage, Runcount, Time per run\n')
      total = sum(self.profile.values())
      for class_name, runtime in self.profile.iteritems():
        runcount = self.profileRuncount[class_name]
        f.write('%s, %.3f, %.3f, %i, %.6f\n'%(class_name, runtime, 100.0*runtime/total, runcount, runtime/runcount))
      f.close()
