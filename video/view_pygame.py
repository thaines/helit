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



import numpy
import pygame
from pygame.locals import *

from video_node import *



class ViewPyGame(VideoNode):
  """An output node for visualising frames - uses pygame and runs in fullscreen - implimented for demo purposes."""
  def __init__(self, width = 1280, height = 720):
    """You provide the width and height - must be a resolution the monitor can enter."""
    pygame.init()
    self.flags = pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE
    self.screen = pygame.display.set_mode((width, height), self.flags)
    pygame.mouse.set_visible(0)
    
    self.surface = None

    self.video = None
    self.channel = 0

  def width(self):
    return self.video.width()

  def height(self):
    return self.video.height()

  def fps(self):
    return self.video.fps()

  def frameCount(self):
    return self.video.frameCount()


  def inputCount(self):
    return 1

  def inputMode(self, channel=0):
    return MODE_OTHER

  def inputName(self, channel=0):
    return 'Video stream to be visualised - supports MODE_RGB and MODE_FLOAT'

  def source(self, toChannel, video, videoChannel=0):
    self.video = video
    self.channel = videoChannel


  def dependencies(self):
    return [self.video]

  def nextFrame(self):
    # Eat up pygame events, dying if requested...
    for event in pygame.event.get():
      if event.type==QUIT:
        return False
      if event.type==KEYDOWN and event.key==pygame.K_ESCAPE:
        return False
        
      if event.type==KEYDOWN and event.key==pygame.K_SPACE:
        width = self.screen.get_width()
        height = self.screen.get_height()
        
        pygame.display.quit()
        pygame.display.init()
        
        self.flags = self.flags ^ pygame.FULLSCREEN
        
        self.screen = pygame.display.set_mode((width, height), self.flags)
        if self.flags & pygame.FULLSCREEN:
          pygame.mouse.set_visible(0)

    # Fetch the frae we are to draw...
    frame = self.video.fetch(self.channel)
    mode = self.video.outputMode(self.channel)
    if frame==None: return False
    
    # Convert it to something we can blit...
    frame = numpy.swapaxes((frame*255.0).astype(numpy.uint8), 0, 1)
    
    if self.surface==None or frame.shape[0]!=self.surface.get_width() or frame.shape[1]!=self.surface.get_height():
      self.surface = pygame.surfarray.make_surface(frame)
    else:
      pygame.surfarray.blit_array(self.surface, frame)
    
    # Find the drawing position...
    dx = self.screen.get_width()//2 - self.surface.get_width()//2
    dy = self.screen.get_height()//2 - self.surface.get_height()//2
    
    # Draw the frame in the middle of the window...
    self.screen.fill((0,0,0))
    self.screen.blit(self.surface, (dx,dy))
    pygame.display.flip()

    return True


  def outputCount(self):
    return 0
