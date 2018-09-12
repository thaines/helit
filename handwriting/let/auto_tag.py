# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import os
import os.path
import bz2
import cPickle as pickle
import time
import string

import numpy
import numpy.linalg as la

try:
  from scipy import weave
except ImportError:
  import weave

from gi.repository import Gtk

from frf import frf

from utils.start_cpp import start_cpp

from line_graph.utils_gui.viewer import Viewer
from line_graph.utils_gui.tile_image import TileImage

from ddp.ddp import DDP



letters = string.ascii_lowercase + string.ascii_uppercase



class AutoTagDialog(Gtk.Dialog):
  def __init__(self, let):
    Gtk.Dialog.__init__(self, title='Automatic Tagging')
    
    self.let = let
    
    self.add_button('Cancel', Gtk.ResponseType.CANCEL)
    self.add_button('Go', Gtk.ResponseType.OK)
    
    box = self.get_content_area()
    
    view_width = 768.0
    
    source = self.let.image.get_original()
    img = numpy.frombuffer(source.get_data(), numpy.uint8)
    img = img.reshape((source.get_height(), -1, 4))
    
    # Find and load the model and add where it is to the dialog box...
    # (On failure put up a big warning that the box is a noop.)
    if self.let.auto_model==None:
      state = 'Failed to find handwriting model - nothing will happen when you hit go!'
      path = os.path.dirname(os.path.abspath(let.fn)).split(os.sep)
    
      for i in xrange(len(path),1,-1):
        db_fn = os.path.join(*(['/'] + path[:i] + ['hwr.rf']))
        if os.path.isfile(db_fn):
          start = time.clock()
          self.let.auto_model = dict()
          self.let.auto_model['model'] = frf.Forest()
          f = bz2.BZ2File(db_fn, 'r')
          
          parts = pickle.load(f)
          for key, value in parts.iteritems():
            self.let.auto_model[key] = value
  
          # The Forest header...
          initial_head = f.read(frf.Forest.initial_size())
          head_size = frf.Forest.size_from_initial(initial_head)
          head = initial_head + f.read(head_size - len(initial_head))
  
          trees = self.let.auto_model['model'].load(head)
  
          # Each tree in return...
          for _ in xrange(trees):
            header = f.read(frf.Tree.head_size())
            size = frf.Tree.size_from_head(header)
    
            tree = frf.Tree(size)
            memoryview(tree)[:len(header)] = header
            memoryview(tree)[len(header):] = f.read(size - len(header))
    
            self.let.auto_model['model'].append(tree)

          f.close()

          end = time.clock()
        
          state = 'Loaded model: %s in %.1f seconds' % (db_fn, end - start)
          self.let.alg_time += end - start
          break
    
      box.pack_start(Gtk.Label(state), False, False, 8)
    
    # Get the ranges of the lines, and extract a box for each, creating a viewer and text entry for each...
    self.parts = []
    for line in self.let.ruled.image_slices(self.let.image.get_size()):
      line  = list(line)
      
      part = dict()
      part['line'] = line
      
      spacing = 0.5 * (line[2] - line[1])
      line[1] -= spacing
      line[2] += spacing
      
      viewer = Viewer()
      box.pack_start(viewer, False, False, 0)
      timage = TileImage(None)
      viewer.add_layer(timage)
      
      if line[1]<0: line[1] = 0
      if line[2]>=source.get_height(): line[2] = source.get_height()-1
      timage.from_array(img[int(line[1]):int(line[2]), :, :])
      viewer.set_size_request(view_width, (line[2] - line[1]) * view_width / source.get_width())
      
      part['viewer'] = viewer
      part['tile_image'] = timage
      
      entry = Gtk.Entry()
      box.pack_start(entry, False, False, 0)
      
      part['entry'] = entry
      
      self.parts.append(part)
    
    box.show_all()
    
  
  def run(self):
    response = Gtk.Dialog.run(self)
    
    if response==Gtk.ResponseType.OK and self.let.auto_model!=None:
      # Clear up all tags/splits currently on the line graph before we start this crazy dance...
      start_time = time.clock()
      
      print 'Terminating existing tags/splits...'
      for tag in self.let.line.get_tags():
        self.let.line.rem(tag[1], tag[2])
        
      for split in self.let.line.get_splits():
        self.let.line.rem(split[0], split[1])
      
      self.let.line.segment() # Just incase it goes pear shaped - stops errors on returning to the original interface.
      
      # Extract features from the line graph...
      print 'Extracting features...'
      start = time.time()
      feats = self.let.line.features(**self.let.auto_model['feat'])
      end = time.time()
      print '...done in %.1f seconds' % (end - start)
      
      # Run the model on the features...
      print 'Classifying features...'
      start = time.time()
      probs = self.let.auto_model['model'].predict(feats)[0]['prob'] # [vertex, class probability]
      end = time.time()
      print '...done in %.1f seconds' % (end - start)
      
      # Extract the location of each feature in line space...
      print 'Getting vertices in line space...'
      hg = self.let.ruled.homography
      ihg = la.inv(hg)
      pos = self.let.line.pos(ihg)
      min_x = pos[:,0].min() - 1.0 # Bias term to give space for the tail 'ligaments'.
      max_x = pos[:,0].max() + 1.0 # "
      
      # Parameters for below...
      steps = 1024
      empty_lig_bias = 4.0
      log_prob_cap = 32.0
      
      # Array for collating dots so we can visualise the below...
      dots = []
      
      # For each line that has information...
      for part in self.parts:
        text = part['entry'].get_text().strip()
        if len(text)==0: continue
          
        line_no = part['line'][0]
        print 'Processing line %i...' % line_no
        
        
        # Run dynamic programming...
        ## Work out what the states are, create a mapping from classes...
        words = text.split()
        punc_words = []
        for word in words:
          while len(word)!=0 and (word[0] not in letters):
            punc_words.append(word[0])
            word = word[1:]
          
          tail = []
          while len(word)!=0 and (word[-1] not in letters):
            tail.append(word[-1])
            word = word[:-1]
          
          if len(word)!=0:
            punc_words.append(word)
          punc_words += tail
        
        coded = '_'.join(map(lambda w: '_'.join(w), punc_words))
        coded = '_' + coded + '_'
        classed = numpy.array(map(lambda c: self.let.auto_model['classes'].index(c) if c in self.let.auto_model['classes'] else 0, coded))
        
        print coded
        
        def word_to_wes(word):
          if len(word)==1:
            if word not in letters: return 'b'
            else: return 's' # s for snake - head and tail.
          
          if len(word)==2: return 'h t'
          return 'h ' + ' '.join('b' * (len(word)-2)) + ' t' # h = head, b = body, t = tail.
        wes = ' ' + ' '.join(map(word_to_wes, punc_words)) + ' '
        
        print 'Found %i sections' % len(classed)
        
        
        ## Create the discrete dynamic programming solver...
        dp = DDP()
        dp.prepare(steps, len(classed))
        
        ## Do the unary terms...
        print 'Calculating DP unary terms...'
        uc = numpy.zeros((steps, classed.shape[0]), dtype=numpy.float32)
        uw = numpy.zeros(steps, dtype=numpy.float32)
        
        code = start_cpp() + """
        // Iterate each feature and if its on the line factor it into the unary cost...
         for (int i=0; i<Nprobs[0]; i++)
         {
          float y = POS2(i, 1);
          if ((y>=(line_no-0.5)) && (y<=(line_no+1.5)))
          {
           // Calculate the weight...
            float weight = 1.0;
            if (y<line_no) weight = 2.0 * (y - (line_no-0.5));
            if (y>(line_no+1)) weight = 2.0 * ((line_no+1.5) - y);
          
           // Identify its bin...
            int bin = Nuc[0] * (POS2(i, 0) - float(min_x)) / (float(max_x) - float(min_x));
            if (bin>=Nuc[0]) bin = Nuc[0] - 1;
            
           // Sum it in, applying the mapping from sequence positions to classes (doing an incrimental mean)...
            UW1(bin) += weight;
            for (int j=0; j<Nuc[1]; j++)
            {
             float p = PROBS2(i, CLASSED1(j));
             float nlp = -log(p);
             if (nlp>float(log_prob_cap)) nlp = float(log_prob_cap);
             UC2(bin, j) += (nlp - UC2(bin, j)) * weight / UW1(bin);
            }
          }
         }
         
        // Check all unary costs - any that are zero cost bias strongly towards being ligaments (Which are always class 0)...
         for (int i=0; i<Nuc[0]; i++)
         {
          if (UW1(i)<1.0)
          {
           float bias = float(empty_lig_bias) * (1.0 - UW1(i));
          
           // Nothing - bias to be a ligament...
            for (int j=0; j<Nuc[1]; j++)
            {
             if (CLASSED1(j)!=0)
             {
              UC2(i, j) += bias;
             }
            }
          }
         }
        """
        
        weave.inline(code, ['line_no', 'min_x', 'max_x', 'empty_lig_bias', 'log_prob_cap', 'uc', 'uw', 'classed', 'probs', 'pos'])
        
        uc[0, 1:] = float('inf')
        dp.unary(0, uc)
        
        ## Do the pairwise terms - weighted to encourage splits in such areas...
        print 'Setting DP pairwise terms...'
        tranc = numpy.clip(uw, 0.5, 1.0)
        tranc = 2.0 * 0.5 * (tranc[:-1] + tranc[1:])
        tranc = numpy.concatenate((numpy.zeros((steps-1), dtype=numpy.float32)[numpy.newaxis,:], tranc[numpy.newaxis,:])).T
        
        dp.pairwise(0, ['ordered'] * (steps-1), tranc)
        
        ## Solve...
        print 'Dynamic programming...'
        start = time.time()
        best, cost = dp.best(classed.shape[0]-1)
        end = time.time()
        print '...done in %.1f seconds' % (end - start)
        print 'MAP cost = %.3f' % cost
      
        # Convert the transitions to x positions...
        splits = [min_x] # x coordinate of each split, plus bounds
        for i in xrange(1, steps):
          if best[i-1]!=best[i]:
            val = (i / float(steps)) * (max_x - min_x) + min_x
            splits.append(val)
        splits.append(max_x)
        
        # Print out information about each characters limits...
        #for i in xrange(len(coded)):
        #  print 'char: %s :: (%.3f - %.3f)' % (coded[i], splits[i], splits[i+1])
        
        # Find the centres of the letters and tag them...
        final_tag = [None] * len(coded)
        for i in xrange(len(coded)):
          if coded[i]!='_':
            # The tag we are applying...
            if wes[i]=='b': tag = coded[i]
            elif wes[i]=='h': tag = '_' + coded[i]
            elif wes[i]=='t': tag = coded[i] + '_'
            elif wes[i]=='s': tag = '_' + coded[i] + '_'
            else: raise RuntimeError('coded and wes vectors do not match - please slap Tom')
          
            final_tag[i] = tag
            
            # Measure the height of a line in pixels, so we can scale costs in the below...
            top = hg.dot(numpy.array([0.5 * (splits[i] + splits[i+1]), line_no, 1.0]))
            bot = hg.dot(numpy.array([0.5 * (splits[i] + splits[i+1]), line_no + 1.0, 1.0]))
            
            top /= top[2]
            bot /= bot[2]
            
            height = numpy.sqrt(numpy.square(top - bot).sum())
            
            # Loop a bunch of positions on the vertical - select the one that is closest to the vertical - saves implimenting a closest to capsule method with a reasonable approximation...
            best_n = None
            for offset in numpy.linspace(0.0, 1.0, 11):
              # Its position - center of bounding box...
              tag_pos = numpy.array([0.5 * (splits[i] + splits[i+1]), line_no + offset, 1.0])
            
              # Translate from line to space to pixel space...
              tag_pos = hg.dot(tag_pos)
              tag_pos /= tag_pos[2]
            
              # Find the nearest point...
              n = self.let.line.nearest(tag_pos[0], tag_pos[1])
              
              # Adjust the cost with a center bias...
              n = list(n)
              n[0] += 0.25 * height * numpy.fabs(offset - 0.75)
              
              # Decide if we are going to use it...
              if best_n==None or best_n[0] > n[0]:
                best_n = n
            
            # Create...
            self.let.line.add_tag(best_n[1], best_n[2], tag)

        # Find the intersects of the split points and add the splits...
        for si, split in enumerate(splits[2:-2]):
          si += 2
          
          # Generate end points of a line...
          start = numpy.array([split, line_no - 0.5, 1.0])
          end = numpy.array([split, line_no + 1.5, 1.0])
          
          # Apply the homography to get back to image space...
          start = hg.dot(start)
          start /= start[2]
          
          end = hg.dot(end)
          end /= end[2]
          
          # Do the intersect...
          cuts = self.let.line.intersect(start[0], start[1], end[0], end[1])
          
          # Filter all cuts that do not seperate the relevant tags - this avoids cutting off tails, and generally only results in one possible cut being left...
          e_before = final_tag[si-1]
          e_after = final_tag[si]
          
          if e_before==None: e_before = final_tag[si-2]
          if e_after==None and (si+1)<len(final_tag): e_after = final_tag[si+1]
          
          def valid_cut(cut):
            before, after = self.let.line.between(cut[0], cut[1])
            
            # Note: Orientation is not necesarily the same order, so before and after could be swapped...
            for b, a in [(before, after), (after, before)]:
              if b==None:
                b_ok = e_before==None
              else:
                b_ok = e_before==b[1]
            
              if a==None:
                a_ok = e_after==None
              else:
                a_ok = e_after==a[1]
              
              if b_ok and a_ok: return True
            
            return False
          
          cuts = filter(valid_cut, cuts)
          
          # Apply all remaining cuts (Generally only 1, but could be two for a double connection, e.g. double t)...
          for cut in cuts:
            self.let.line.add_split(cut[0], cut[1])
        
        # Add this lines dynamic programming unary terms into the dots visualisation, so we can see whats driving the output...
        
        ## Calculate the matrix to go from a column-label coordinate system to the viewers coordinate system, plus an appropriate scale...
        pre_scale = numpy.eye(3, dtype=numpy.float32)
        pre_scale[0,0] = (max_x - min_x) / steps
        pre_scale[0,2] = min_x
        pre_scale[1,1] = pre_scale[0,0]
        pre_scale[1,2] = line_no + 1.5
        to_dot = hg.dot(pre_scale)
        
        size = 0.5 * to_dot[0,0] / to_dot[2,2]
        
        ## Create a dot for each cost value...
        d = numpy.empty(((uc.shape[0]-1)*uc.shape[1], 6), dtype=numpy.float32)
        max_cost = uc[1:,:].max()
        
        d[:,0], d[:,1] = map(lambda a: a.flatten(), numpy.meshgrid(numpy.arange(1, uc.shape[0]), numpy.arange(uc.shape[1]), indexing='ij'))
        d[:,2] = size
        d[:,3] = numpy.clip((uc[1:,:].flatten() / max_cost) * 3.0 - 2.0, 0.0, 1.0)
        d[:,4] = numpy.clip((uc[1:,:].flatten() / max_cost) * 3.0 - 1.0, 0.0, 1.0)
        d[:,5] = numpy.clip((uc[1:,:].flatten() / max_cost) * 3.0 - 0.0, 0.0, 1.0)
        
        for i in xrange(d.shape[0]):
          p = to_dot.dot([d[i,0], d[i,1], 1.0])
          p /= p[2]
          d[i,1] = p[0]
          d[i,0] = p[1]
        
        dots.append(d)

        
      # Recalculate the segments - above almost certainly messed them up...
      print 'Recalculating segments...'
      self.let.line.segment()
        
      # Hack the i dot problem - find all untagged segments and link them to a suitable segment if there is one that makes sense - check for alternate segments with tags in the same vertical...
      print 'Planning tittle assignment...'
      to_create = [] # Two stage for computational sanity.
      planned = dict() # To avoid double links.
      vert_seg = self.let.line.get_segs()
      pos = self.let.line.pos()
      
      for seg in xrange(self.let.line.segments):
        adj = self.let.line.adjacent(seg)
        tags = self.let.line.get_tags(seg)
        if len(adj)==0 and len(tags)==0:
          # We have an island with no tags and no neighbours - see if we can find it a friend...
          ## Find all edges in the same vertical, with a little extra wiggle room added in - we convert to line space for this as it makes things sane...
          bounds = self.let.line.get_bounds(seg) # bounds = (min_x, max_x, min_y, max_y)
          
          pnts = numpy.empty((4, 3), dtype=numpy.float32)
          pnts[0,:] = ihg.dot([bounds[0], bounds[2], 1.0])
          pnts[1,:] = ihg.dot([bounds[1], bounds[2], 1.0])
          pnts[2,:] = ihg.dot([bounds[0], bounds[3], 1.0])
          pnts[3,:] = ihg.dot([bounds[1], bounds[3], 1.0])
          pnts /= pnts[:, 2:3]

          search = [pnts[:,0].min(), pnts[:,0].max(), pnts[:,1].min(), pnts[:,1].max()]
          search[0] -= 0.25
          search[1] += 0.25
          search[2] -= 0.25
          search[3] += 1.0
          
          pnts[0,:] = hg.dot([search[0], search[2], 1.0])
          pnts[1,:] = hg.dot([search[1], search[2], 1.0])
          pnts[2,:] = hg.dot([search[0], search[3], 1.0])
          pnts[3,:] = hg.dot([search[1], search[3], 1.0])
          pnts /= pnts[:, 2:3]
          
          search = [pnts[:,0].min(), pnts[:,0].max(), pnts[:,1].min(), pnts[:,1].max()]
          edges = self.let.line.within(*search)
          
          ## Convert edges into segment numbers, using a set to remove duplicates...
          friends = set()
          for es in edges:
            for ei in xrange(*es.indices(self.let.line.edge_count)):
              edge = self.let.line.get_edge(ei)
              
              friends.add(vert_seg[edge[0]])
              friends.add(vert_seg[edge[1]])
          
          friends.discard(seg)
          if len(friends)==0: continue # Nothing to befriend - give up.
          
          ## Find out which segment is closest - use bounding boxes for this, and include a bias to prefer vertical closeness...
          vertical_scale = 0.25
          
          friend = None
          friend_dist_sqr = None
          
          for elem in friends:
            elem_bounds = self.let.line.get_bounds(elem)
            
            dx = (bounds[0] - elem_bounds[1]) if elem_bounds[1] < bounds[0] else ((elem_bounds[0] - bounds[1]) if bounds[1] < elem_bounds[0] else 0.0)
            dy = (bounds[2] - elem_bounds[3]) if elem_bounds[3] < bounds[2] else ((elem_bounds[2] - bounds[3]) if bounds[3] < elem_bounds[2] else 0.0)
            dy *= vertical_scale
            dist_sqr = dx * dx + dy * dy
            
            if friend==None or friend_dist_sqr > dist_sqr:
              friend = elem
              friend_dist_sqr = dist_sqr
          
          ## Check we have not already ordered this link...
          key = (min(seg, friend), max(seg, friend))
          if key in planned: continue
          planned[key] = True
          
          ## Find the vertex closest to the others bounding box in each segment...
          friend_bounds = self.let.line.get_bounds(friend)
          
          rv = numpy.where(vert_seg==seg)[0]
          dx = numpy.zeros(rv.shape, dtype=numpy.float32)
          dy = numpy.zeros(rv.shape, dtype=numpy.float32)
          
          tfw = pos[rv][:,0] < friend_bounds[0]
          dx[tfw] = friend_bounds[0] - pos[rv][tfw,0]
          tfw = pos[rv][:,0] > friend_bounds[1]
          dx[tfw] = pos[rv][tfw,0] - friend_bounds[1]
          tfw = pos[rv][:,1] < friend_bounds[2]
          dy[tfw] = friend_bounds[2] - pos[rv][tfw,1]
          tfw = pos[rv][:,1] > friend_bounds[3]
          dy[tfw] = pos[rv][tfw,1] - friend_bounds[3]
          
          v_seg = rv[numpy.argmin(numpy.square(dx) + numpy.square(dy))]

          rv = numpy.where(vert_seg==friend)[0]
          if rv.shape[0]==0: # Posible, though indicative of a previous issue.
            print 'Considered auto-link to a 0 vertex segment - weird'
            continue 
          dx = numpy.zeros(rv.shape, dtype=numpy.float32)
          dy = numpy.zeros(rv.shape, dtype=numpy.float32)
          
          tfw = pos[rv][:,0] < bounds[0]
          dx[tfw] = bounds[0] - pos[rv][tfw,0]
          tfw = pos[rv][:,0] > bounds[1]
          dx[tfw] = pos[rv][tfw,0] - bounds[1]
          tfw = pos[rv][:,1] < bounds[2]
          dy[tfw] = bounds[2] - pos[rv][tfw,1]
          tfw = pos[rv][:,1] > bounds[3]
          dy[tfw] = pos[rv][tfw,1] - bounds[3]
          
          v_friend = rv[numpy.argmin(numpy.square(dx) + numpy.square(dy))]
          
          ## Link these two vertices (which have to be turned into edges) together...          
          edges_seg = self.let.line.vertex_to_edges(v_seg)
          edges_friend = self.let.line.vertex_to_edges(v_friend)
          
          if len(edges_seg)!=0 and len(edges_friend)!=0:
            t_seg = 0.0 if edges_seg[0][1]==False else 1.0
            t_friend = 0.0 if edges_friend[0][1]==False else 1.0
            to_create.append((edges_seg[0][0], t_seg, edges_friend[0][0], t_friend))
            
            print 'Going to linked segment %i to segment %i' % (seg, friend)
      
      # Apply the above and rebuild the segments...
      print 'Enacting tittle assignment plan...'
      for cmd in to_create:
        self.let.line.add_link(*cmd)
      self.let.line.segment()
      
      print 'Auto-tagging done'
      end_time = time.clock()
      self.let.alg_time += end_time - start_time
