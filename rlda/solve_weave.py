# -*- coding: utf-8 -*-

# Copyright 2010 Tom SF Haines

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
import scipy.special
import scipy.weave as weave

import solve_shared as shared



def iniGibbs(st):
  # Code...
  code = """
  // t phase - iterate the state matrix...
   for (int w=0;w<Nstate[0];w++)
   {
    int doc = STATE2(w,0);
    int region = ir[STATE2(w,2)];
    int word = STATE2(w,3);
   
    // Calculate the probability distribution based on the current state, unnormalised but with a sum so it can be normalised in use...
     float sum = 0.0;
     for (int t=0;t<NdistT[0];t++)
     {
      float val = (DT2(doc,t) + alpha) * (WRT3(word,region,t) + beta);
      val /= mt[t] + (Nwrt[0]*Nwrt[1]*beta);
      sum += val;
      distT[t] = val;
     }

    // Draw a sample from the distribution, update the state accordingly...
     int topic = 0;
     float total = 0.0;
     for (;topic<NdistT[0]-1;topic++)
     {
      total += distT[topic]/sum;
      if (total>randomT[w]) break;
     }
     STATE2(w,1) = topic;

    // Add the sample into the relevant data structures...
     DT2(doc,topic) += 1;
     WRT3(word,region,topic) += 1;
     mt[topic] += 1;
     mr[region] += 1;
   }


  // r phase - iterate the ir matrix...
   for (int i=0;i<Nir[0];i++)
   {
    int region = ir[i];
    
    // Remove all the state rows that use the identifier in question from dt and wrt...
     for (int w=sIndex[i];w<sIndex[i+1];w++)
     {
      //int doc = STATE2(w,0);
      int topic = STATE2(w,1);
      int word = STATE2(w,3);
      
      //DT2(doc,topic) -= 1;
      WRT3(word,region,topic) -= 1;
      //mt[topic] -= 1;
      mr[region] -= 1;
     }

    // Calculate the distribution - have to loop the rows...
     float sum = float(NdistR[0]);
     for (int r=0;r<NdistR[0];r++) distR[r] = 1.0;

     for (int w=sIndex[i];w<sIndex[i+1];w++)
     {
      int topic = STATE2(w,1);
      int word = STATE2(w,3);
      
      float nSum = 0.0;
      for (int r=0;r<NdistR[0];r++)
      {
       float val = (distR[r] / sum) * (WRT3(word,r,topic) + gamma);
       val /= mr[r] + (Nwrt[0]*Nwrt[2]*gamma);
       nSum += val;
       distR[r] = val;
      }
      sum = nSum;
     }

    // Draw from the distribution, update the ir matrix...
     float total = 0.0;
     for (region=0;region<NdistR[0]-1;region++)
     {
      total += distR[region]/sum;
      if (total>randomR[i]) break;
     }
     ir[i] = region;

    // Add all the rows that use the identifier in question to dr and wrt...
     for (int w=sIndex[i];w<sIndex[i+1];w++)
     {
      //int doc = STATE2(w,0);
      int topic = STATE2(w,1);
      int word = STATE2(w,3);

      //DT2(doc,topic) += 1;
      WRT3(word,region,topic) += 1;
      //mt[topic] += 1;
      mr[region] += 1;
     }
   }
  """
  
  # Move relevant variables into the local namespace...
  alpha = st.alpha
  beta = st.beta
  gamma = st.gamma
  state = st.state
  sIndex = st.sIndex
  ir = st.ir
  wrt = st.wrt
  mt = st.mt
  mr = st.mr
  dt = st.dt

  # Create extra variables as required...
  distT = numpy.empty(wrt.shape[2], dtype = numpy.float_)
  distR = numpy.empty(wrt.shape[1], dtype = numpy.float_)
  
  randomT = numpy.random.random(state.shape[0])
  randomR = numpy.random.random(ir.shape[0])

  # Run it...
  weave.inline(code, ['alpha', 'beta', 'gamma', 'state', 'sIndex', 'ir', 'wrt', 'mt', 'mr', 'dt', 'distT', 'distR', 'randomT', 'randomR'])



def gibbs(st, iterCount, tCount, rCount, next, randMemUsage = 64*1024*1024):
  # Code...
  code = """
  // Iterate the given number of times...
   for (int iter=0;iter<iters;iter++)
   {
    // t phase - iterate the state matrix...
     for (int tp=0;tp<tCount;tp++)
     {
      for (int w=0;w<Nstate[0];w++)
      {
       int doc = STATE2(w,0);
       int topic = STATE2(w,1);
       int region = ir[STATE2(w,2)];
       int word = STATE2(w,3);

       // Remove the sample from the relevant data structures...
        DT2(doc,topic) -= 1;
        WRT3(word,region,topic) -= 1;
        mt[topic] -= 1;
        //mr[region] -= 1;

       // Calculate the probability distribution based on the current state, unnormalised but with a sum so it can be normalised in use...
        float sum = 0.0;
        for (int t=0;t<NdistT[0];t++)
        {
         float val = (DT2(doc,t) + alpha) * (WRT3(word,region,t) + beta);
         val /= mt[t] + (Nwrt[0]*Nwrt[1]*beta);
         sum += val;
         distT[t] = val;
        }

       // Draw a sample from the distribution, update the state accordingly...
        float total = 0.0;
        float limit = RANDOMT3(iter,tp,w) * sum;
        for (topic=0;topic<NdistT[0]-1;topic++)
        {
         total += distT[topic];
         if (total>limit) break;
        }
        STATE2(w,1) = topic;

       // Add the sample into the relevant data structures...
        DT2(doc,topic) += 1;
        WRT3(word,region,topic) += 1;
        mt[topic] += 1;
        //mr[region] += 1;
      }
     }


    // r phase - iterate the ir matrix...
     for (int rp=0;rp<rCount;rp++)
     {
      for (int i=0;i<Nir[0];i++)
      {
       int region = ir[i];

       // Remove all the state rows that use the identifier in question from dt and wrt...
        for (int w=sIndex[i];w<sIndex[i+1];w++)
        {
         //int doc = STATE2(w,0);
         int topic = STATE2(w,1);
         int word = STATE2(w,3);

         //DT2(doc,topic) -= 1;
         WRT3(word,region,topic) -= 1;
         //mt[topic] -= 1;
         mr[region] -= 1;
        }

       // Calculate the distribution - have to loop the rows...
        float sum = float(NdistR[0]);
        for (int r=0;r<NdistR[0];r++) distR[r] = 1.0;

        for (int w=sIndex[i];w<sIndex[i+1];w++)
        {
         int topic = STATE2(w,1);
         int word = STATE2(w,3);

         float nSum = 0.0;
         if ((sum>1e3)||(sum<1e-3)) // Only divide when needed.
         {
          for (int r=0;r<NdistR[0];r++)
          {
           float val = (distR[r] / sum) * (WRT3(word,r,topic) + gamma);
           val /= mr[r] + (Nwrt[0]*Nwrt[2]*gamma);
           nSum += val;
           distR[r] = val;
          }
         }
         else
         {
          for (int r=0;r<NdistR[0];r++)
          {
           float val = distR[r] * (WRT3(word,r,topic) + gamma);
           val /= mr[r] + (Nwrt[0]*Nwrt[2]*gamma);
           nSum += val;
           distR[r] = val;
          }
         }
         sum = nSum;
        }

       // Draw from the distribution, update the ir matrix...
        float total = 0.0;
        float limit = RANDOMR3(iter,rp,i) * sum;
        for (region=0;region<NdistR[0]-1;region++)
        {
         total += distR[region];
         if (total>limit) break;
        }
        ir[i] = region;

       // Add all the rows that use the identifier in question to dr and wrt...
        for (int w=sIndex[i];w<sIndex[i+1];w++)
        {
         //int doc = STATE2(w,0);
         int topic = STATE2(w,1);
         int word = STATE2(w,3);

         //DT2(doc,topic) += 1;
         WRT3(word,region,topic) += 1;
         //mt[topic] += 1;
         mr[region] += 1;
        }
      }
     }
   }
  """

  # Move relevant variables into the local namespace...
  alpha = st.alpha
  beta = st.beta
  gamma = st.gamma
  state = st.state
  sIndex = st.sIndex
  ir = st.ir
  wrt = st.wrt
  mt = st.mt
  mr = st.mr
  dt = st.dt

  # Create extra variables as required...
  distT = numpy.empty(wrt.shape[2], dtype = numpy.float_)
  distR = numpy.empty(wrt.shape[1], dtype = numpy.float_)

  # Run it...
  chunkSize = randMemUsage/(8*(tCount*state.shape[0] + rCount*ir.shape[0])) + 1
  while iterCount!=0:
    iters = min(chunkSize,iterCount)
    iterCount -= iters

    randomT = numpy.random.random((iters,tCount,state.shape[0]))
    randomR = numpy.random.random((iters,rCount,ir.shape[0]))
    
    weave.inline(code, ['iters', 'tCount', 'rCount', 'alpha', 'beta', 'gamma', 'state', 'sIndex', 'ir', 'wrt', 'mt', 'mr', 'dt', 'distT', 'distR', 'randomT', 'randomR'])

    next(iters)



def fitModel(state,params,next):
  """Given a state object generates samples."""
  iniGibbs(state)
  next()

  if params.burnIn>params.lag:
    gibbs(state,params.burnIn-params.lag,params.iterT,params.iterR,next)

  for i in xrange(params.samples):
    gibbs(state,params.lag,params.iterT,params.iterR,next)
    state.sample()
    next()



def fit(corpus, params, callback = None):
  """Complete fitting function - given a corpus fits a model. params is a Params object. callback if provided should take two numbers - the first is the number of iterations done, the second the number of iterations that need to be done; used to report progress. Note that it will probably not be called for every iteration, as that would be frightfully slow."""

  # Class to allow progress to be reported...
  class Reporter:
    def __init__(self,params,callback):
      self.doneIters = 0
      self.totalIters = params.runs * (1 + params.burnIn + params.samples + (params.samples-1)*params.lag)
      self.callback = callback

      if self.callback:
        self.callback(self.doneIters,self.totalIters)

    def next(self,amount = 1):
      self.doneIters += amount
      if self.callback:
        self.callback(self.doneIters,self.totalIters)

  report = Reporter(params,callback)
  s = shared.State(corpus)

  # Iterate and do each of the runs...
  for r in xrange(params.runs):
    ss = shared.State(s)
    fitModel(ss,params,report.next)
    s.absorbClone(ss)

  # Extract the final model into the corpus...
  s.extractModel(corpus)



def iniGibbsDoc(state, tCount, ir, wrt, alpha):
  # Code...
  code = """
  // Draw a new region assignment for each identifier...
   for (int i=0;i<Ni2r[0];i++)
   {
    float total = 0.0;
    for (i2r[i]=0; i2r[i]<(Nir[1]-1); i2r[i]++)
    {
     total += IR2(i,i2r[i]);
     if (total>randomR[i]) break;
    }
   }
  
  // Iterate the state matrix and reassign topics...
   for (int w=0;w<Nstate[0];w++)
   {
    int region = i2r[STATE2(w,1)];
    int word = STATE2(w,2);

    // Calculate the probability distribution based on the current state, unnormalised but with a sum so it can be normalised in use...
     float sum = 0.0;
     for (int t=0;t<Ndist[0];t++)
     {
      float val = (tCount[t] + alpha) * WRT3(word,region,t);
      sum += val;
      dist[t] = val;
     }

    // Draw a sample from the distribution, update the state accordingly...
     int topic = 0;
     float total = 0.0;
     for (;topic<Ndist[0]-1;topic++)
     {
      total += dist[topic]/sum;
      if (total>randomT[w]) break;
     }
     STATE2(w,0) = topic;

    // Add the sample into the documents topic count array...
     tCount[topic] += 1;
   }
  """

  # Create extra variables needed...
  i2r = numpy.empty(ir.shape[0], dtype = numpy.int_)
  dist = numpy.empty(wrt.shape[2], dtype = numpy.float_)

  randomT = numpy.random.random(state.shape[0])
  randomR = numpy.random.random(ir.shape[0])

  # Run it...
  weave.inline(code, ['state', 'tCount', 'ir', 'wrt', 'alpha', 'i2r', 'dist', 'randomT', 'randomR'])



def gibbsDoc(state, tCount, ir, wrt, iters, alpha):
  # Code...
  code = """
  // Iterate...
   for (int iter=0;iter<innerIters;iter++)
   {
    // Draw a new region assignment for each identifier...
     for (int i=0;i<Ni2r[0];i++)
     {
      float total = 0.0;
      for (i2r[i]=0; i2r[i]<(Nir[1]-1); i2r[i]++)
      {
       total += IR2(i,i2r[i]);
       if (total>RANDOMR2(iter,i)) break;
      }
     }

    // Iterate the state matrix and reassign topics...
     for (int w=0;w<Nstate[0];w++)
     {
      int topic = STATE2(w,0);
      int region = i2r[STATE2(w,1)];
      int word = STATE2(w,2);

      // Remove the sample from the documents topic count array...
       tCount[topic] -= 1;

      // Calculate the probability distribution based on the current state, unnormalised but with a sum so it can be normalised in use...
       float sum = 0.0;
       for (int t=0;t<Ndist[0];t++)
       {
        float val = (tCount[t] + alpha) * WRT3(word,region,t);
        sum += val;
        dist[t] = val;
       }

      // Draw a sample from the distribution, update the state accordingly...
       float total = 0.0;
       for (topic=0;topic<Ndist[0]-1;topic++)
       {
        total += dist[topic]/sum;
        if (total>RANDOMT2(iter,w)) break;
       }
       STATE2(w,0) = topic;

      // Add the sample into the documents topic count array...
       tCount[topic] += 1;
     }
   }
  """

  # Create extra variables needed...
  i2r = numpy.empty(ir.shape[0], dtype = numpy.int_)
  dist = numpy.empty(wrt.shape[2], dtype = numpy.float_)

  chunkSize = (64*1024*1024)/(8*(state.shape[0] + ir.shape[0])) + 1
  while iters!=0:
    innerIters = min((iters,chunkSize))
    iters -= innerIters
    
    randomT = numpy.random.random((innerIters,state.shape[0]))
    randomR = numpy.random.random((innerIters,ir.shape[0]))

    # Run it, doing all the iterations...
    weave.inline(code, ['innerIters', 'state', 'tCount', 'ir', 'wrt', 'alpha', 'i2r', 'dist', 'randomT', 'randomR'])

  return i2r



def fitModelDoc(state, tCount, irR, wrtT, wrt, alpha, params, norm):
  samples = []

  # Construct an array for quickly calculating normalising constants...
  if norm<0.0:
    logFact = numpy.log(numpy.arange(state.shape[0]+1))
    logFact[0] = 0.0
    for i in xrange(1,logFact.shape[0]): logFact[i] += logFact[i-1]
  
  # Initialise, do the iterations, collect samples...
  iniGibbsDoc(state,tCount,irR,wrtT,alpha)

  if params.burnIn>params.lag:
    gibbsDoc(state,tCount,irR,wrtT,params.burnIn-params.lag,alpha)

  for i in xrange(params.samples):
    i2r = gibbsDoc(state,tCount,irR,wrtT,params.lag,alpha)

    # Make a sample - need a copy of the topics distribution plus the negative log likelihood of each region given the current assignment...
    if norm<0.0:
      # Standard multinomial used for per-region negative log likelihood - suffers from regions having lots of samples always being less likelly than regions with a few...
      ## Marginalise etc. to get log P(w,t|r)...
      nll_wrt = wrt.copy() * (tCount.astype(numpy.float_) + alpha)
      for r in xrange(nll_wrt.shape[1]):
        nll_wrt[:,r,:] /= nll_wrt[:,r,:].sum()
      nll_wrt = numpy.log(nll_wrt)

      ## Sum for all samples...
      nlr = numpy.zeros(wrt.shape[1],dtype=numpy.float_)
      count_wr = numpy.zeros((wrt.shape[0],wrt.shape[1]),dtype=numpy.int_)
      for i in xrange(state.shape[0]):
        t = state[i,0]
        r = i2r[state[i,1]]
        w = state[i,2]
        nlr[r] += nll_wrt[w,r,t]
        count_wr[w,r] += 1

      ## Normalise...
      for r in xrange(nlr.shape[0]):
        nlr[r] += logFact[count_wr[:,r].sum()]
        nlr[r] -= logFact[count_wr[:,r]].sum()

      ## Make *negative* log likelihood...
      nlr *= -1.0
    else:
      # Re-weighted multinomial, which assigns negative log likelihoods to regions that are size agnostic...
      ## Marginalise etc. to get log P(w,t|r)...
      nll_wrt = wrt.copy() * (tCount.astype(numpy.float_) + alpha)
      for r in xrange(nll_wrt.shape[1]):
        nll_wrt[:,r,:] /= nll_wrt[:,r,:].sum()
      nll_wrt = numpy.log(nll_wrt)
      
      ## Sum how many samples are in each region...
      count_wrt = numpy.zeros(wrt.shape,dtype=numpy.float_)
      for i in xrange(state.shape[0]):
        t = state[i,0]
        r = i2r[state[i,1]]
        w = state[i,2]
        count_wrt[w,r,t] += 1.0

      ## Normalise the regions...
      for r in xrange(count_wrt.shape[1]):
        rSum = count_wrt[:,r,:].sum()
        if rSum>0.5: count_wrt[:,r,:] *= norm/rSum

      ## Sum the per-region multinomial terms...
      nlr = numpy.empty(wrt.shape[1],dtype=numpy.float_)
      nlr = (nll_wrt*count_wrt).sum(axis=2).sum(axis=0)
      
      ## Normalise each region...
      nlr += scipy.special.gammaln(norm + 1.0)
      nlr -= scipy.special.gammaln(count_wrt + 1.0).sum(axis=2).sum(axis=0)

      ## Make *negative* log likelihood...
      nlr *= -1.0

    samples.append((tCount.copy(),nlr,count_wrt.sum(axis=2).sum(axis=0)))

  # Return a list of samples...
  return samples



def fitDoc(doc, ir, wrt, alpha, params, norm):
  """Given a document, the two parts of a model (ir and wrt) plus an alpha value and params object this Gibbs samples under the assumption that ir and wrt are correct, to determine the documents model, i.e. the multinomial from which topics are drawn for the given document. norm indicates the method used to calculate the region negative log likelihood - negtaive means it uses the standard multinomial distribution, positive means it renormalises the samples to be weighted such that there are norm many samples all together, and then uses an updated multinomial distribution that deals with non-integers. This allows the comparison of regions with very different numbers of samples in."""
  # Normalise the relevant rows/columns of ir and wrt to generate the matrices we need...
  irR = (ir.T/ir.sum(axis=1)).T
  wrtT = wrt.copy()
  for t in xrange(wrt.shape[2]): wrtT[:,:,t] /= wrtT[:,:,t].sum()

  # Construct the state matrix - each row is (topic,identifier,word)...
  state = numpy.empty((doc.getSampleCount(),3),dtype=numpy.uint)
  index = 0
  words = doc.getWords()
  for w in xrange(words.shape[0]):
    for c in xrange(words[w,2]):
      state[index,0] = 10000000 # Deliberate bad value, so bad initialisation would cause a crash.
      state[index,1] = words[w,0]
      state[index,2] = words[w,1]
      index += 1
  assert(index==state.shape[0])

  # Do all the runs, collate the samples...
  samples = []
  for r in xrange(params.getRuns()):
    tCount = numpy.zeros(wrt.shape[2],dtype=numpy.uint)
    samples += fitModelDoc(state,tCount,irR,wrtT,wrt,alpha,params,norm)

  # Merge the samples to get the final model, write it into the given doc object...
  model = numpy.zeros(wrt.shape[2],dtype=numpy.float_)
  for i,sample in enumerate(samples):
    model += ((sample[0].astype(numpy.float_) + alpha) - model) / float(i+1)
  doc.setModel(model)

  # Combine the region probability estimates, write the region negative log likelihoods to the document...
  doc.nllRegion = numpy.zeros(wrt.shape[1],dtype=numpy.float_)
  doc.sizeRegion = numpy.zeros(wrt.shape[1],dtype=numpy.float_)
  for i,sample in enumerate(samples):
    doc.nllRegion += (sample[1] - doc.nllRegion) / float(i+1)
    doc.sizeRegion += (sample[2].astype(numpy.float_) - doc.sizeRegion) / float(i+1)
