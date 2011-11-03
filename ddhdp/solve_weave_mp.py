# Copyright 2011 Tom SF Haines

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



import time
import multiprocessing as mp
import multiprocessing.synchronize # To make sure we have all the functionality.

from solve_shared import Params, State
from solve_weave import gibbs_run
from model import DocModel



def gibbs_run_wrap(state, doneIters):
  """Wrapper around gibbs_run to make it suitable for multiprocessing."""
  def next(amount = 1):
    doneIters.value += amount

  gibbs_run(state, next)
  return state



def gibbs_all_mp(state, callback = None):
  """Identical to gibbs_all, except it does each run in a different process, to fully stress the computer."""

  # Need the parameters object so we do the correct amount of work...
  params = state.getParams()
  
  # Create a pool of worker processes...
  pool = mp.Pool()

  # Create a value for sub-processes to report back their progress with...
  manager = mp.Manager()
  doneIters = manager.Value('i',0)
  totalIters = params.runs * (max((params.burnIn,params.lag)) + params.samples + (params.samples-1)*params.lag)

  # Create a callback for when a job completes...
  def onComplete(s):
    state.absorbClone(s)

  # Create all the jobs, wait for their completion, report progress...
  try:
    jobs = []
    for r in xrange(params.runs):
      jobs.append(pool.apply_async(gibbs_run_wrap,(State(state),doneIters), callback = onComplete))
  finally:
    # Close the pool and wait for all the jobs to complete...
    pool.close()
    while len(jobs)!=0:
      if jobs[0].ready():
        del jobs[0]
        continue
      time.sleep(0.01)
      if callback!=None: callback(doneIters.value,totalIters)
    pool.join()



def gibbs_doc_mp(model, doc, params = None, callback = None):
  """Runs Gibbs iterations on a single document, by sampling with a prior constructed from each sample in the given Model. params applies to each sample, so should probably be much more limited than usual - the default if its undefined is to use 1 run and 1 sample and a burn in of only 500. Returns a DocModel with all the relevant samples in."""

  # Initialisation stuff - handle params, create the state and the DocModel object, plus a reporter...
  if params==None:
    params = Params()
    params.runs = 1
    params.samples = 1
    params.burnIn = 500

  state = State(doc, params)
  dm = DocModel()

  # Create a pool of worker processes...
  pool = mp.Pool()

  # Create a value for sub-processes to report back their progress with...
  manager = mp.Manager()
  doneIters = manager.Value('i',0)
  totalIters = model.sampleCount() * params.runs * (params.burnIn + params.samples + (params.samples-1)*params.lag)

  # Create a callback for when a job completes...
  def onComplete(s):
    dm.addFrom(s.getModel())

  # Create all the jobs, wait for their completion, report progress...
  try:
    jobs = []
    for sample in model.sampleList():
      tempState = State(state)
      tempState.setGlobalParams(sample)
      tempState.addPrior(sample)
      jobs.append(pool.apply_async(gibbs_run_wrap,(tempState,doneIters), callback = onComplete))
  finally:
    # Close the pool and wait for all the jobs to complete...
    pool.close()
    while len(jobs)!=0:
      if jobs[0].ready():
        del jobs[0]
        continue
      time.sleep(0.01)
      if callback!=None: callback(doneIters.value,totalIters)
    pool.join()

  # Return...
  return dm
