# -*- coding: utf-8 -*-

# Copyright (c) 2011, Tom SF Haines
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import multiprocessing as mp
import multiprocessing.synchronize # To make sure we have all the functionality.

import types
import marshal

import unittest



def repeat(x):
  """A generator that repeats the input forever - can be used with the mp_map function to give data to a function that is constant."""
  while True: yield x



def run_code(code,args):
  """Internal use function that does the work in each process."""
  code = marshal.loads(code)
  func = types.FunctionType(code, globals(), '_')

  return func(*args)



def mp_map(func, *iters, **keywords):
  """A multiprocess version of the map function. Note that func must limit itself to the data provided - if it accesses anything else (globals, locals to its definition.) it will fail. There is a repeat generator provided in this module to work around such issues. Note that, unlike map, this iterates the length of the shortest of inputs, rather than the longest - whilst this makes it not a perfect substitute it makes passing constant argumenmts easier as they can just repeat for infinity."""
  if 'pool' in keywords: pool = keywords['pool']
  else: pool = mp.Pool()

  code = marshal.dumps(func.func_code)

  jobs = []
  for args in zip(*iters):
    jobs.append(pool.apply_async(run_code,(code,args)))

  for i in xrange(len(jobs)):
    jobs[i] = jobs[i].get()

  return jobs



class TestMpMap(unittest.TestCase):
  def test_simple1(self):
    data = ['a','b','c','d']
    
    def noop(data):
      return data
  
    data_noop = mp_map(noop, data)
    
    self.assertEqual(data, data_noop)

  def test_simple2(self):
    data = [x for x in xrange(1000)]

    data_double = mp_map(lambda a: a*2, data)

    self.assertEqual(map(lambda a: a*2,data), data_double)

  def test_gen(self):
    def gen():
      for i in xrange(100): yield i

    data_double = mp_map(lambda a: a*2, gen())

    self.assertEqual(map(lambda a: a*2,gen()), data_double)

  def test_repeat(self):
    def mult(a,b):
      return a*b

    data = [x for x in xrange(50,5000,5)]

    data_triple = mp_map(mult, data, repeat(3))

    self.assertEqual(map(lambda a: a*3,data),data_triple)

  def test_none(self):
    data = []
    data_sqr = mp_map(lambda x: x*x, data)

    self.assertEqual([],data_sqr)



if __name__ == '__main__':
  unittest.main()
