# -*- coding: utf-8 -*-

# Copyright 2010 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



from params import Kernel
from params import Params
from params_sets import ParamsRange
from params_sets import ParamsSet
from dataset import Dataset
from model import Model
from loo import looPair
from loo import looPairRange
from loo import looPairBrute
from loo import looPairSelect
from multiclass import MultiModel

import smo



def solvePair(params,dataset,negLabel,posLabel):
  """Solves for a pair of labels - you provide a parameters object, a data set and the labels to assign to -1 and +1 respectivly. It then returns a Model object."""
  s = smo.SMO()
  s.setParams(params)
  s.setData(dataset.getTrainData(negLabel,posLabel))

  s.solve()

  return s.getModel()
