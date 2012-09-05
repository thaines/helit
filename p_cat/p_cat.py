# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



from prob_cat import ProbCat

try: from classify_gaussian import ClassifyGaussian
except: pass

try: from classify_kde import ClassifyKDE
except: pass

try: from classify_bag_kde import ClassifyBagKDE
except: pass

try: from classify_dpgmm import ClassifyDPGMM
except: pass

try: from classify_df import ClassifyDF
except: pass

try: from classify_df_kde import ClassifyDF_KDE
except: pass
