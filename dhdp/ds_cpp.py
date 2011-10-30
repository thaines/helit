# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



from scipy import weave
import unittest
from utils.start_cpp import start_cpp

from dp_utils.dp_utils import dp_utils_code



# Data structure for storing the state of the model, for use with the c++ Gibbs sampling code. A custom structure is used for speed and to keep the code clean...
ds_code = dp_utils_code + start_cpp() + """

// Details specific for a topic - basically its multinomial and some helper stuff...
class Topic
{
 public:
   Topic():wc(0) {}
  ~Topic() {delete[] wc;}

  int id; // Used when moving this data structure to and from python.
  int * wc; // Indexed by word id this contains the count of how many words with that id are assigned to this topic - from the prior the multinomial can hence be worked out.
  int wcTotal; // Sum of above.

  float prob; // Helper for resampling.
  float probAux; // "
};



class Sample; // Predeclaration required for below.

// Stuff for the clustering - basically everything that goes into a cluster, including its DP...
class ClusterInst
{
 public:
   ClusterInst():topic(0) {}
  ~ClusterInst()
  {
   if (topic) topic->DecRef();
  }

  ItemRef<Topic,Conc> * GetTopic() {return topic;}

  void SetTopic(ItemRef<Topic,Conc> * nt, bool safe=true)
  {
   if (safe&&nt) nt->IncRef();
   if (safe&&topic) topic->DecRef();
   topic = nt;
  }

  int id; // Used when moving this data structure to and from python.

  float prob; // Helper for resampling.
  Sample * first; // For a temporary linked list when resampling the topic.

 protected:
  ItemRef<Topic,Conc> * topic;
};

class Cluster : public ListRef<ClusterInst,Conc>
{
 public:
   Cluster() {}
  ~Cluster() {}

  int id; // Used when moving this data structure to and from python.

  float prob; // Helper for resampling.
};



class DocInst
{
 public:
   DocInst():clusterInst(0) {}
  ~DocInst()
  {
   if (clusterInst) clusterInst->DecRef();
  }

  ItemRef<ClusterInst,Conc> * GetClusterInst() {return clusterInst;}

  void SetClusterInst(ItemRef<ClusterInst,Conc> * nci, bool safe=true)
  {
   if (safe&&nci) nci->IncRef();
   if (safe&&clusterInst) clusterInst->DecRef();
   
   clusterInst = nci;
  }

  int id; // Used when moving this data structure to and from python.

  float prob; // Helper for resampling.

  ItemRef<Topic,Conc> * topic; // Temporary value, used to store the topic whilst disconnected from a cluster inst during the cluster resampling process.

  Sample * first; // For a temporary linked list when resampling the cluster instance.

 protected:
  ItemRef<ClusterInst,Conc> * clusterInst;
};

class Sample
{
 public:
   Sample():word(-1),docInst(0) {}
  ~Sample()
  {
   if (docInst) docInst->DecRef();
  }

  int GetWord() {return word;}
  void SetWord(int w) {word = w;}

  ItemRef<DocInst,Conc> * GetDocInst() {return docInst;}
  void SetDocInst(ItemRef<DocInst,Conc> * ndi, bool safe=true)
  {
   if (safe&&ndi)
   {
    ndi->IncRef();
    ItemRef<Topic,Conc> * topic = ndi->GetClusterInst()->GetTopic();
    topic->wcTotal += 1;
    topic->wc[word] += 1;
   }
   
   if (safe&&docInst)
   {
    ItemRef<Topic,Conc> * topic = docInst->GetClusterInst()->GetTopic();
    topic->wcTotal -= 1;
    topic->wc[word] -= 1;
    docInst->DecRef();
   }
   
   docInst = ndi;
  }

  Sample * next; // Used for a temporary linked list whilst resampling higher up the hierachy.

 protected:
  int word;
  ItemRef<DocInst,Conc> * docInst;
};

class Document : public ListRef<DocInst,Conc>
{
 public:
   Document():cluster(0),sampleCount(0),sample(0) {}
  ~Document()
  {
   if (cluster) cluster->DecRef();
   delete[] sample;
  }

  ItemRef<Cluster,Conc> * GetCluster() {return cluster;}
  void SetCluster(ItemRef<Cluster,Conc> * nc, bool safe=true)
  {
   if (safe&&nc) nc->IncRef();
   if (safe&&cluster) cluster->DecRef();
   cluster = nc;
  }

  int SampleCount() {return sampleCount;}
  Sample & GetSample(int i) {return sample[i];}
  void SetSamples(int count,Sample * array) // Takes owenership of the given array, must be declared with new[]
  {
   sampleCount = count;
   delete[] sample;
   sample = array;
  }
  
 protected:
  ItemRef<Cluster,Conc> * cluster;
  int sampleCount;
  Sample * sample; // Declared with new[]
};



// Final State object - represents an entire model...
class State
{
 public:
   State():seperateClusterConc(false), seperateDocumentConc(false), oneCluster(false), calcBeta(false), beta(0), betaSum(0.0), docCount(0), doc(0) {}
  ~State()
  {
   for (int d=0;d<docCount;d++)
   {
    doc[d].SetSamples(0,0);
    while (doc[d].Size()!=0) doc[d].First()->Suicide();
    doc[d].SetCluster(0);
   }
   delete[] doc;
   
   while (clusters.Size()!=0)
   {
    ItemRef<Cluster,Conc> * victim = clusters.First();
    while (victim->Size()!=0) victim->First()->Suicide();
    victim->Suicide();
   }
   while (topics.Size()!=0) topics.First()->Suicide();

   delete[] beta;
  }

  // Algorithm behavioural flags, indicate if concentration parameters for clusters and documents are shared or calculated on a per entity basis, and if we should fix it to a single cluster to acheive HDP-like behaviour...
   bool dnrDocInsts;
   bool dnrCluInsts;
   bool seperateClusterConc;
   bool seperateDocumentConc;
   bool oneCluster;
   bool calcBeta;

  // Parameters - only need these once as most can be stored where they are needed...
   float * beta;
   float betaSum;
   Conc rho; // Needed for new clusters.

   int wordCount; // Number of unique word types.
  
  // Basic DP that provides topics, contains multinomial distributions etc...
   ListRef<Topic,Conc> topics; 

  // DDP that provides clusters - you draw DP's from this...
   ListRef<Cluster,Conc> clusters;

  // All the documents...
   int docCount;
   Document * doc; // Declared with new[]
};



// Goes through the given State object and verifies that the ref counts match the number of references - for debugging. (Obviously no good if there is a prior.) printf's out any errors...
void VerifyState(State & state)
{
 // Verify topic counts...
  int * counts = new int[state.topics.Size()];
  {
   ItemRef<Topic,Conc> * targ = state.topics.First();
   int id = 0;
   while (targ->Valid())
   {
    targ->id = id;
    counts[id] = 0;
    
    id += 1;
    targ = targ->Next();
   }
   if (id!=state.topics.Size()) printf("Size of topics is incorrect\\n");
  }

  {
   ItemRef<Cluster,Conc> * targ = state.clusters.First();
   while (targ->Valid())
   {
    ItemRef<ClusterInst,Conc> * targ2 = targ->First();
    while (targ2->Valid())
    {
     if (targ2->GetTopic()) counts[targ2->GetTopic()->id] += 1;
     targ2 = targ2->Next();
    }

    targ = targ->Next();
   }
  }

  {
   ItemRef<Topic,Conc> * targ = state.topics.First();
   int total = 0;
   while (targ->Valid())
   {
    total += targ->RefCount();
    if (counts[targ->id]!=targ->RefCount())
    {
     printf("Topic %i has the wrong refcount\\n",targ->id);
    }
    targ = targ->Next();
   }
   if (total!=state.topics.RefTotal()) printf("Topics ref-total is incorrect\\n");
  }

  delete[] counts;


 // Verify cluster counts...
  counts = new int[state.clusters.Size()];
  {
   ItemRef<Cluster,Conc> * targ = state.clusters.First();
   int id = 0;
   while (targ->Valid())
   {
    targ->id = id;
    counts[id] = 0;

    id += 1;
    targ = targ->Next();
   }
   if (id!=state.clusters.Size()) printf("Size of clusters is incorrect\\n");
  }

  for (int d=0;d<state.docCount;d++)
  {
   if (state.doc[d].GetCluster())
   {
    counts[state.doc[d].GetCluster()->id] += 1;
   }
  }

  {
   ItemRef<Cluster,Conc> * targ = state.clusters.First();
   int total = 0;
   while (targ->Valid())
   {
    total += targ->RefCount();
    if (counts[targ->id]!=targ->RefCount())
    {
     printf("Cluster %i has the wrong refcount\\n",targ->id);
    }
    targ = targ->Next();
   }
   if (total!=state.clusters.RefTotal()) printf("Clusters ref-total is incorrect\\n");
  }

  delete[] counts;


 // Verify cluster instance counts...
  int cluInstSum = 0;
  {
   ItemRef<Cluster,Conc> * targ = state.clusters.First();
   while (targ->Valid())
   {
    cluInstSum += targ->Size();
    targ = targ->Next();
   }
  }

  counts = new int[cluInstSum];
  {
   ItemRef<Cluster,Conc> * targ = state.clusters.First();
   int id = 0;
   while (targ->Valid())
   {
    ItemRef<ClusterInst,Conc> * targ2 = targ->First();
    int startId = id;
    while (targ2->Valid())
    {
     targ2->id = id;
     counts[id] = 0;

     id += 1;
     targ2 = targ2->Next();
    }

    if ((id-startId)!=targ->Size()) printf("Size of cluster instance %i is incorrect\\n",targ->id);
    
    targ = targ->Next();
   }
  }

  for (int d=0;d<state.docCount;d++)
  {
   ItemRef<DocInst,Conc> * targ = state.doc[d].First();
   while (targ->Valid())
   {
    if (targ->GetClusterInst())
    {
     counts[targ->GetClusterInst()->id] += 1;
    }

    targ = targ->Next();
   }
  }

  {
   ItemRef<Cluster,Conc> * targ = state.clusters.First();
   while (targ->Valid())
   {
    int total = 0;
    ItemRef<ClusterInst,Conc> * targ2 = targ->First();
    while (targ2->Valid())
    {
     total += targ2->RefCount();
     if (targ2->RefCount()!=counts[targ2->id])
     {
      printf("Cluster instance %i of cluster %i has a bad refcount\\n",targ2->id,targ->id);
     }

     targ2 = targ2->Next();
    }

    if (total!=targ->RefTotal()) printf("Cluster instance %i has a bad ref total\\n",targ->id);

    targ = targ->Next();
   }
  }

  delete[] counts;
  

 // Verify document instance counts...
  for (int d=0;d<state.docCount;d++)
  {
   counts = new int[state.doc[d].Size()];
   {
    ItemRef<DocInst,Conc> * targ = state.doc[d].First();
    int id = 0;
    while (targ->Valid())
    {
     targ->id = id;
     counts[id] = 0;
     
     id += 1;
     targ = targ->Next();
    }
    if (id!=state.doc[d].Size()) printf("Doc %i has an invalid size\\n",d);
   }

   for (int s=0;s<state.doc[d].SampleCount();s++)
   {
    Sample & sam = state.doc[d].GetSample(s);
    if (sam.GetDocInst())
    {
     counts[sam.GetDocInst()->id] += 1;
    }
   }

   {
    ItemRef<DocInst,Conc> * targ = state.doc[d].First();
    int total = 0;
    while (targ->Valid())
    {
     total += targ->RefCount();
     if (targ->RefCount()!=counts[targ->id])
     {
      printf("Document %i, instance %i has a bad ref count\\n",d,targ->id);
     }

     targ = targ->Next();
    }
    if (total!=state.doc[d].RefTotal()) printf("Doc %i has an invalid ref total\\n",d);
   }

   delete[] counts;
  }
}

"""



class TestDS(unittest.TestCase):
  """Test code for the data structure."""
  def test_compile(self):
    code = start_cpp(dual_hdp_ds) + """
    State state;
    """
    weave.inline(code, support_code=dual_hdp_ds)



# If this file is run do the unit tests...
if __name__ == '__main__':
  unittest.main()
