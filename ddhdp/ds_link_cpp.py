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



from scipy import weave
import unittest
from utils.start_cpp import start_cpp

from ds_cpp import ds_code



# Provides code for converting from the python to the C++ data structure, and back again - this is so the data can be stored in a suitable form in both situations, though it comes at the expense of a complex conversion...
ds_link_code = ds_code + start_cpp() + """

// Helper for extracting a boolean from a Python object...
bool GetObjectBool(PyObject * obj, const char * name)
{
 PyObject * boolObj = PyObject_GetAttrString(obj, name);
 bool ret = boolObj==Py_True;
 Py_DECREF(boolObj);
 return ret;
}

// Helper converter for the Conc class and its python equivalent, PriorConcDP - just need to go one way. Given an object and the name of the variable in the object that is the PriorConcDP object...
void ConcPyToCpp(PyObject * obj, const char * name, Conc & out)
{
 PyObject * pyConc = PyObject_GetAttrString(obj,name);

 PyObject * alpha = PyObject_GetAttrString(pyConc,"_PriorConcDP__alpha");
 out.alpha = PyFloat_AsDouble(alpha);
 Py_DECREF(alpha);

 PyObject * beta = PyObject_GetAttrString(pyConc,"_PriorConcDP__beta");
 out.beta = PyFloat_AsDouble(beta);
 Py_DECREF(beta);

 PyObject * conc = PyObject_GetAttrString(pyConc,"_PriorConcDP__conc");
 out.conc = PyFloat_AsDouble(conc);
 Py_DECREF(conc);

 Py_DECREF(pyConc);
}



// Python -> C++ - given pointers to the State class and a State object...
// (The State object should be empty when passed.)
void StatePyToCpp(PyObject * from, State * to)
{
 // Extract the flags...
  to->dnrDocInsts = GetObjectBool(from,"dnrDocInsts");
  to->dnrCluInsts = GetObjectBool(from,"dnrCluInsts");
  to->seperateClusterConc = GetObjectBool(from,"seperateClusterConc");
  to->seperateDocumentConc = GetObjectBool(from,"seperateDocumentConc");
  to->oneCluster = GetObjectBool(from,"oneCluster");
  to->calcBeta = GetObjectBool(from,"calcBeta");
  to->calcCluBmn = GetObjectBool(from,"calcCluBmn");
  to->calcPhi = GetObjectBool(from,"calcPhi");
  to->resampleConcs = GetObjectBool(from,"resampleConcs");
  to->behSamples = GetObjectInt(from,"behSamples");


 // Extract all the parameters, though only rho and beta get stored in the state - others get used later when filling out other structures...
  Conc alpha;
  ConcPyToCpp(from,"alpha",alpha);

  PyArrayObject * beta = (PyArrayObject*)PyObject_GetAttrString(from,"beta");
  to->beta = new float[beta->dimensions[0]];
  to->betaSum = 0.0;
  for (int i=0;i<beta->dimensions[0];i++)
  {
   to->beta[i] = Float1D(beta,i);
   to->betaSum += to->beta[i];
  }
  Py_DECREF(beta);

  Conc gamma;
  ConcPyToCpp(from,"gamma",gamma);

  ConcPyToCpp(from,"rho",to->rho);

  Conc mu;
  ConcPyToCpp(from,"mu",mu);

  PyArrayObject * phi = (PyArrayObject*)PyObject_GetAttrString(from,"phi");
  to->phi = new float[phi->dimensions[0]];
  for (int i=0;i<phi->dimensions[0];i++)
  {
   to->phi[i] = Float1D(phi,i);
  }
  Py_DECREF(phi);


 // Number of behaviours...
  {
   PyObject * abnorms = PyObject_GetAttrString(from,"abnorms");
   to->behCount = 1 + PyDict_Size(abnorms);
   Py_DECREF(abnorms);
  }


 // Store the flag set matrix (Involves calling a python function.)...
  {
   PyObject * fia = PyObject_GetAttrString(from,"fia");
   PyObject * func = PyObject_GetAttrString(fia,"getFlagMatrix");

   to->flagSets = (PyArrayObject*)PyObject_CallObject(func, 0);

   Py_DECREF(func);
   Py_DECREF(fia);
  }


 // Create the topic objects...
  PyArrayObject * topicWord = (PyArrayObject*)PyObject_GetAttrString(from,"topicWord");
  PyArrayObject * topicUse = (PyArrayObject*)PyObject_GetAttrString(from,"topicUse");
  
  int topicCount = topicWord->dimensions[0];
  int wordCount = topicWord->dimensions[1];
  to->wordCount = wordCount;

  ItemRef<Topic,Conc> ** topicArray = new ItemRef<Topic,Conc>*[topicCount];
  for (int t=0;t<topicCount;t++)
  {
   ItemRef<Topic,Conc> * topic = to->topics.Append();
   topicArray[t] = topic;
   
   topic->id = t;
   topic->wc = new int[wordCount];
   topic->wcTotal = 0;
   for (int w=0;w<wordCount;w++)
   {
    int val = Int2D(topicWord,t,w);
    topic->wc[w] = val;
    topic->wcTotal += val;
   }
   topic->beh = 0;
   
   topic->IncRef(Int1D(topicUse,t));
  }

  Py_DECREF(topicUse);
  Py_DECREF(topicWord);

  PyObject * topicConc = PyObject_GetAttrString(from,"topicConc");
  to->topics.Body().alpha = gamma.alpha;
  to->topics.Body().beta = gamma.beta;
  to->topics.Body().conc = PyFloat_AsDouble(topicConc);
  Py_DECREF(topicConc);


 // Do the abnormal topics...
  PyArrayObject * abnormTopicWord = (PyArrayObject*)PyObject_GetAttrString(from, "abnormTopicWord");
  ItemRef<ClusterInst,Conc> ** abArray = new ItemRef<ClusterInst,Conc>*[to->behCount];
  for (int b=0;b<to->behCount;b++)
  {
   ItemRef<Topic,Conc> * topic = to->behTopics.Append();
   ItemRef<ClusterInst,Conc> * cluInst = to->behCluInsts.Append();
   abArray[b] = cluInst;
   topic->IncRef();
   cluInst->IncRef();
   cluInst->SetTopic(topic,false);

   topic->id = -1;
   topic->wc = new int[wordCount];
   topic->wcTotal = 0;
   for (int w=0;w<wordCount;w++)
   {
    int val = Int2D(abnormTopicWord,b,w);
    topic->wc[w] = val;
    topic->wcTotal += val;
   }
   topic->beh = b;
   
   cluInst->id = -1;
  }
  Py_DECREF(abnormTopicWord);


 // Now create the clusters...
  PyObject * cluster = PyObject_GetAttrString(from,"cluster");
  PyArrayObject * clusterUse = (PyArrayObject*)PyObject_GetAttrString(from,"clusterUse");
  int clusterCount = PyList_Size(cluster);

  ItemRef<Cluster,Conc> ** clusterArray = new ItemRef<Cluster,Conc>*[clusterCount];
  ItemRef<ClusterInst,Conc> *** clusterInstArray = new ItemRef<ClusterInst,Conc>**[clusterCount];
  for (int c=0;c<clusterCount;c++)
  {
   PyObject * cluEntry = PyList_GetItem(cluster,c);
   PyArrayObject * cluInst = (PyArrayObject*)PyTuple_GetItem(cluEntry,0);
   PyObject * cluConc = PyTuple_GetItem(cluEntry,1);
   PyArrayObject * cluBMN = (PyArrayObject*)PyTuple_GetItem(cluEntry,2);
   PyArrayObject * cluPriorBMN = (PyArrayObject*)PyTuple_GetItem(cluEntry,3);
   
   // Create the cluster instance...
    ItemRef<Cluster,Conc> * clu = to->clusters.Append();
    clu->id = c;
    clusterArray[c] = clu;
    clu->IncRef(Int1D(clusterUse,c));

   // Create the clusters topic instances, including filling in the counts...
    clusterInstArray[c] = new ItemRef<ClusterInst,Conc>*[cluInst->dimensions[0]];
    for (int ci=0;ci<cluInst->dimensions[0];ci++)
    {
     ItemRef<ClusterInst,Conc> * nci = clu->Append();
     nci->id = ci;
     clusterInstArray[c][ci] = nci;
     
     int topic = Int2D(cluInst,ci,0);
     int users = Int2D(cluInst,ci,1);

     if (topic!=-1) nci->SetTopic(topicArray[topic],false);
     nci->IncRef(users);
    }

   // Fill in the clusters concentration stuff...
    clu->Body().alpha = to->rho.alpha;
    clu->Body().beta  = to->rho.beta;
    clu->Body().conc  = PyFloat_AsDouble(cluConc);

   // Do the multinomial...
    float * bmn = new float[to->behCount];
    for (int b=0;b<to->behCount;b++)
    {
     bmn[b] = Float1D(cluBMN, b);
    }
    clu->SetBMN(bmn);

   // Do the prior on bmn...
    int * bmnPrior = new int[to->flagSets->dimensions[0]];
    for (int fs=0;fs<to->flagSets->dimensions[0];fs++)
    {
     bmnPrior[fs] = Int1D(cluPriorBMN, fs);
    }
    clu->SetBehCountPrior(bmnPrior);
  }
  
  Py_DECREF(clusterUse);
  Py_DECREF(cluster);

  PyObject * clusterConc = PyObject_GetAttrString(from,"clusterConc");
  to->clusters.Body().alpha = mu.alpha;
  to->clusters.Body().beta = mu.beta;
  to->clusters.Body().conc = PyFloat_AsDouble(clusterConc);
  Py_DECREF(clusterConc);


 // Finally, create the documents...
  PyObject * docList = PyObject_GetAttrString(from,"doc");
  to->docCount = PyList_Size(docList);
  delete[] to->doc;
  to->doc = new Document[to->docCount];

  for (int d=0;d<to->docCount;d++)
  {
   // Get the relevant entities...
    PyObject * fromDoc = PyList_GetItem(docList,d);
    Document & toDoc = to->doc[d];

   // Setup the link to the cluster...
    PyObject * clusterIndex = PyObject_GetAttrString(fromDoc,"cluster");
    int cluIndex = PyInt_AsLong(clusterIndex);
    Py_DECREF(clusterIndex);
    if (cluIndex!=-1) toDoc.SetCluster(clusterArray[cluIndex],false);

   // Prep the documents DP...
    PyArrayObject * use = (PyArrayObject*)PyObject_GetAttrString(fromDoc,"use");
    ItemRef<DocInst,Conc> ** docInstArray = new ItemRef<DocInst,Conc>*[use->dimensions[0]];
    for (int di=0;di<use->dimensions[0];di++)
    {
     ItemRef<DocInst,Conc> * docInst = toDoc.Append();
     docInst->id = di;
     docInstArray[di] = docInst;

     int ciBeh = Int2D(use,di,0);
     int ciIndex = Int2D(use,di,1);
     int ciUse = Int2D(use,di,2);

     if (ciBeh!=-1)
     {
      if (ciBeh==0)
      {
       docInst->SetClusterInst(clusterInstArray[cluIndex][ciIndex],false);
      }
      else
      {
       docInst->SetClusterInst(abArray[ciBeh]);
      }
     }
     docInst->IncRef(ciUse);
    }
    Py_DECREF(use);

    PyObject * docConc = PyObject_GetAttrString(fromDoc,"conc");
    toDoc.Body().alpha = alpha.alpha;
    toDoc.Body().beta = alpha.beta;
    toDoc.Body().conc = PyFloat_AsDouble(docConc);
    Py_DECREF(docConc);

   // Store the samples...
    PyArrayObject * samples = (PyArrayObject*)PyObject_GetAttrString(fromDoc,"samples");
    Sample * sArray = new Sample[samples->dimensions[0]];
    for (int s=0;s<samples->dimensions[0];s++)
    {
     int di = Int2D(samples,s,0);
     if (di!=-1) sArray[s].SetDocInst(docInstArray[di],false);
     
     sArray[s].SetWord(Int2D(samples,s,1));
    }
    toDoc.SetSamples(samples->dimensions[0],sArray);
    Py_DECREF(samples);

   // Do the abnormality vectors...
    PyArrayObject * behFlags = (PyArrayObject*)PyObject_GetAttrString(fromDoc,"behFlags");
    PyArrayObject * behCounts = (PyArrayObject*)PyObject_GetAttrString(fromDoc,"behCounts");

    unsigned char * bFlags = new unsigned char[behFlags->dimensions[0]];
    int * bCounts = new int[behCounts->dimensions[0]];

    for (int b=0;b<behFlags->dimensions[0];b++)
    {
     bFlags[b] = Byte1D(behFlags,b);
     bCounts[b] = Int1D(behCounts,b);
    }

    toDoc.SetBehFlags(bFlags);
    toDoc.SetFlagIndex(GetObjectInt(fromDoc,"behFlagsIndex"));
    toDoc.SetBehCounts(bCounts);

    Py_DECREF(behCounts);
    Py_DECREF(behFlags);

   // Clean up...
    delete[] docInstArray;
  }

  Py_DECREF(docList);


 // Some temporary storage...
  to->tempWord = new int[to->wordCount];


 // Clean up...
  for (int c=0;c<clusterCount;c++) delete[] clusterInstArray[c];
  delete[] clusterInstArray;
  delete[] clusterArray;
  delete[] abArray;
  delete[] topicArray;
}



// C++ -> Python - given pointers to the State class and a State object...
// Note that this assumes that the State object was created from the PyObject in the first place - if not it will almost certainly break.
void StateCppToPy(State * from, PyObject * to)
{
 // Update the initial values of alpha, gamma, rho and mu to the current values...
 {
  float alpha = from->doc[0].Body().conc;
  float gamma = from->topics.Body().conc;
  float rho = from->rho.conc;
  float mu = from->clusters.Body().conc;

  PyObject * pyAlpha = PyFloat_FromDouble(alpha);
  PyObject * pyGamma = PyFloat_FromDouble(gamma);
  PyObject * pyRho = PyFloat_FromDouble(rho);
  PyObject * pyMu = PyFloat_FromDouble(mu);

  PyObject * alphaStore = PyObject_GetAttrString(to, "alpha");
  PyObject * gammaStore = PyObject_GetAttrString(to, "gamma");
  PyObject * rhoStore = PyObject_GetAttrString(to, "rho");
  PyObject * muStore = PyObject_GetAttrString(to, "mu");
  
  PyObject_SetAttrString(alphaStore, "conc", pyAlpha);
  PyObject_SetAttrString(gammaStore, "conc", pyGamma);
  PyObject_SetAttrString(rhoStore, "conc", pyRho);
  PyObject_SetAttrString(muStore, "conc", pyMu);

  Py_DECREF(pyAlpha);
  Py_DECREF(pyGamma);
  Py_DECREF(pyRho);
  Py_DECREF(pyMu);
 }
 
 
 // Extract beta - it could of been updated...
  npy_intp size[2];
  size[0] = from->wordCount;
  PyArrayObject * beta = (PyArrayObject*)PyArray_SimpleNew(1,size,NPY_FLOAT);
  for (int i=0;i<from->wordCount;i++)
  {
   Float1D(beta,i) = from->beta[i];
  }
  PyObject_SetAttrString(to,"beta",(PyObject*)beta);
  Py_DECREF(beta);

 // Extract phi - same as for beta...
  size[0] = from->behCount;
  PyArrayObject * phi = (PyArrayObject*)PyArray_SimpleNew(1,size,NPY_FLOAT);
  for (int i=0;i<from->behCount;i++)
  {
   Float1D(phi,i) = from->phi[i];
  }
  PyObject_SetAttrString(to,"phi",(PyObject*)phi);
  Py_DECREF(phi);

  
 // Update the topics information - replace current...
  size[0] = from->topics.Size();
  size[1] = from->wordCount;
  
  PyArrayObject * topicWord = (PyArrayObject*)PyArray_SimpleNew(2,size,NPY_INT);
  PyArrayObject * topicUse = (PyArrayObject*)PyArray_SimpleNew(1,size,NPY_INT);

  {
   ItemRef<Topic,Conc> * targ = from->topics.First();
   for (int t=0;t<topicWord->dimensions[0];t++)
   {
    targ->id = t;
    for (int w=0;w<topicWord->dimensions[1];w++)
    {
     Int2D(topicWord,t,w) = targ->wc[w];
    }
    Int1D(topicUse,t) = targ->RefCount();
    targ = targ->Next();
   }
  }

  PyObject_SetAttrString(to,"topicUse",(PyObject*)topicUse);
  PyObject_SetAttrString(to,"topicWord",(PyObject*)topicWord);
  Py_DECREF(topicUse);
  Py_DECREF(topicWord);
  
  PyObject * topicConc = PyFloat_FromDouble(from->topics.Body().conc);
  PyObject_SetAttrString(to,"topicConc",topicConc);
  Py_DECREF(topicConc);


 // Update the abnormal topic information - treat as an update...
  PyArrayObject * abnormTopicWord = (PyArrayObject*)PyObject_GetAttrString(to, "abnormTopicWord");
  {
   ItemRef<Topic,Conc> * topic = from->behTopics.First();
   while (topic->Valid())
   {
    for (int w=0;w<abnormTopicWord->dimensions[1];w++)
    {
     Int2D(abnormTopicWord,topic->beh,w) = topic->wc[w];
    }
    topic = topic->Next();
   }
  }
  Py_DECREF(abnormTopicWord);


 // Update the clusters information - replace current...
  size[0] = from->clusters.Size();
  
  PyObject * cluster = PyList_New(size[0]);
  PyArrayObject * clusterUse = (PyArrayObject*)PyArray_SimpleNew(1,size,NPY_INT);

  {
   ItemRef<Cluster,Conc> * clu = from->clusters.First();
   for (int c=0;c<from->clusters.Size();c++)
   {
    clu->id = c;
    PyObject * tup = PyTuple_New(4);
    PyList_SetItem(cluster,c,tup);

    size[0] = clu->Size();
    size[1] = 2;
    PyArrayObject * clusterInstance = (PyArrayObject*)PyArray_SimpleNew(2,size,NPY_INT);
    size[0] = from->behCount;
    PyArrayObject * behMultinomial = (PyArrayObject*)PyArray_SimpleNew(1,size,NPY_FLOAT);
    size[0] = from->flagSets->dimensions[0];
    PyArrayObject * behPriorMulti = (PyArrayObject*)PyArray_SimpleNew(1,size,NPY_INT);
    
    PyTuple_SetItem(tup, 0, (PyObject*)clusterInstance);
    PyTuple_SetItem(tup, 1, PyFloat_FromDouble(clu->Body().conc));
    PyTuple_SetItem(tup, 2, (PyObject*)behMultinomial);
    PyTuple_SetItem(tup, 3, (PyObject*)behPriorMulti);

    ItemRef<ClusterInst,Conc> * cluInst = clu->First();
    for (int ci=0;ci<clu->Size();ci++)
    {
     cluInst->id = ci;

     if (cluInst->GetTopic()) Int2D(clusterInstance,ci,0) = cluInst->GetTopic()->id;
     else Int2D(clusterInstance,ci,0) = -1;
     Int2D(clusterInstance,ci,1) = cluInst->RefCount();
     
     cluInst = cluInst->Next();
    }

    for (int b=0;b<from->behCount;b++)
    {
     Float1D(behMultinomial,b) = clu->GetBMN()[b];
    }

    if (clu->GetBehCountPrior()) // Easier to introduce it here - lets the rest of the code be null pointer safe as I made it that due to incrimental implimentation anyway.
    {
     for (int fs=0;fs<from->flagSets->dimensions[0];fs++)
     {
      Int1D(behPriorMulti,fs) = clu->GetBehCountPrior()[fs];
     }
    }
    else
    {
     for (int fs=0;fs<from->flagSets->dimensions[0];fs++)
     {
      Int1D(behPriorMulti,fs) = 0;
     }
    }

    Int1D(clusterUse,c) = clu->RefCount();

    clu = clu->Next();
   }
  }

  PyObject_SetAttrString(to,"clusterUse",(PyObject*)clusterUse);
  PyObject_SetAttrString(to,"cluster",cluster);
  Py_DECREF(clusterUse);
  Py_DECREF(cluster);

  PyObject * clusterConc = PyFloat_FromDouble(from->clusters.Body().conc);
  PyObject_SetAttrString(to,"clusterConc",clusterConc);
  Py_DECREF(clusterConc);


 // Update the documents information - keep it simple by just overwriting cluster and sample assignments whilst replacing the per-document DP...
  PyObject * docList = PyObject_GetAttrString(to,"doc");
  for (int d=0;d<from->docCount;d++)
  {
   Document & fromDoc = from->doc[d];
   PyObject * toDoc = PyList_GetItem(docList,d);

   // Set cluster...
    int clusterID = -1;
    if (fromDoc.GetCluster()) clusterID = fromDoc.GetCluster()->id;
    PyObject * cluID = PyInt_FromLong(clusterID);
    PyObject_SetAttrString(toDoc,"cluster",cluID);
    Py_DECREF(cluID);

   // Replace DP...
    size[0] = fromDoc.Size();
    size[1] = 3;
    PyArrayObject * use = (PyArrayObject*)PyArray_SimpleNew(2,size,NPY_INT);

    ItemRef<DocInst,Conc> * docInst = fromDoc.First();
    for (int di=0;di<size[0];di++)
    {
     docInst->id = di;

     if (docInst->GetClusterInst())
     {
      Int2D(use,di,0) = docInst->GetClusterInst()->GetTopic()->beh;
      Int2D(use,di,1) = docInst->GetClusterInst()->id;
     }
     else
     {
      Int2D(use,di,0) = -1;
      Int2D(use,di,1) = -1;
     }
     Int2D(use,di,2) = docInst->RefCount();
     
     docInst = docInst->Next();
    }

    PyObject_SetAttrString(toDoc,"use",(PyObject*)use);
    Py_DECREF(use);

    PyObject * conc = PyFloat_FromDouble(fromDoc.Body().conc);
    PyObject_SetAttrString(toDoc,"conc",conc);
    Py_DECREF(conc);

   // Update samples DP assignments...
    {
     PyArrayObject * samples = (PyArrayObject*)PyObject_GetAttrString(toDoc,"samples");
     for (int s=0;s<fromDoc.SampleCount();s++)
     {
      Sample & sam = fromDoc.GetSample(s);

      if (sam.GetDocInst()) Int2D(samples,s,0) = sam.GetDocInst()->id;
      else Int2D(samples,s,0) = -1;
     }
     Py_DECREF(samples);
    }

   // Update behaviour counts...
    {
     PyArrayObject * behCounts = (PyArrayObject*)PyObject_GetAttrString(toDoc,"behCounts");
     for (int b=0;b<from->behCount;b++)
     {
      Int1D(behCounts,b) = fromDoc.GetBehCounts()[b];
     }
     Py_DECREF(behCounts);
    }
  }
  Py_DECREF(docList);
}

"""



class TestDSLink(unittest.TestCase):
  """Test code for the data structure."""
  def test_compile(self):
    code = start_cpp(dual_hdp_ds_link) + """
    """
    weave.inline(code, support_code=dual_hdp_ds_link)



# If this file is run do the unit tests...
if __name__ == '__main__':
  unittest.main()
