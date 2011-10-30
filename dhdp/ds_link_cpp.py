# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



from scipy import weave
import unittest
from utils.start_cpp import start_cpp

from ds_cpp import ds_code



# Provides code for converting from python to the c++ data structure and back again - this is so the data can be stored in a suitable form in both situations, though it comes at the expense of a complex conversion...
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
   
   topic->IncRef(Int1D(topicUse,t));
  }

  Py_DECREF(topicUse);
  Py_DECREF(topicWord);

  PyObject * topicConc = PyObject_GetAttrString(from,"topicConc");
  to->topics.Body().alpha = gamma.alpha;
  to->topics.Body().beta = gamma.beta;
  to->topics.Body().conc = PyFloat_AsDouble(topicConc);
  Py_DECREF(topicConc);


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
    clu->Body().beta = to->rho.beta;
    clu->Body().conc = PyFloat_AsDouble(cluConc);
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
     
     int ciIndex = Int2D(use,di,0);
     int ciUse = Int2D(use,di,1);

     if (ciIndex!=-1) docInst->SetClusterInst(clusterInstArray[cluIndex][ciIndex],false);
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

   // Clean up...
    delete[] docInstArray;
  }

  Py_DECREF(docList);


 // Clean up...
  for (int c=0;c<clusterCount;c++) delete[] clusterInstArray[c];
  delete[] clusterInstArray;
  delete[] clusterArray; 
  delete[] topicArray;
}



// C++ -> Python - given pointers to the State class and a State object...
// Note that this assumes that the State object was created from the PyObject in the first place - if not it will almost certainly break.
void StateCppToPy(State * from, PyObject * to)
{
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


 // Update the clusters information - replace current...
  size[0] = from->clusters.Size();
  
  PyObject * cluster = PyList_New(size[0]);
  PyArrayObject * clusterUse = (PyArrayObject*)PyArray_SimpleNew(1,size,NPY_INT);

  {
   ItemRef<Cluster,Conc> * clu = from->clusters.First();
   for (int c=0;c<from->clusters.Size();c++)
   {
    clu->id = c;
    PyObject * pair = PyTuple_New(2);
    PyList_SetItem(cluster,c,pair);

    size[0] = clu->Size();
    size[1] = 2;
    PyArrayObject * clusterInstance = (PyArrayObject*)PyArray_SimpleNew(2,size,NPY_INT);
    PyTuple_SetItem(pair, 0, (PyObject*)clusterInstance);
    PyTuple_SetItem(pair, 1, PyFloat_FromDouble(clu->Body().conc));

    ItemRef<ClusterInst,Conc> * cluInst = clu->First();
    for (int ci=0;ci<clu->Size();ci++)
    {
     cluInst->id = ci;

     if (cluInst->GetTopic()) Int2D(clusterInstance,ci,0) = cluInst->GetTopic()->id;
     else Int2D(clusterInstance,ci,0) = -1;
     Int2D(clusterInstance,ci,1) = cluInst->RefCount();
     
     cluInst = cluInst->Next();
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
    size[1] = 2;
    PyArrayObject * use = (PyArrayObject*)PyArray_SimpleNew(2,size,NPY_INT);

    ItemRef<DocInst,Conc> * docInst = fromDoc.First();
    for (int di=0;di<size[0];di++)
    {
     docInst->id = di;

     if (docInst->GetClusterInst()) Int2D(use,di,0) = docInst->GetClusterInst()->id;
     else Int2D(use,di,0) = -1;
     Int2D(use,di,1) = docInst->RefCount();
     
     docInst = docInst->Next();
    }

    PyObject_SetAttrString(toDoc,"use",(PyObject*)use);
    Py_DECREF(use);

    PyObject * conc = PyFloat_FromDouble(fromDoc.Body().conc);
    PyObject_SetAttrString(toDoc,"conc",conc);
    Py_DECREF(conc);

   // Update samples DP assignments...
    PyArrayObject * samples = (PyArrayObject*)PyObject_GetAttrString(toDoc,"samples");
    
    for (int s=0;s<fromDoc.SampleCount();s++)
    {
     Sample & sam = fromDoc.GetSample(s);

     if (sam.GetDocInst()) Int2D(samples,s,0) = sam.GetDocInst()->id;
     else Int2D(samples,s,0) = -1;
    }
    
    Py_DECREF(samples);
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
