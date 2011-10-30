# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import unittest

from params import Params
from solve_shared import State
from model import DocModel
from utils.start_cpp import start_cpp

from ds_link_cpp import ds_link_code

from scipy import weave




# Shared code used to Gibbs sample the model - provides operations used repeatedly by the sampling code. Note that this contains all the heavy code used by the system - the rest is basically just loops. Additionally the data structure code is prepended to this, so this is the only shared code...
shared_code = ds_link_code + start_cpp() + """
// Code for resampling a documents cluster assignment...
void ResampleDocumentCluster(State & state, Document & doc)
{
 // If the document does not currently have a cluster then create one for it - let 'em cluster in non-initialisation iterations...
  if (doc.GetCluster()==0)
  {
   ItemRef<Cluster,Conc> * newC = state.clusters.Append();
   newC->Body().alpha = state.rho.alpha;
   newC->Body().beta  = state.rho.beta;
   newC->Body().conc  = state.rho.conc;
   doc.SetCluster(newC);
   return;
  }

 // Fill probAux of the topics with the counts of how many of each topic exist in the document whilst at the same time detaching the cluster instances from the document instances...
  {
   ItemRef<Topic,Conc> * topic = state.topics.First();
   while (topic->Valid())
   {
    topic->probAux = 0.0;
    topic = topic->Next();
   }
  }
  
  {
   ItemRef<DocInst,Conc> * docInst = doc.First();
   while (docInst->Valid())
   {
    docInst->topic = docInst->GetClusterInst()->GetTopic();
    docInst->topic->IncRef(); // Could be that this is the last (indirect) reference to the topic, and the next line could delete it - would be bad.
    docInst->SetClusterInst(0);
    
    docInst->topic->probAux += 1.0;
    
    docInst = docInst->Next();
   }
  }

 // Detach the document from its current cluster...
  doc.SetCluster(0);


 // Work out the log probabilities of assigning one of the known clusters to the document - store them in the cluster prob values. Uses the topic prob values as intermediates, for the probability of drawing each topic from the cluster...
 float maxLogProb = -1e100;
 {
  ItemRef<Cluster,Conc> * cluster = state.clusters.First();
  while (cluster->Valid())
  {
   // We need the probability of drawing each topic from the cluster, which we write into the prob variable of the topics...
    // Zero out the prob values of the topics...
    {
     ItemRef<Topic,Conc> * topic = state.topics.First();
     while (topic->Valid())
     {
      topic->prob = 0.0;
      topic = topic->Next();
     }
    }

    // Count how many times each topic has been drawn from the cluster, storing in the topic prob values...
    {
     ItemRef<ClusterInst,Conc> * cluInst = cluster->First();
     while (cluInst->Valid())
     {
      cluInst->GetTopic()->prob += cluInst->RefCount();
      cluInst = cluInst->Next();
     }
    }

    // Normalise whilst adding in the probability of drawing the given topic...
    // (There is some cleverness here to account for the extra references to the topics obtained from the document being resampled.)
    {
     ItemRef<Topic,Conc> * topic = state.topics.First();
     while (topic->Valid())
     {
      topic->prob += cluster->Body().conc * float(topic->RefCount()-topic->probAux) / (state.topics.RefTotal() - doc.Size() + state.topics.Body().conc);
      topic->prob /= cluster->RefTotal() + cluster->Body().conc;
      topic = topic->Next();
     }
    }

  // Now calculate the log probability of the cluster - involves a loop over the topics plus the inclusion of the probability of drawing this cluster...
   cluster->prob = log(cluster->RefCount());
   //cluster->prob -= log(state.clusters.RefTotal() + state.clusters.Body().conc);
   
   {
    ItemRef<Topic,Conc> * topic = state.topics.First();
    while (topic->Valid())
    {
     cluster->prob += topic->probAux * log(topic->prob);
     topic = topic->Next();
    }
   }

   if (cluster->prob>maxLogProb) maxLogProb = cluster->prob;

   cluster = cluster->Next();
  }
 }


 // Calculate the log probability of assigning a new cluster - involves quite a few terms, including a loop over the topics to get many of them...
  float probNew = log(state.clusters.Body().conc);
  //probNew -= log(state.clusters.RefTotal() + state.clusters.Body().conc);

  probNew += lnGamma(doc.Body().conc);
  probNew -= lnGamma(doc.Body().conc + doc.Size());

  {
   ItemRef<Topic,Conc> * topic = state.topics.First();
   while (topic->Valid())
   {
    float tProb = float(topic->RefCount()-topic->probAux) / (state.topics.RefTotal() - doc.Size() + state.topics.Body().conc);
    float tWeight = doc.Body().conc * tProb;

    probNew += lnGamma(tWeight + topic->probAux);
    probNew -= lnGamma(tWeight);

    topic = topic->Next();
   }
  }

  if (probNew>maxLogProb) maxLogProb = probNew;


 // Convert from logs to actual probabilities, with partial normalisation and summing for implicit precise normalisation later...
  float sumProb = 0.0;
  
  probNew = exp(probNew - maxLogProb);
  sumProb += probNew;
  
  {
   ItemRef<Cluster,Conc> * cluster = state.clusters.First();
   while (cluster->Valid())
   {
    cluster->prob = exp(cluster->prob - maxLogProb);
    sumProb += cluster->prob;
    cluster = cluster->Next();
   }
  }


 // Draw which cluster we are to assign; in the event of a new cluster create it...
  ItemRef<Cluster,Conc> * selected = 0;
  {
   float rand = sample_uniform() * sumProb;
   ItemRef<Cluster,Conc> * cluster = state.clusters.First();
   while (cluster->Valid())
   {
    rand -= cluster->prob;
    if (rand<0.0)
    {
     selected = cluster;
     break;
    }
    cluster = cluster->Next();
   }
  }

  if (selected==0)
  {
   selected = state.clusters.Append();
   selected->Body().alpha = state.rho.alpha;
   selected->Body().beta = state.rho.beta;
   selected->Body().conc = state.rho.conc;
  }
  

 // Update the document with its new cluster - consists of setting the documents cluster and updating the document instances to use the new cluster, which requires more sampling...
  doc.SetCluster(selected);

  ItemRef<DocInst,Conc> * docInst = doc.First();
  while(docInst->Valid())
  {
   // Update the cluster instance for this document instance - treat as a draw from the cluster DP with a hard requiremement that we draw an instance with the same topic as currently (What to do here is not given by the dual-hdp paper - this is just one option amung many, choosen for being good for convergance and relativly easy to impliment.)...
    // Sum weights from the cluster instances, but only when they are the correct topic; also add in the probability of creating a new cluster instance with the relevant topic...
     float probSum = selected->Body().conc * float(docInst->topic->RefCount()) / (state.topics.RefTotal() + state.topics.Body().conc);
     {
      ItemRef<ClusterInst,Conc> * targ2 = selected->First();
      while (targ2->Valid())
      {
       if (targ2->GetTopic()==docInst->topic) probSum += targ2->RefCount();
       targ2 = targ2->Next();
      }
     }

    // Select the relevant one...
     ItemRef<ClusterInst,Conc> * relevant = 0;
     {
      float rand = sample_uniform() * probSum;
      ItemRef<ClusterInst,Conc> * cluInst = selected->First();
      while (cluInst->Valid())
      {
       if (cluInst->GetTopic()==docInst->topic)
       {
        rand -= cluInst->RefCount();
        if (rand<0.0)
        {
         relevant = cluInst;
         break;
        }
       }
       cluInst = cluInst->Next();
      }
     }

     if (relevant==0)
     {
      relevant = selected->Append();
      relevant->SetTopic(docInst->topic);
     }

   // Assign it...
    docInst->SetClusterInst(relevant);

   // Temporary with topic in is no longer needed - decriment the reference...
    docInst->topic->DecRef();

   docInst = docInst->Next();
  } 
}



// Code for resampling the topics associated with cluster instances - single function that does them all - designed this way for efficiency reasons...
void ResampleClusterInstances(State & state)
{
 // First construct a linked list in each ClusterInst of all samples currently assigned to that ClusterInst, ready for the next bit - quite an involved process due to the multiple levels...
  {
   ItemRef<Cluster,Conc> * cluster = state.clusters.First();
   while (cluster->Valid())
   {
    ItemRef<ClusterInst,Conc> * cluInst = cluster->First();
    while (cluInst->Valid())
    {
     cluInst->first = 0;
     cluInst = cluInst->Next();
    }
    cluster = cluster->Next();
   }
  }
 
  for (int d=0;d<state.docCount;d++)
  {
   Document & doc = state.doc[d];
   for (int s=0;s<doc.SampleCount();s++)
   {
    Sample & sam = doc.GetSample(s);
    ItemRef<ClusterInst,Conc> * ci = sam.GetDocInst()->GetClusterInst();
    sam.next = ci->first;
    ci->first = &sam;
   }
  }


 // Now iterate all the cluster instances and resample each in turn...
  {
   ItemRef<Cluster,Conc> * cluster = state.clusters.First();
   while (cluster->Valid())
   {
    ItemRef<ClusterInst,Conc> * cluInst = cluster->First();
    while (cluInst->Valid())
    {
     // First decriment the topic word counts for all the using samples and remove its topic...
      int sampleCount = 0;
      {
       Sample * sam = cluInst->first;
       while (sam)
       {
        ItemRef<Topic,Conc> * topic = sam->GetDocInst()->GetClusterInst()->GetTopic();
        topic->wc[sam->GetWord()] -= 1;
        topic->wcTotal -= 1;
        sampleCount += 1;
        
        sam = sam->next;
       }
      }
      cluInst->SetTopic(0);


     // Iterate the topics and calculate the log probability of each, find maximum log probability...
      float maxLogProb = -1e100;
      {
       ItemRef<Topic,Conc> * topic = state.topics.First();
       while (topic->Valid())
       {
        topic->prob = log(topic->RefCount());
        float samDiv = log(topic->wcTotal + state.betaSum);
        Sample * sam = cluInst->first;
        while (sam)
        {
         topic->prob += log(topic->wc[sam->GetWord()] + state.beta[sam->GetWord()]) - samDiv;
         sam = sam->next;
        }

        if (topic->prob>maxLogProb) maxLogProb = topic->prob;
        topic = topic->Next();
       }
      }

     // Calculate the log probability of a new topic; maintain maximum...
      float probNew = log(state.topics.Body().conc);
      {
       Sample * sam = cluInst->first;
       while (sam)
       {
        probNew += log(state.beta[sam->GetWord()]/state.betaSum);
        sam = sam->next;
       }
      }
      
      if (probNew>maxLogProb) maxLogProb = probNew;


     // Convert log probabilities to actual probabilities in a numerically safe way, and sum them up for selection...
      float probSum = 0.0;

      probNew = exp(probNew-maxLogProb);
      probSum += probNew;

      {
       ItemRef<Topic,Conc> * topic = state.topics.First();
       while (topic->Valid())
       {
        topic->prob = exp(topic->prob-maxLogProb);
        probSum += topic->prob;
      
        topic = topic->Next();
       }
      }


     // Select the resampled topic, creating a new one if required...
      ItemRef<Topic,Conc> * nt = 0;
      float rand = probSum * sample_uniform();

      {
       ItemRef<Topic,Conc> * topic = state.topics.First();
       while (topic->Valid())
       {
        rand -= topic->prob;
        if (rand<0.0)
        {
         nt = topic;
         break;
        }
        topic = topic->Next();
       }
      }

      if (nt==0)
      {
       nt = state.topics.Append();
       nt->wc = new int[state.wordCount];
       for (int w=0;w<state.wordCount;w++) nt->wc[w] = 0;
       nt->wcTotal = 0;
      }


     // Finally set its topic and sum back in the topic usage by its using samples...
      cluInst->SetTopic(nt);
      {
       Sample * sam = cluInst->first;
       while (sam)
       {
        ItemRef<Topic,Conc> * topic = sam->GetDocInst()->GetClusterInst()->GetTopic();
        topic->wc[sam->GetWord()] += 1;
        topic->wcTotal += 1;

        sam = sam->next;
       }
      }

     cluInst = cluInst->Next();
    }
    cluster = cluster->Next();
   }
  }

}



// Code for resampling a document instance's cluster instance - actually does all document instances for a single document with each call, for efficiency reasons...
void ResampleDocumentInstances(State & state, Document & doc)
{
 // First construct a linked list in each DocInst of the samples contained within - needed to do the next task efficiently...
 {
  ItemRef<DocInst,Conc> * docInst = doc.First();
  while (docInst->Valid())
  {
   docInst->first = 0;
   docInst = docInst->Next();
  }
 }

 for (int s=0;s<doc.SampleCount();s++)
 {
  Sample & sam = doc.GetSample(s);
  sam.next = sam.GetDocInst()->first;
  sam.GetDocInst()->first = &sam;
 }


 // Now iterate all DocInst in the document, resampling each in turn...
 {
  ItemRef<DocInst,Conc> * docInst = doc.First();
  while (docInst->Valid())
  {
   // Detach from its cluster instance, removing all topic references at the same time...
    int sampleCount = 0;
    {
     Sample * sample = docInst->first;
     while (sample)
     {
      ItemRef<Topic,Conc> * topic = sample->GetDocInst()->GetClusterInst()->GetTopic();
      topic->wc[sample->GetWord()] -= 1;
      topic->wcTotal -= 1;
      sampleCount += 1;
      sample = sample->next;
     }
    }
    docInst->SetClusterInst(0);


   // Iterate the topics and determine the log probability of each topic for the sample in probAux and the log probability of drawing a new cluster with the given topic in prob. The latter has its max recorded for numerically stable normalisation later...
    float maxLogProb = -1e100;
    {
     ItemRef<Topic,Conc> * topic = state.topics.First();
     float baseTopicLogProb = log(doc.GetCluster()->Body().conc) - log(state.topics.RefTotal() + state.topics.Body().conc);
     while (topic->Valid())
     {
      topic->probAux = 0.0;
      Sample * sample = docInst->first;
      float samDiv = log(topic->wcTotal + state.betaSum);
      while (sample)
      {
       topic->probAux += log(topic->wc[sample->GetWord()] + state.beta[sample->GetWord()]) - samDiv;
       sample = sample->next;
      }
      
      topic->prob = baseTopicLogProb + log(topic->RefCount()) + topic->probAux;
      if (topic->prob>maxLogProb) maxLogProb = topic->prob;

      topic = topic->Next();
     }
    }

   // Iterate the cluster instances and calculate their log probabilities, maintaining knowledge of the maximum...
    {
     ItemRef<ClusterInst,Conc> * cluInst = doc.GetCluster()->First();
     while (cluInst->Valid())
     {
      cluInst->prob = log(cluInst->RefCount()) + cluInst->GetTopic()->probAux;
      if (cluInst->prob>maxLogProb) maxLogProb = cluInst->prob;
      
      cluInst = cluInst->Next();
     }
    }

   // Calculate the log probability of a new topic and new cluster instance, factor into the maximum...
    float probAllNew = log(doc.GetCluster()->Body().conc) + log(state.topics.Body().conc) - log(state.topics.RefTotal() + state.topics.Body().conc);
    {
     Sample * sample = docInst->first;
     while (sample)
     {
      probAllNew += log(state.beta[sample->GetWord()]/state.betaSum);
      sample = sample->next;
     }
    }
    if (probAllNew>maxLogProb) maxLogProb = probAllNew;


   // Use the maximum log probability to convert all values to normal probabilities in a numerically safe way, storing a sum ready for drawing from the various options...
    float probSum = 0.0;
    
    probAllNew = exp(probAllNew-maxLogProb);
    probSum += probAllNew;

    {
     ItemRef<Topic,Conc> * topic = state.topics.First();
     while (topic->Valid())
     {
      topic->prob = exp(topic->prob-maxLogProb);
      probSum += topic->prob;
      topic = topic->Next();
     }
    }

    {
     ItemRef<ClusterInst,Conc> * cluInst = doc.GetCluster()->First();
     while (cluInst->Valid())
     {
      cluInst->prob = exp(cluInst->prob-maxLogProb);
      probSum += cluInst->prob;
      cluInst = cluInst->Next();
     }
    }


   // Draw the new cluster instance - can involve creating a new one and even creating a new topic...
    ItemRef<ClusterInst,Conc> * nci = 0;
    float rand = sample_uniform() * probSum;

    {
     ItemRef<ClusterInst,Conc> * cluInst = doc.GetCluster()->First();
     while (cluInst->Valid())
     {
      rand -= cluInst->prob;
      if (rand<0.0)
      {
       nci = cluInst;
       break;
      }
      cluInst = cluInst->Next();
     }
    }

    if (nci==0)
    {
     nci = doc.GetCluster()->Append();

     ItemRef<Topic,Conc> * topic = state.topics.First();
     while (topic->Valid())
     {
      rand -= topic->prob;
      if (rand<0.0)
      {
       nci->SetTopic(topic);
       break;
      }
      topic = topic->Next();
     }
    }

    if (nci->GetTopic()==0)
    {
     ItemRef<Topic,Conc> * nt = state.topics.Append();
     nt->wc = new int[state.wordCount];
     for (int w=0;w<state.wordCount;w++) nt->wc[w] = 0;
     nt->wcTotal = 0;

     nci->SetTopic(nt);
    }


   // Reattach its new cluster instance, and incriment the topic word counts...
    docInst->SetClusterInst(nci);
    {
     Sample * sample = docInst->first;
     while (sample)
     {
      ItemRef<Topic,Conc> * topic = sample->GetDocInst()->GetClusterInst()->GetTopic();
      topic->wc[sample->GetWord()] += 1;
      topic->wcTotal += 1;
      sample = sample->next;
     }
    }
   
   docInst = docInst->Next();
  }
 }
}



// Code for resampling a samples topic instance assignment...
// (Everything must be assigned - no null pointers on the chain from sample to topic.)
void ResampleSample(State & state, Document & doc, Sample & sam)
{
 // Remove the samples current assignment...
  sam.SetDocInst(0);

 // Assign probabilities to the various possibilities - there are temporary variables in the data structure to make this elegant. Sum up the total probability ready for the sampling phase. In all cases an entity is assigned the probability of using that entity with everything below it being created from scratch...
  float pSum = 0.0;
  // Calculate the probabilities of various 'new' events...
   float probNewDocInst = doc.Body().conc / (doc.RefTotal() + doc.Body().conc);
   float probNewCluInst = probNewDocInst * doc.GetCluster()->Body().conc / (doc.GetCluster()->RefTotal() + doc.GetCluster()->Body().conc);
   float probNewTopic = probNewCluInst * state.topics.Body().conc / (state.topics.RefTotal() + state.topics.Body().conc);
   
  // The probability of a new topic...
   pSum += probNewTopic * state.beta[sam.GetWord()] / state.betaSum;
   
  // The topics - keep the probabilities of drawing the word in question from the topic in the aux variables, to save computation in the following steps...
  {
   ItemRef<Topic,Conc> * topic = state.topics.First();
   float divisor = state.topics.RefTotal() + state.topics.Body().conc;
   while (topic->Valid())
   {
    topic->probAux = (topic->wc[sam.GetWord()] + state.beta[sam.GetWord()]) / (topic->wcTotal + state.betaSum);
    topic->prob = topic->probAux * probNewCluInst * topic->RefCount() / divisor;
    pSum += topic->prob;

    topic = topic->Next();
   }
  }

  // The cluster instances...
  {
   ItemRef<ClusterInst,Conc> * cluInst = doc.GetCluster()->First();
   float divisor = doc.GetCluster()->RefTotal() + doc.GetCluster()->Body().conc;
   while (cluInst->Valid())
   {
    cluInst->prob = cluInst->GetTopic()->probAux * probNewDocInst * cluInst->RefCount() / divisor;
    pSum += cluInst->prob;
   
    cluInst = cluInst->Next();
   }
  }

  // The document instances...
  {
   ItemRef<DocInst,Conc> * docInst = doc.First();
   float divisor = doc.RefTotal() + doc.Body().conc;
   while (docInst->Valid())
   {
    docInst->prob = docInst->GetClusterInst()->GetTopic()->probAux * docInst->RefCount() / divisor;
    pSum += docInst->prob;
    
    docInst = docInst->Next();
   }
  }

 // Now draw from the distribution and assign the result, creating new entities as required. The checking is done in order of (typically) largest to smallest, to maximise the chance of an early bail out...
  // Draw the random uniform, scaled by the pSum - we will repeatedly subtract from this random variable for each item - when it becomes negative we have found the item to draw...
   float rand = sample_uniform() * pSum;
  
  // Check the document instances...
   {
    ItemRef<DocInst,Conc> * docInst = doc.First();
    while (docInst->Valid())
    {
     rand -= docInst->prob;
     if (rand<0.0)
     {
      // A document instance has been selected - simplest reassignment case...
       sam.SetDocInst(docInst);

      return;
     }

     docInst = docInst->Next();
    }
   }

  // Check the cluster instances - would involve a new document instance...
  {
   ItemRef<ClusterInst,Conc> * cluInst = doc.GetCluster()->First();
   while (cluInst->Valid())
   {
    rand -= cluInst->prob;
    if (rand<0.0)
    {
     // A cluster instance has been selected - need to create a new document instance...
      ItemRef<DocInst,Conc> * ndi = doc.Append();
      ndi->SetClusterInst(cluInst);

      sam.SetDocInst(ndi);
      
     return;
    }

    cluInst = cluInst->Next();
   }
  }

  // Check the topics - would involve both a new cluster and document instance...
  {
   ItemRef<Topic,Conc> * topic = state.topics.First();
   while (topic->Valid())
   {
    rand -= topic->prob;
    if (rand<0.0)
    {
     // A topic has been selected - need a new cluster and a new document instance...
      ItemRef<ClusterInst,Conc> * nci = doc.GetCluster()->Append();
      nci->SetTopic(topic);
     
      ItemRef<DocInst,Conc> * ndi = doc.Append();
      ndi->SetClusterInst(nci);

      sam.SetDocInst(ndi);

     return;
    }

    topic = topic->Next();
   }
  }

  // If we have got this far then its a new topic, with a new cluster and document instance as well...
   ItemRef<Topic,Conc> * nt = state.topics.Append();
   nt->wc = new int[state.wordCount];
   for (int w=0;w<state.wordCount;w++) nt->wc[w] = 0;
   nt->wcTotal = 0;

   ItemRef<ClusterInst,Conc> * nci = doc.GetCluster()->Append();
   nci->SetTopic(nt);

   ItemRef<DocInst,Conc> * ndi = doc.Append();
   ndi->SetClusterInst(nci);

   sam.SetDocInst(ndi);
}



// Code for resampling all the concentration parameters - just have to iterate through and call all the resampling methods...
void ResampleConcs(State & state)
{
 // Concentrations for DPs from which topics and clusters are drawn...
  state.topics.Body().ResampleConc(state.topics.RefTotal(), state.topics.Size());
  state.clusters.Body().ResampleConc(state.clusters.RefTotal(), state.clusters.Size());

 // Concentrations for clusters...
  if (state.seperateClusterConc)
  {
   ItemRef<Cluster,Conc> * cluster = state.clusters.First();
   while (cluster->Valid())
   {
    cluster->Body().ResampleConc(cluster->RefTotal(), cluster->Size());
    cluster = cluster->Next();
   }
  }
  else
  {
   if (state.clusters.Size()>0)
   {
    SampleConcDP scdp;
    scdp.SetPrior(state.rho.alpha,state.rho.beta);
    scdp.SetPrevConc(state.clusters.First()->Body().conc);

    ItemRef<Cluster,Conc> * cluster = state.clusters.First();
    while (cluster->Valid())
    {
     scdp.AddDP(cluster->RefTotal(), cluster->Size());
     cluster = cluster->Next();
    }

    double newConc = scdp.Sample();

    cluster = state.clusters.First();
    while (cluster->Valid())
    {
     cluster->Body().conc = newConc;
     cluster = cluster->Next();
    }
   }
  }

 // Concentrations for documents...
  if (state.seperateDocumentConc)
  {
   for (int d=0;d<state.docCount;d++)
   {
    state.doc[d].Body().ResampleConc(state.doc[d].RefTotal(), state.doc[d].Size());
   }
  }
  else
  {
   SampleConcDP scdp;
   scdp.SetPrior(state.doc[0].Body().alpha,state.doc[0].Body().beta);
   scdp.SetPrevConc(state.doc[0].Body().conc);

   for (int d=0;d<state.docCount;d++)
   {
    scdp.AddDP(state.doc[d].RefTotal(), state.doc[d].Size());
   }

   double newConc = scdp.Sample();

   for (int d=0;d<state.docCount;d++)
   {
    state.doc[d].Body().conc = newConc;
   }
  }
}

"""



# The actual function for Gibbs iterating the data structure - takes as input the State object as 'state' and the number of iterations to do as 'iters'...
gibbs_code = start_cpp(shared_code) + """
State s;
StatePyToCpp(state, &s);
float * mn = new float[s.wordCount];

for (int iter=0;iter<iters;iter++)
{
 // Iterate the documents...
  for (int d=0;d<s.docCount;d++)
  {
   // Resample the documents cluster...
    if (s.oneCluster)
    {
     if (s.doc[d].GetCluster()==0)
     {
      if (s.clusters.Size()==0)
      {
       ItemRef<Cluster,Conc> * newC = s.clusters.Append();
       newC->Body().alpha = s.rho.alpha;
       newC->Body().beta = s.rho.beta;
       newC->Body().conc = s.rho.conc;
       s.doc[d].SetCluster(newC);
      }
      else
      {
       s.doc[d].SetCluster(s.clusters.First());
      }
     }
    }
    else
    {
     ResampleDocumentCluster(s, s.doc[d]);
    }

   // Resample the documents samples (words)...
    for (int w=0;w<s.doc[d].SampleCount();w++)
    {
     ResampleSample(s, s.doc[d], s.doc[d].GetSample(w));
    }

   // Resample the cluster instance that each document instance is assigned to...
    if (!s.dnrDocInsts)
    {
     ResampleDocumentInstances(s,s.doc[d]);
    }
  }

 // Resample the cluster instances assigned topics...
  if (!s.dnrCluInsts)
  {
   ResampleClusterInstances(s);
  }

 // Resample the many concentration parameters...
  ResampleConcs(s);

 // If requested recalculate beta...
  if (s.calcBeta)
  {
   EstimateDir ed(s.wordCount);
   
   ItemRef<Topic,Conc> * topic = s.topics.First();
   while (topic->Valid())
   {
    float div = 0.0;
    for (int i=0;i<s.wordCount;i++)
    {
     mn[i] = topic->wc[i] + s.beta[i];
     div += mn[i];
    }
    for (int i=0;i<s.wordCount;i++) mn[i] /= div;
   
    ed.Add(mn); // Not actually correct - we are using the mean of the distribution from which we should draw the multinomial, rather than actually drawing. This is easier however, and not that unreasonable.
    topic = topic->Next();
   }

   ed.Update(s.beta);
   s.betaSum = 0.0;
   for (int i=0;i<s.wordCount;i++) s.betaSum += s.beta[i];
  }

 // Verify the state is consistant - for debugging (Only works when there is no prior)...
  //VerifyState(s);
}

delete[] mn;
StateCppToPy(&s, state);

"""



class ProgReporter:
  """Class to allow progress to be reported."""
  def __init__(self,params,callback,mult = 1):
    self.doneIters = 0
    self.totalIters = mult * params.runs * (max((params.burnIn,params.lag)) + params.samples + (params.samples-1)*params.lag)
    self.callback = callback

    if self.callback:
      self.callback(self.doneIters,self.totalIters)

  def next(self, amount = 1):
    self.doneIters += amount
    if self.callback:
      self.callback(self.doneIters,self.totalIters)



def gibbs(state, total_iters, next, step = 64):
  """Does the requested number of Gibbs iterations to the passed in state. If state has not been initialised the first iteration will be an incrimental construction."""
  while total_iters>0:
    iters = total_iters
    if iters>step: iters = step
    total_iters -= iters
    
    weave.inline(gibbs_code, ['state', 'iters'], support_code=shared_code)
    
    next(iters)


def gibbs_run(state, next):
  """Does a single run on the given state object, adding the relevant samples."""
  params = state.getParams()
  if params.burnIn>params.lag: gibbs(state, params.burnIn-params.lag,next)

  for s in xrange(params.samples):
    gibbs(state, params.lag,next)
    state.sample()
    next()


def gibbs_all(state, callback = None):
  """Does all the runs requested by a states params object, collating all the samples into the State."""
  params = state.getParams()
  reporter = ProgReporter(params,callback)
  
  for r in xrange(params.runs):
    tempState = State(state)
    gibbs_run(tempState,reporter.next)
    state.absorbClone(tempState)



def gibbs_doc(model, doc, params = None, callback = None):
  """Runs Gibbs iterations on a single document, by sampling with a prior constructed from each sample in the given Model. params applies to each sample, so should probably be much more limited than usual - the default if its undefined is to use 1 run and 1 sample and a burn in of only 500. Returns a DocModel with all the relevant samples in."""
  
  # Initialisation stuff - handle params, create the state and the DocModel object, plus a reporter...
  if params==None:
    params = Params()
    params.runs = 1
    params.samples = 1
    params.burnIn = 500

  state = State(doc, params)
  dm = DocModel()
  reporter = ProgReporter(params,callback,model.sampleCount())

  # Iterate and run for each sample in the model...
  for sample in model.sampleList():
    tempState = State(state)
    tempState.setGlobalParams(sample)
    tempState.addPrior(sample)
    gibbs_run(tempState,reporter.next)
    dm.addFrom(tempState.getModel())

  # Return...
  return dm



class TestShared(unittest.TestCase):
  """Test code for the data structure."""
  def test_compile(self):
    code = start_cpp(shared_code) + """
    """
    weave.inline(code, support_code=shared_code)



# If this file is run do the unit tests...
if __name__ == '__main__':
  unittest.main()
