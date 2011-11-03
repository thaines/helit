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



import unittest
import math

from params import Params
from solve_shared import State
from model import DocModel
from utils.start_cpp import start_cpp

from ds_link_cpp import ds_link_code

from scipy import weave




# Shared code used to Gibbs sample the model - provides operations used repeatedly by the sampling code. Note that this contains all the heavy code used by the system - the rest is basically just loops. Additionally the data structure code is prepended to this, so this is the only shared code...
shared_code = ds_link_code + start_cpp() + """

#include <sys/time.h>

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
   float * bmn = new float[state.behCount];
   float bmnDiv = 0.0;
   for (int b=0;b<state.behCount;b++)
   {
    bmn[b] = state.phi[b];
    bmnDiv += state.phi[b];
   }
   for (int b=0;b<state.behCount;b++) bmn[b] /= bmnDiv;
   newC->SetBMN(bmn);
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

  int normalDocInst = 0;
  {
   ItemRef<DocInst,Conc> * docInst = doc.First();
   while (docInst->Valid())
   {
    docInst->topic = docInst->GetClusterInst()->GetTopic();
    if (docInst->topic->beh==0) // Only need to redo the normal ones, as that is all resampling the cluster affects.
    {
     docInst->topic->IncRef(); // Could be that this is the last (indirect) reference to the topic, and the next line could delete it - would be bad.
     docInst->SetClusterInst(0);
    
     docInst->topic->probAux += 1.0;
     normalDocInst += 1;
    }
    
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
      topic->prob += cluster->Body().conc * float(topic->RefCount()-topic->probAux) / (state.topics.RefTotal() - normalDocInst + state.topics.Body().conc);
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

   // Factor in the probability of the clusters distribution over behaviours...
    float bmnDiv = cluster->GetBMN()[0];
    bool hasAbnorm = false;
    for (int b=1;b<state.behCount;b++)
    {
     if (doc.GetBehFlags()[b]!=0)
     {
      bmnDiv += cluster->GetBMN()[b];
      hasAbnorm = true;
     }
    }
    
    if (hasAbnorm)
    {
     cluster->prob += lnGamma(doc.SampleCount()+1.0);
     for (int b=0;b<state.behCount;b++)
     {
      if (doc.GetBehFlags()[b]!=0)
      {
       cluster->prob += doc.GetBehCounts()[b] * log(cluster->GetBMN()[b]/bmnDiv);
       cluster->prob -= lnGamma(doc.GetBehCounts()[b]+1.0);
      }
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
    float tProb = float(topic->RefCount()-topic->probAux) / (state.topics.RefTotal() - normalDocInst + state.topics.Body().conc);
    float tWeight = doc.Body().conc * tProb;

    probNew += lnGamma(tWeight + topic->probAux);
    probNew -= lnGamma(tWeight);

    topic = topic->Next();
   }
  }

  {
   float phiDiv = state.phi[0];
   bool hasAbnorm = false;
   for (int b=1;b<state.behCount;b++)
   {
    if (doc.GetBehFlags()[b]!=0)
    {
     phiDiv += state.phi[b];
     hasAbnorm = true;
    }
   }

   if (hasAbnorm)
   {
    probNew += lnGamma(doc.SampleCount()+1.0);
    for (int b=0;b<state.behCount;b++)
    {
     if (doc.GetBehFlags()[b]!=0)
     {
      probNew += doc.GetBehCounts()[b] * log(state.phi[b]/phiDiv);
      probNew -= lnGamma(doc.GetBehCounts()[b]+1.0);
     }
    }
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
   float * bmn = new float[state.behCount];
   float bmnDiv = 0.0;
   for (int b=0;b<state.behCount;b++)
   {
    bmn[b] = state.phi[b];
    bmnDiv += state.phi[b];
   }
   for (int b=0;b<state.behCount;b++) bmn[b] /= bmnDiv;
   selected->SetBMN(bmn);
  }
  

 // Update the document with its new cluster - consists of setting the documents cluster and updating the document instances to use the new cluster, which requires more sampling...
  doc.SetCluster(selected);

  ItemRef<DocInst,Conc> * docInst = doc.First();
  while(docInst->Valid())
  {
   if (docInst->topic->beh==0)
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
   }

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
    sam.next = ci->first; // Note that doing this for abnormal cluster instances makes no sense, but causes no harm either, hence leaving it with the simpler code.
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
      {
       Sample * sam = cluInst->first;
       while (sam)
       {
        ItemRef<Topic,Conc> * topic = sam->GetDocInst()->GetClusterInst()->GetTopic();
        topic->wc[sam->GetWord()] -= 1;
        topic->wcTotal -= 1;
        
        sam = sam->next;
       }
      }
      cluInst->SetTopic(0);

     // Count the number of each word type used by all the children of the cluster instance...
      {
       for (int w=0;w<state.wordCount;w++) state.tempWord[w] = 0;
       Sample * sam = cluInst->first;
       while (sam)
       {
        state.tempWord[sam->GetWord()] += 1;
        sam = sam->next;
       }
      }


     // Iterate the topics and calculate the log probability of each, find maximum log probability...
      float maxLogProb = -1e100;
      {
       ItemRef<Topic,Conc> * topic = state.topics.First();
       while (topic->Valid())
       {
        topic->prob = log(topic->RefCount());
        float samDiv = log(topic->wcTotal + state.betaSum);
        for (int w=0;w<state.wordCount;w++)
        {
         if (state.tempWord[w]!=0)
         {
          topic->prob += state.tempWord[w]*(log(topic->wc[w] + state.beta[w]) - samDiv);
         }
        }

        if (topic->prob>maxLogProb) maxLogProb = topic->prob;
        topic = topic->Next();
       }
      }

     // Calculate the log probability of a new topic; maintain maximum...
      float probNew = log(state.topics.Body().conc);
      {
       for (int w=0;w<state.wordCount;w++)
       {
        if (state.tempWord[w]!=0)
        {
         probNew += state.tempWord[w]*log(state.beta[w]/state.betaSum);
        }
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
       nt->beh = 0;
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
 // Calculate the normaliser for the behaviour multinomial and the log probability of normal behaviour...
  float bmnNorm = 0.0;
  for (int b=0;b<state.behCount;b++)
  {
   if (doc.GetBehFlags()[b]!=0) bmnNorm += doc.GetCluster()->GetBMN()[b];
  }
  float logProbNorm = log(doc.GetCluster()->GetBMN()[0]/bmnNorm);
 
 // Construct a linked list in each DocInst of the samples contained within - needed to do the next task efficiently...
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
   // Detach from its cluster instance, removing all topic references at the same time, also, count how many words it has...
    {
     for (int w=0;w<state.wordCount;w++) state.tempWord[w] = 0;
     Sample * sample = docInst->first;
     while (sample)
     {
      ItemRef<Topic,Conc> * topic = sample->GetDocInst()->GetClusterInst()->GetTopic();
      topic->wc[sample->GetWord()] -= 1;
      state.tempWord[sample->GetWord()] += 1;
      topic->wcTotal -= 1;
      doc.GetBehCounts()[topic->beh] -= 1;
      sample = sample->next;
     }
    }
    docInst->SetClusterInst(0);

   
   // Iterate the topics and determine the log probability of each topic for the samples in probAux and the log probability of drawing a new cluster instance with the given topic in prob. The latter has its max recorded for numerically stable normalisation later...
    float maxLogProb = -1e100;
    float logTopicNorm = log(state.topics.RefTotal() + state.topics.Body().conc);
    float logCluNorm = log(doc.GetCluster()->RefTotal() + doc.GetCluster()->Body().conc);
    {
     float baseTopicLogProb = logProbNorm + log(doc.GetCluster()->Body().conc) - logCluNorm - logTopicNorm;
     ItemRef<Topic,Conc> * topic = state.topics.First();
     while (topic->Valid())
     {
      topic->probAux = 0.0;
      
      float samDiv = log(topic->wcTotal + state.betaSum);
      for (int w=0;w<state.wordCount;w++)
      {
       if (state.tempWord[w]!=0)
       {
        topic->probAux += state.tempWord[w]*(logf(topic->wc[w] + state.beta[w]) - samDiv); // Don't normalise for arbitrary order, as same constant for all.
       }
      }
      
      topic->prob = baseTopicLogProb + logf(topic->RefCount()) + topic->probAux;
      if (topic->prob>maxLogProb) maxLogProb = topic->prob;

      topic = topic->Next();
     }
    }

   // Iterate the cluster instances and calculate their log probabilities, maintaining knowledge of the maximum...
    {
     ItemRef<ClusterInst,Conc> * cluInst = doc.GetCluster()->First();
     while (cluInst->Valid())
     {
      cluInst->prob = logProbNorm + logf(cluInst->RefCount()) - logCluNorm + cluInst->GetTopic()->probAux;
      if (cluInst->prob>maxLogProb) maxLogProb = cluInst->prob;
      
      cluInst = cluInst->Next();
     }
    }

   // Calculate the log probability of a new topic and new cluster instance, factor into the maximum...
    float probAllNew = logProbNorm + log(doc.GetCluster()->Body().conc) - logCluNorm + log(state.topics.Body().conc) - logTopicNorm;
    {
     for (int w=0;w<state.wordCount;w++)
     {
      if (state.tempWord[w]!=0)
      {
       probAllNew += state.tempWord[w]*logf(state.beta[w]/state.betaSum); // Ignore ordering irrelevance normalisation, as done throughout due to being constant.
      }
     }
    }
    if (probAllNew>maxLogProb) maxLogProb = probAllNew;
    
   // Do all the abnormal topics - same idea as previously...
    {
     ItemRef<Topic,Conc> * topic = state.behTopics.First()->Next();
     while (topic->Valid())
     {
      if (doc.GetBehFlags()[topic->beh]!=0)
      {
       topic->prob = log(doc.GetCluster()->GetBMN()[topic->beh]/bmnNorm);

       float samDiv = log(topic->wcTotal + state.betaSum);
       for (int w=0;w<state.wordCount;w++)
       {
        if (state.tempWord[w]!=0)
        {
         topic->prob += state.tempWord[w]*(logf(topic->wc[w] + state.beta[w]) - samDiv); // Don't normalise for arbitrary ordering, as same constant for all.
        }
       }
       
       if (topic->prob>maxLogProb) maxLogProb = topic->prob;
      }
      
      topic = topic->Next();
     }
    }


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
    
    {
     ItemRef<Topic,Conc> * topic = state.behTopics.First()->Next();
     while (topic->Valid())
     {
      if (doc.GetBehFlags()[topic->beh]!=0)
      {
       topic->prob = exp(topic->prob-maxLogProb);
       probSum += topic->prob;
      }
      topic = topic->Next();
     }
    }


   // Draw the new cluster instance - can involve creating a new one and even creating a new topic...
    ItemRef<ClusterInst,Conc> * nci = 0;
    float rand = sample_uniform() * probSum;

    // Is it a normal cluster instance that already exists?..
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
    
    // Is it an abnormal topic?..
     if (nci==0)
     {
      ItemRef<ClusterInst,Conc> * cluInst = state.behCluInsts.First()->Next();
      while (cluInst->Valid())
      {
       if (doc.GetBehFlags()[cluInst->GetTopic()->beh]!=0)
       {
        rand -= cluInst->GetTopic()->prob;
        if (rand<0.0)
        {
         nci = cluInst;
         break;
        }
       }
       cluInst = cluInst->Next();
      }
     }
    
    // Is it a new cluster instance?..
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
    
    // Is it a new topic as well as a new cluster instance?..
     if (nci->GetTopic()==0)
     {
      ItemRef<Topic,Conc> * nt = state.topics.Append();
      nt->wc = new int[state.wordCount];
      for (int w=0;w<state.wordCount;w++) nt->wc[w] = 0;
      nt->wcTotal = 0;
      nt->beh = 0;

      nci->SetTopic(nt);
     }


   // Reattach its resampled cluster instance, and incriment the topic word counts...
    docInst->SetClusterInst(nci);
    {
     Sample * sample = docInst->first;
     while (sample)
     {
      ItemRef<Topic,Conc> * topic = sample->GetDocInst()->GetClusterInst()->GetTopic();
      topic->wc[sample->GetWord()] += 1;
      topic->wcTotal += 1;
      doc.GetBehCounts()[topic->beh] += 1;
      sample = sample->next;
     }
    }
   
   docInst = docInst->Next();
  }
 }
}



// Helper for below, seperated out as required seperate for the left to right algorithm later on. Returns the sum of all the probabilities of the options for the sample just calculated, and leaves correct values in all the relevant ->prob variables...
float CalcSampleProb(State & state, Document & doc, Sample & sam)
{
 float pSum = 0.0;
 
 // Calculate the normalising constant for the associated clusters behaviour multinomial given the documents behaviour flags...
  float bmvDiv = 0.0;
  for (int b=0;b<state.behCount;b++)
  {
   if (doc.GetBehFlags()[b]!=0) bmvDiv += doc.GetCluster()->GetBMN()[b];
  }

 // Probability of going for something normal...
  float probNormal = doc.GetCluster()->GetBMN()[0] / bmvDiv;

 // Calculate the probabilities of various 'new' events...
  float probNewDocInst = doc.Body().conc / (doc.RefTotal() + doc.Body().conc);
  float probNewCluInst = probNewDocInst * doc.GetCluster()->Body().conc / (doc.GetCluster()->RefTotal() + doc.GetCluster()->Body().conc);
  float probNewTopic = probNewCluInst * state.topics.Body().conc / (state.topics.RefTotal() + state.topics.Body().conc);

 // The probability of a new topic...
  pSum += probNormal * probNewTopic * state.beta[sam.GetWord()] / state.betaSum;

 // The topics - keep the probabilities of drawing the word in question from the topic in the aux variables, to save computation in the following steps...
  float betaWeight = state.beta[sam.GetWord()];
  {
   ItemRef<Topic,Conc> * topic = state.topics.First();
   float base = probNormal * probNewCluInst / (state.topics.RefTotal() + state.topics.Body().conc);
   while (topic->Valid())
   {
    topic->probAux = (topic->wc[sam.GetWord()] + betaWeight) / (topic->wcTotal + state.betaSum);
    topic->prob = topic->probAux * topic->RefCount() * base;
    pSum += topic->prob;

    topic = topic->Next();
   }
  }

 // The abnormal topics...
 {
  ItemRef<Topic,Conc> * topic = state.behTopics.First()->Next();
  while (topic->Valid())
  {
   if (doc.GetBehFlags()[topic->beh]!=0)
   {
    topic->probAux = (topic->wc[sam.GetWord()] + betaWeight) / (topic->wcTotal + state.betaSum);
    float probBeh = doc.GetCluster()->GetBMN()[topic->beh] / bmvDiv;
    topic->prob = probBeh * probNewDocInst * topic->probAux;
    pSum += topic->prob;
   }
   topic = topic->Next();
  }
 }

 // The cluster instances...
 {
  ItemRef<ClusterInst,Conc> * cluInst = doc.GetCluster()->First();
  float base = probNormal * probNewDocInst / (doc.GetCluster()->RefTotal() + doc.GetCluster()->Body().conc);
  while (cluInst->Valid())
  {
   cluInst->prob = cluInst->GetTopic()->probAux * cluInst->RefCount() * base;
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

 return pSum;
}



// Code for resampling a samples topic instance assignment...
// (Everything must be assigned - no null pointers on the chain from sample to topic.)
// (You can seperatly call CalcSampleProb and put its return value in pSum if you want, though that requires that you really, really know what your doing.)
void ResampleSample(State & state, Document & doc, Sample & sam, float pSum = -1.0)
{
 // Remove the samples current assignment...
  if (sam.GetDocInst())
  {
   int beh = sam.GetDocInst()->GetClusterInst()->GetTopic()->beh;
   doc.GetBehCounts()[beh] -= 1;
   sam.SetDocInst(0);
  }

  
 // Assign probabilities to the various possibilities - there are temporary variables in the data structure to make this elegant. Sum up the total probability ready for the sampling phase. In all cases an entity is assigned the probability of using that entity with everything below it being created from scratch...
  if (pSum<0.0)
  {
   pSum = CalcSampleProb(state, doc, sam);
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

       int beh = sam.GetDocInst()->GetClusterInst()->GetTopic()->beh;
       doc.GetBehCounts()[beh] += 1;

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
      doc.GetBehCounts()[0] += 1;
      
     return;
    }

    cluInst = cluInst->Next();
   }
  }

  // Check the abnormal topics...
  {
   ItemRef<ClusterInst,Conc> * cluInst = state.behCluInsts.First()->Next();
   while (cluInst->Valid())
   {
    if (doc.GetBehFlags()[cluInst->GetTopic()->beh]!=0)
    {
     rand -= cluInst->GetTopic()->prob;
     if (rand<0.0)
     {
      // An abnormal topic has been selected - need a new document instance...
       ItemRef<DocInst,Conc> * ndi = doc.Append();
       ndi->SetClusterInst(cluInst);
       
      sam.SetDocInst(ndi);
      doc.GetBehCounts()[cluInst->GetTopic()->beh] += 1;

      return;
     }
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
      doc.GetBehCounts()[0] += 1;

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
   nt->beh = 0;

   ItemRef<ClusterInst,Conc> * nci = doc.GetCluster()->Append();
   nci->SetTopic(nt);

   ItemRef<DocInst,Conc> * ndi = doc.Append();
   ndi->SetClusterInst(nci);

   sam.SetDocInst(ndi);
   doc.GetBehCounts()[0] += 1;
}



// Code for resampling all the concentration parameters - just have to iterate through and call all the resampling methods...
void ResampleConcs(State & state, bool doClu = true, bool doDoc = true)
{
 // Concentrations for DPs from which topics and clusters are drawn...
  state.topics.Body().ResampleConc(state.topics.RefTotal(), state.topics.Size());
  state.clusters.Body().ResampleConc(state.clusters.RefTotal(), state.clusters.Size());

 // Concentrations for clusters...
  if (doClu)
  {
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

     state.rho.conc = newConc;
    }
   }
  }

 // Concentrations for documents...
  if (doDoc)
  {
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
}



// Helper function used during the left to right algorithm - a comparator for the qsort function for sorting an array of ints...
int compareInt(const void * lhs, const void * rhs)
{
 return *(int*)lhs - *(int*)rhs;
}


// Helper function for timming...
float micro_seconds()
{
 static double prev = 0.0;

 timeval tv;
 gettimeofday(&tv,0);
 double now = tv.tv_sec + (tv.tv_usec/1e6);

 float ret = (now-prev)*1e6;
 prev = now;
 return ret;
}

"""



# The actual function for Gibbs iterating the data structure - takes as input the State object as 'state' and the number of iterations to do as 'iters'...
gibbs_code = start_cpp(shared_code) + """
// State...
State s;
StatePyToCpp(state, &s);

// Declare some stuff for efficiency...
float * mn = new float[s.wordCount];

SMP smp(s.flagSets->dimensions[1], s.flagSets->dimensions[0]);
smp.SetFIA(s.flagSets);
smp.SetSampleCount(s.behSamples);

// If there is only one behaviour force disable bmn and phi estimation - things go pear-shaped otherwise...
if (s.flagSets->dimensions[1]<2) s.calcCluBmn = false;

// No point resampling phi if not resampling the bmn's...
if (s.calcCluBmn==false) s.calcPhi = false;


// Iterations...
bool verbose = false;
for (int iter=0;iter<iters;iter++)
{
 if (verbose) printf("iter %i | %f\\n", iter,  micro_seconds());
 
 // Iterate the documents...
  for (int d=0;d<s.docCount;d++)
  {
   if (verbose) printf("iter %i, doc %i | %f\\n", iter, d, micro_seconds());
   
   // Resample the documents cluster...
    if (s.doc[d].GetCluster()==0)
    {
     if (s.clusters.Size()==0)
     {
      ItemRef<Cluster,Conc> * newC = s.clusters.Append();
      newC->Body().alpha = s.rho.alpha;
      newC->Body().beta = s.rho.beta;
      newC->Body().conc = s.rho.conc;
      float * bmn = new float[s.behCount];
      for (int b=0;b<s.behCount;b++) bmn[b] = s.phi[b];
      newC->SetBMN(bmn);
      s.doc[d].SetCluster(newC);
     }
     else
     {
      s.doc[d].SetCluster(s.clusters.First());
     }
    }
    else
    {
     if (!s.oneCluster)
     {
      ResampleDocumentCluster(s, s.doc[d]);
     }
    }

   if (verbose) printf("resampled cluster | %f\\n", micro_seconds());

   // Resample the documents samples (words)...
    for (int ss=0;ss<s.doc[d].SampleCount();ss++)
    {
     ResampleSample(s, s.doc[d], s.doc[d].GetSample(ss));
    }

   if (verbose) printf("resampled words | %f\\n", micro_seconds());

   // Resample the cluster instance that each document instance is assigned to...
    if (!s.dnrDocInsts)
    {
     ResampleDocumentInstances(s,s.doc[d]);
    }

   if (verbose) printf("resampled doc instances | %f\\n", micro_seconds());

   // Resample the many concentration parameters every document - need to do this regularly to make sure the initialisation values don't cause the algorithm to get stuck (Plus its such a cheap operation that it doesn't matter if its done too frequently.)...
    if (s.resampleConcs)
    {
     ResampleConcs(s);
    }

   if (verbose) printf("resampled concentrations | %f\\n", micro_seconds());
  }


 // Resample the cluster instances assigned topics...
  if (!s.dnrCluInsts)
  {
   if (verbose) printf("resampling cluster instances... | %f\\n", micro_seconds());
   ResampleClusterInstances(s);
  }


 // Resample each clusters bmn...
  if (s.calcCluBmn)
  {
   if (verbose) printf("resampling cluster bmn's... | %f\\n", micro_seconds());
   
   // Update the prior for the smp object from phi...
    smp.SetPrior(s.phi);

   // Go through the documents and construct a list of documents belonging to each cluster...
    ItemRef<Cluster,Conc> * targClu = s.clusters.First();
    while (targClu->Valid())
    {
     targClu->first = 0;
     targClu = targClu->Next();
    }

    for (int d=0;d<s.docCount;d++)
    {
     targClu = s.doc[d].GetCluster();
     s.doc[d].next = targClu->first;
     targClu->first = &s.doc[d];
    }

   // Iterate and do the calculation for each cluster...
    targClu = s.clusters.First();
    while (targClu->Valid())
    {
     // Reset the smp object, add the prior...
      smp.Reset();
      int * priorPower = targClu->GetBehCountPrior();
      if (priorPower) smp.Add(priorPower);

     // Add samples by iterating the relevant documents...
      Document * targ = targClu->first;
      while (targ)
      {
       if (targ->GetFlagIndex()>=s.flagSets->dimensions[1])
       {
        smp.Add(targ->GetFlagIndex(), targ->GetBehCounts());
       }
       targ = targ->next;
      }

     // Extract the estimate...
      smp.Mean(targClu->GetBMN());

     targClu = targClu->Next();
    }
  }


 // Resample phi, the prior on the cluster bmn-s...
  if (s.calcPhi)
  {
   if (verbose) printf("resampling phi... | %f\\n", micro_seconds());
   EstimateDir ed(s.behCount);

   ItemRef<Cluster,Conc> * cluster = s.clusters.First();
   while (cluster->Valid())
   {
    ed.Add(cluster->GetBMN()); // Not actually correct - see below with beta for reason/justification.
    cluster = cluster->Next();
   }

   ed.Update(s.phi);
  }


 // If requested recalculate beta...
  if (s.calcBeta&&((s.topics.Size()+s.behTopics.Size()-1)>1))
  {
   if (verbose) printf("resampling beta... | %f\\n", micro_seconds());
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

   topic = s.behTopics.First()->Next(); // Skip the normal behaviour dummy.
   while (topic->Valid())
   {
    float div = 0.0;
    for (int i=0;i<s.wordCount;i++)
    {
     mn[i] = topic->wc[i] + s.beta[i];
     div += mn[i];
    }
    for (int i=0;i<s.wordCount;i++) mn[i] /= div;

    ed.Add(mn);
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



def leftRightNegLogProbWord(sample, doc, cluster, particles, cap):
  """Does a left to right estimate of the negative log probability of the words in the given document, given a sample, the documents abnormalities and a cluster assignment. cap defines a cap on the number of documents resampled before each word is sampled for inclusion - set to a negative number for no cap, but be warned that the algorithm is then O(n^2) with regard to the number of words in the document. Should be set quite high in practise for a reasonable trade off between quality and run-time."""
  code = start_cpp(shared_code) + """
  // Setup - create the state, extract the document, set its cluster...
   State state;
   StatePyToCpp(stateIn, &state);
   Document & doc = state.doc[0];

   if (cluster>=0)
   {
    // Existing cluster...
     doc.SetCluster(state.clusters.Index(cluster));
   }
   else
   {
    // New cluster...
     ItemRef<Cluster,Conc> * newC = state.clusters.Append();
     newC->Body().alpha = state.rho.alpha;
     newC->Body().beta  = state.rho.beta;
     newC->Body().conc  = state.rho.conc;
     float * bmn = new float[state.behCount];
     float bmnDiv = 0.0;
     for (int b=0;b<state.behCount;b++)
     {
      bmn[b] = state.phi[b];
      bmnDiv += state.phi[b];
     }
     for (int b=0;b<state.behCount;b++) bmn[b] /= bmnDiv;
     newC->SetBMN(bmn);
     
     doc.SetCluster(newC);
   }

  // If the cap is negative set it to include all words, otherwise we need some storage...
   int * samIndex = 0;
   if (cap<0) cap = doc.SampleCount();
   else
   {
    samIndex = new int[cap];
   }

  
  // Create some memory for storing the results into, zeroed out...
   float * samProb = new float[doc.SampleCount()];
   for (int s=0;s<doc.SampleCount();s++) samProb[s] = 0.0; 


  // Do all the particles, summing the results into the samProb array...
   for (int p=0;p<particles;p++)
   {
    // Reset the document to have no assignments to words...
     for (int s=0;s<doc.SampleCount();s++)
     {
      doc.GetSample(s).SetDocInst(0);
     }

    // Iterate and factor in the result from each sample...
     for (int s=0;s<doc.SampleCount();s++)
     {
      // Resample preceding samples - 3 scenarios with regards to the cap...
      // (Note that duplication is allowed in the random sample selection - whilst strictly forbidden the situation is such that it can not cause any issues.)
       if (s<=cap)
       {
        // Less or equal number of samples than the cap - do them all...
         for (int s2=0;s2<s;s2++)
         {
          ResampleSample(state, doc, doc.GetSample(s2));
         }
       }
       else
       {
        if (s<=cap*2)
        {
         // Need to miss some samples out, but due to numbers its best to randomly select the ones to miss rather than the ones to do...
          int missCount = s-cap;
          for (int m=0;m<missCount;m++) samIndex[m] = sample_nat(s);
          qsort(samIndex, missCount, sizeof(int), compareInt);

          for (int s2=0;s2<samIndex[0];s2++)
          {
           ResampleSample(state, doc, doc.GetSample(s2));
          }

          for (int m=0;m<missCount-1;m++)
          {
           for (int s2=samIndex[m]+1;s2<samIndex[m+1];s2++)
           {
            ResampleSample(state, doc, doc.GetSample(s2));
           }
          }
          
          for (int s2=samIndex[missCount-1]+1;s2<s;s2++)
          {
           ResampleSample(state, doc, doc.GetSample(s2));
          }
        }
        else
        {
         // Need to select a subset of samples to do...
          for (int m=0;m<cap;m++) samIndex[m] = sample_nat(s);
          qsort(samIndex, cap, sizeof(int), compareInt);

          for (int m=0;m<cap;m++)
          {
           ResampleSample(state, doc, doc.GetSample(samIndex[m]));
          }
        }
       }

      // Calculate the contribution of this sample, whilst simultaneously filling out so we can make a draw from them...
       float pSum = CalcSampleProb(state, doc, doc.GetSample(s));
       samProb[s] += (pSum - samProb[s]) / (p+1);

      // Draw an assignment for the current sample, ready for the next iteration...
       ResampleSample(state, doc, doc.GetSample(s), pSum);
     }
   }


  // Sumarise the results buffer into a single log probability and return it...
   float ret = 0.0;
   for (int s=0;s<doc.SampleCount();s++) ret += log(samProb[s]);
   return_val = ret;


  // Clean up...
   delete[] samIndex;
   delete[] samProb;
  """

  stateIn = State(doc, Params())
  stateIn.setGlobalParams(sample)
  stateIn.addPrior(sample)

  ret = weave.inline(code,['stateIn','cluster','particles','cap'] , support_code=shared_code)

  return -ret # Convert to negative log on the return - before then stick to positive.



class TestShared(unittest.TestCase):
  """Test code for the data structure."""
  def test_compile(self):
    code = start_cpp(shared_code) + """
    """
    weave.inline(code, support_code=shared_code)



# If this file is run do the unit tests...
if __name__ == '__main__':
  unittest.main()
