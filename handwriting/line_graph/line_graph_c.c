// Copyright 2016 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

#include "line_graph_c.h"

#include <stdlib.h>



// Declare the actual type object...
static PyTypeObject LineGraphType;



// Delete a chain of EdgeInt...
void EdgeInt_free(EdgeInt * to_die)
{
 while (to_die!=NULL)
 {
  EdgeInt * bird = to_die;
  to_die = to_die->next;
  free(bird);
 }
}



void SplitTag_free(SplitTag * this)
{
 // If it is paired remove its partner...
  if (this->other!=NULL)
  {
   this->other->other = NULL;
   this->other->tag = NULL; // Both point to the same memory - its bad form to assasinate it twice.
   SplitTag_free(this->other);
  }
 
 // Remove it...
  this->next->prev = this->prev;
  this->prev->next = this->next;

 // Terminate its tag storage...
  free(this->tag);

 // Throw a car at its head...
  free(this);
}

void Region_free(Region * this)
{
 if (this!=NULL)
 {
  Region_free(this->child_low);
  Region_free(this->child_high);
  free(this);
 }
}



void LineGraph_new(LineGraph * this)
{
 // Intialise everything to be empty...
  this->vertex_count = 0;
  this->vertex = NULL;

  this->edge_count = 0;
  this->edge = NULL;

  this->root = NULL;
  this->segments = -1;
}

void LineGraph_dealloc(LineGraph * this)
{
 int i;

 // Go through and free the splits/tags...
  for (i=0; i<this->edge_count; i++)
  {
   Edge * target = &this->edge[i];
   while (target->dummy.next!=&target->dummy)
   {
    SplitTag_free(target->dummy.next);
   }
  }

 // Free the vertex and edge arrays...
  this->vertex_count = 0;
  free(this->vertex);
  this->vertex = NULL;

  this->edge_count = 0;
  free(this->edge);
  this->edge = NULL;

 // Take out the spatial indexing structure...
  Region_free(this->root);
  this->root = NULL;
  
  this->segments = -1;
}


static PyObject * LineGraph_new_py(PyTypeObject * type, PyObject * args, PyObject * kwds)
{
 // Allocate the object...
  LineGraph * self = (LineGraph*)type->tp_alloc(type, 0);

 // On success construct it...
  if (self!=NULL) LineGraph_new(self);

 // Return the new object...
  return (PyObject*)self;
}

static void LineGraph_clear_py(LineGraph * self)
{
 LineGraph_dealloc(self);
}

static void LineGraph_dealloc_py(LineGraph * self)
{
 LineGraph_dealloc(self);
 self->ob_type->tp_free((PyObject*)self);
}



// Given two edges this swaps them, updating all pointers accordingly. Note the indices are for edges, not half_edges -= they are doubled and the two half edges moved as a pair...
static void LineGraph_edge_swap(LineGraph * this, int ai, int bi)
{
 if (ai==bi) return;

 // Get the pointers...
  Edge * a = &this->edge[ai];
  Edge * b = &this->edge[bi];

 // Redirect the reverse pointers involved...
  a->pos.reverse = &b->neg;
  a->neg.reverse = &b->pos;
  b->pos.reverse = &a->neg;
  b->neg.reverse = &a->pos;

 // Adjust as necessary the incident pointers from vertices...
  if (a->pos.dest->incident==&a->neg) a->pos.dest->incident = &b->neg;
  if (a->neg.dest->incident==&a->pos) a->neg.dest->incident = &b->pos;
  if (b->pos.dest->incident==&b->neg) b->pos.dest->incident = &a->neg;
  if (b->neg.dest->incident==&b->pos) b->neg.dest->incident = &a->pos;

 // Correct the outline pointers...
  HalfEdge * apn = a->pos.next;
  HalfEdge * app = a->pos.prev;
  HalfEdge * ann = a->neg.next;
  HalfEdge * anp = a->neg.prev;
  HalfEdge * bpn = b->pos.next;
  HalfEdge * bpp = b->pos.prev;
  HalfEdge * bnn = b->neg.next;
  HalfEdge * bnp = b->neg.prev;
  
  apn->prev = &b->pos;
  app->next = &b->pos;
  ann->prev = &b->neg;
  anp->next = &b->neg;
  bpn->prev = &a->pos;
  bpp->next = &a->pos;
  bnn->prev = &a->neg;
  bnp->next = &a->neg;

 // Repoint the SplitTag pointers...
  SplitTag * st;

  st = &a->dummy;
  do
  {
   st->loc = b;
   st = st->next;
  }
  while (st!=&a->dummy);

  st = &b->dummy;
  do
  {
   st->loc = a;
   st = st->next;
  }
  while (st!=&b->dummy);

 // Update the dummy next/prev pointers...
  SplitTag * an = a->dummy.next;
  SplitTag * ap = a->dummy.prev;
  SplitTag * bn = b->dummy.next;
  SplitTag * bp = b->dummy.prev;
  
  an->prev = &b->dummy;
  ap->next = &b->dummy;
  bn->prev = &a->dummy;
  bp->next = &a->dummy;

 // Swap the actual contents...
  Edge temp;

  memcpy(&temp, a, sizeof(Edge));
  memcpy(a, b, sizeof(Edge));
  memcpy(b, &temp, sizeof(Edge));
}

// Recursive, reorders edges...
static Region * LineGraph_new_region(LineGraph * this, int start, int end, int depth)
{
 int i;

 // Create the region object, fill it in...
  Region * ret = (Region*)malloc(sizeof(Region));
  
  ret->min_x = this->edge[start].pos.dest->x;
  ret->max_x = ret->min_x;
  ret->min_y = this->edge[start].pos.dest->y;
  ret->max_y = ret->min_y;

  for (i=start; i<end; i++)
  {
   Edge * targ = &this->edge[i];
   Vertex * pos = targ->pos.dest;
   Vertex * neg = targ->neg.dest;
   
   if ((pos->x-pos->radius)<ret->min_x) ret->min_x = pos->x - pos->radius;
   if ((pos->x+pos->radius)>ret->max_x) ret->max_x = pos->x + pos->radius;
   if ((pos->y-pos->radius)<ret->min_y) ret->min_y = pos->y - pos->radius;
   if ((pos->y+pos->radius)>ret->max_y) ret->max_y = pos->y + pos->radius;

   if ((neg->x-neg->radius)<ret->min_x) ret->min_x = neg->x - neg->radius;
   if ((neg->x+neg->radius)>ret->max_x) ret->max_x = neg->x + neg->radius;
   if ((neg->y-neg->radius)<ret->min_y) ret->min_y = neg->y - neg->radius;
   if ((neg->y+neg->radius)>ret->max_y) ret->max_y = neg->y + neg->radius;
  }

  ret->begin = start;
  ret->end = end;

 // Decide if its worth splitting - return if not...
  if ((((ret->max_x-ret->min_x)<16.0)&&((ret->max_y-ret->min_y)<16.0))||((end-start)<16)||(depth>128))
  {
   ret->child_low = NULL;
   ret->child_high = NULL;

   return ret;
  }

 // Choose a split point and axis...
  int axis; // 0 for x, 1 for y.
  float split;
  if ((ret->max_x-ret->min_x)>(ret->max_y-ret->min_y))
  {
   axis = 0;
   split = 0.5 * (ret->max_x + ret->min_x);
  }
  else
  {
   axis = 1;
   split = 0.5 * (ret->max_y + ret->min_y);
  }

 // Sort the edges into their split points...
  int low = start;
  int high = end-1;
  while (1)
  {
   // Advance forward until we bump into an edge that should not be in the low set...
    while (low<=high)
    {
     Edge * targ = &this->edge[low];
     Vertex * pos = targ->pos.dest;
     Vertex * neg = targ->neg.dest;
     
     if (axis==0)
     {
      if ((0.5*(pos->x + neg->x)) > split) break;
     }
     else
     {
      if ((0.5*(pos->y + neg->y)) > split) break;
     }

     low += 1;
    }

   // Advance backwards until we bump into an edge that should not be in the high set...
    while (high>=low)
    {
     Edge * targ = &this->edge[high];
     Vertex * pos = targ->pos.dest;
     Vertex * neg = targ->neg.dest;
     
     if (axis==0)
     {
      if ((0.5*(pos->x + neg->x)) < split) break;
     }
     else
     {
      if ((0.5*(pos->y + neg->y)) < split) break;
     }

     high -= 1;
    }

   // If we have crossed over then we are done...
    if (low>high) break;

   // Swap the two positions over...
    LineGraph_edge_swap(this, low, high);

    low += 1;
    high -= 1;
  }

 // Recurse to create the children...
  if ((low!=start)&&(low!=end))
  {
   ret->child_low = LineGraph_new_region(this, start, low, depth+1);
   ret->child_high = LineGraph_new_region(this, low, end, depth+1);
  }
  else
  { 
   ret->child_low = NULL;
   ret->child_high = NULL;
  }

 // Return...
  return ret;
}

void LineGraph_new_spatial_index(LineGraph * this)
{
 // Delete the old...
  Region_free(this->root);

 // Create the new...
  if (this->edge_count>0)
  {
   this->root = LineGraph_new_region(this, 0, this->edge_count, 0);
  }
  else
  {
   this->root = NULL; 
  }
}



typedef struct ListLG ListLG;

struct ListLG
{
 LineGraph * lg;
 float * hg; // Owned, 3x3.
 ListLG * next; 
};

HalfEdge * HalfEdge_Transfer(HalfEdge * source, Edge * source_base, Edge * target_base)
{
 if (source==NULL) return NULL;
 return (HalfEdge*)((char*)target_base + ((char*)source - (char*)source_base));
}


void LineGraph_from_many(LineGraph * this, ListLG * data)
{
 // Terminate previous structure...
  LineGraph_dealloc(this);
 
 // Go through the list and count how many vertices and edges are ultimatly required...
  ListLG * targ = data;
  while (targ!=NULL)
  {
   this->vertex_count += targ->lg->vertex_count;
   this->edge_count   += targ->lg->edge_count;
   targ = targ->next;
  }
 
 // Allocate memory for the vertices and edges...
  this->vertex = (Vertex*)malloc(this->vertex_count * sizeof(Vertex));
  this->edge = (Edge*)malloc(this->edge_count * sizeof(Edge));
  
 // Loop through again and copy the structures over, applying the relevant homographies...
  targ = data;
  int v_base = 0;
  int e_base = 0;
  int i;
  
  while (targ!=NULL)
  {
   // Copy over the vertices for this entry...
    for (i=0; i<targ->lg->vertex_count; i++)
    {
     Vertex * s = &targ->lg->vertex[i];
     Vertex * d = &this->vertex[v_base+i];
     
     d->incident = HalfEdge_Transfer(s->incident, targ->lg->edge, this->edge + e_base);
     
     d->x = targ->hg[0] * s->x + targ->hg[1] * s->y + targ->hg[2];
     d->y = targ->hg[3] * s->x + targ->hg[4] * s->y + targ->hg[5];
     float div = targ->hg[6] * s->x + targ->hg[7] * s->y + targ->hg[8];
     d->x /= div;
     d->y /= div;
     
     d->u = s->u;
     d->v = s->v;
     d->w = s->w;
     d->radius = s->radius;
     d->density = s->density;
     d->weight = s->weight;
     d->source = s->source;
    }
   
   // Copy over the edges for this entry (With the SplitTags)...
    for (i=0; i<targ->lg->edge_count; i++)
    {
     Edge * s = &targ->lg->edge[i];
     Edge * d = &this->edge[e_base+i];
     
     d->pos.reverse = &d->neg;
     d->pos.dest = &this->vertex[v_base + (s->pos.dest - targ->lg->vertex)];
     d->pos.next = HalfEdge_Transfer(s->pos.next, targ->lg->edge, this->edge + e_base);
     d->pos.prev = HalfEdge_Transfer(s->pos.prev, targ->lg->edge, this->edge + e_base);
     
     d->neg.reverse = &d->pos;
     d->neg.dest = &this->vertex[v_base + (s->neg.dest - targ->lg->vertex)];
     d->neg.next = HalfEdge_Transfer(s->neg.next, targ->lg->edge, this->edge + e_base);
     d->neg.prev = HalfEdge_Transfer(s->neg.prev, targ->lg->edge, this->edge + e_base);
     
     d->dummy.loc = d;
     d->dummy.next = &d->dummy;
     d->dummy.prev = &d->dummy;
     d->dummy.tag = NULL;
     d->dummy.t = 2.0;
     d->dummy.other = NULL;
     
     SplitTag * sst = s->dummy.next;
     while (sst!=&s->dummy)
     {
      if (sst->other==NULL)
      {
       SplitTag * dst = (SplitTag*)malloc(sizeof(SplitTag));
       dst->loc = d;
       
       dst->next = &d->dummy;
       dst->prev = d->dummy.prev;
       dst->next->prev = dst;
       dst->prev->next = dst;
       
       dst->tag = (sst->tag!=NULL) ? (strdup(sst->tag)) : (NULL);
       dst->t = sst->t;
       dst->other = NULL;
      }
      else
      {
       // Only insert links once, when doing the second tag...
        if (sst->other->loc < sst->loc)
        {
         SplitTag * dst = (SplitTag*)malloc(sizeof(SplitTag));
         dst->loc = d;
       
         dst->next = &d->dummy;
         dst->prev = d->dummy.prev;
         dst->next->prev = dst;
         dst->prev->next = dst;
       
         dst->tag = (sst->tag!=NULL) ? (strdup(sst->tag)) : (NULL);
         dst->t = sst->t;
         dst->other = (SplitTag*)malloc(sizeof(SplitTag));
         
         Edge * od = &this->edge[e_base + (sst->other->loc - targ->lg->edge)];
         dst->other->loc = od;
         
         dst->other->next = &od->dummy;
         dst->other->prev = od->dummy.prev;
         dst->other->next->prev = dst->other;
         dst->other->prev->next = dst->other;
         
         dst->other->tag = dst->tag;
         dst->other->t = sst->other->t;
         dst->other->other = dst;
        }
      }
       
      sst = sst->next; 
     }
     
     d->source = s->source;
    }
   
   // Move to next...
    v_base += targ->lg->vertex_count;
    e_base += targ->lg->edge_count;
    targ = targ->next; 
  }


 // Build the spatial indexing structure...
  LineGraph_new_spatial_index(this); 
}


static PyObject * LineGraph_from_many_py(LineGraph * self, PyObject * args)
{
 // Read through the command line parameters, interpreting each one in turn to build a list of line graphs with homographies attached...
  ListLG * data = NULL;
  Py_ssize_t arg_len = PyTuple_Size(args);
  float * hg = (float*)malloc(9 * sizeof(float));
  hg[0] = 1.0; hg[1] = 0.0; hg[2] = 0.0;
  hg[3] = 0.0; hg[4] = 1.0; hg[5] = 0.0;
  hg[6] = 0.0; hg[7] = 0.0; hg[8] = 1.0;
  
  Py_ssize_t i;
  int error = 0;
  
  for (i=0; i<arg_len; i++)
  {
   PyObject * targ = PyTuple_GetItem(args, i);
   
   if (PyObject_IsInstance(targ, (PyObject*)&LineGraphType))
   {
    // LineGraph...
     ListLG * nlg = (ListLG*)malloc(sizeof(ListLG));
     nlg->next = data;
     data = nlg;
     
     nlg->lg = (LineGraph*)targ;
     nlg->hg = hg;
     
     hg = (float*)malloc(9 * sizeof(float));
     hg[0] = 1.0; hg[1] = 0.0; hg[2] = 0.0;
     hg[3] = 0.0; hg[4] = 1.0; hg[5] = 0.0;
     hg[6] = 0.0; hg[7] = 0.0; hg[8] = 1.0;
   }
   else
   {
    if (PyObject_IsInstance(targ, (PyObject*)&PyArray_Type))
    {
     // A numpy array - hopefully a 3x3 homography...
      PyArrayObject * arr = (PyArrayObject*)targ;
      
      if ((arr->nd!=2)||(arr->dimensions[0]!=3)||(arr->dimensions[1]!=3))
      {
       PyErr_SetString(PyExc_TypeError, "Homographies must be 3x3.");
       error = 1;
       break;
      }
      
      if ((arr->descr->kind!='f')||((arr->descr->elsize!=sizeof(float))&&(arr->descr->elsize!=sizeof(double))))
      {
       PyErr_SetString(PyExc_TypeError, "Homographies must use either 32 or 64 bit floating point numbers.");
       error = 1;
       break;
      }

      float orig[9];
      memcpy(orig, hg, 9 * sizeof(float));
      
      int or, oc, ip;
      if (arr->descr->elsize==sizeof(float))
      {
       for (or=0; or<3; or++)
       {
        for (oc=0; oc<3; oc++)
        {
         int pos = or*3 + oc;
         hg[pos] = 0.0;
         for (ip=0; ip<3; ip++)
         {
          hg[pos] += orig[or*3+ip] * (*(float*)PyArray_GETPTR2(arr, ip, oc));
         }
        }
       }
      }
      else
      {
       for (or=0; or<3; or++)
       {
        for (oc=0; oc<3; oc++)
        {
         int pos = or*3 + oc;
         hg[pos] = 0.0;
         for (ip=0; ip<3; ip++)
         {
          hg[pos] += orig[or*3+ip] * (*(double*)PyArray_GETPTR2(arr, ip, oc));
         }
        }
       } 
      }
    }
    else
    {
     // Unrecognised - an error...
      PyErr_SetString(PyExc_TypeError, "Unrecognised type in argument string.");
      error = 1;
      break;
    }
   }
  }
  
  free(hg);
 
 // Call the method...
  if (error==0)
  {
   LineGraph_from_many(self, data);
  }
  
 // Clean up...
  while (data!=NULL)
  {
   ListLG * victim = data;
   data = data->next;
   free(victim->hg);
   free(victim);
  }
 
 // Return None, unless error...
  if (error==0)
  {
   Py_INCREF(Py_None);
   return Py_None;
  }
  else return NULL; 
}



void LineGraph_from_mask(LineGraph * this, int width, int height, char * mask, float * radius, float * density, float * weight)
{
 // Terminate any previous state...
  LineGraph_dealloc(this);

 // Set of delta for the below...
  int delta[8] = {1,
                  width+1,
                  width,
                  width-1,
                  -1,
                  -width-1,
                  -width,
                  -width+1};

 //  Create an index, by which we can record the node of each pixel, as relevant - makes adding edges easier...
  int * index = (int*)malloc(width * height * sizeof(int));

 // First pass - assign a node for every pixel marked as true and count them and the number of edges...
 // (We double count edges in the below, and fix it later.)
  int y, x;
  for (y=0; y<height; y++)
  {
   for (x=0; x<width; x++)
   {
    int offset = y*width + x;
    if (mask[offset]!=0)
    {
     // Assign a node to it and count the number...
      index[offset] = this->vertex_count;
      this->vertex_count += 1;

     // Count how many edges exist...
      char p[8];
      p[0] = (x+1<width) ? mask[offset+delta[0]] : 0;
      p[1] = ((x+1<width)&&(y+1<height)) ? mask[offset+delta[1]] : 0;
      p[2] = (y+1<height) ? mask[offset+delta[2]] : 0;
      p[3] = ((x>0)&&(y+1<height)) ? mask[offset+delta[3]] : 0;
      p[4] = (x>0) ? mask[offset+delta[4]] : 0;
      p[5] = ((x>0)&&(y>0)) ? mask[offset+delta[5]] : 0;
      p[6] = (y>0) ? mask[offset+delta[6]] : 0;
      p[7] = ((x+1<width)&&(y>0)) ? mask[offset+delta[7]] : 0;

      this->edge_count += p[0] + p[1] + p[2] + p[3] + p[4] + p[5] + p[6] + p[7];

     // Factor in the removal of short-cuts, which are otherwise counted by the above...
      if ((p[0]!=0)&&(p[1]==0)&&(p[2]!=0)) this->edge_count -= 2;
      if ((p[2]!=0)&&(p[3]==0)&&(p[4]!=0)) this->edge_count -= 2;
      if ((p[4]!=0)&&(p[5]==0)&&(p[6]!=0)) this->edge_count -= 2;
      if ((p[6]!=0)&&(p[7]==0)&&(p[0]!=0)) this->edge_count -= 2;
      
      if ((p[0]!=0)&&(p[1]!=0)&&(p[2]!=0)) this->edge_count -= 1;
      if ((p[2]!=0)&&(p[3]!=0)&&(p[4]!=0)) this->edge_count -= 1;
      if ((p[4]!=0)&&(p[5]!=0)&&(p[6]!=0)) this->edge_count -= 1;
      if ((p[6]!=0)&&(p[7]!=0)&&(p[0]!=0)) this->edge_count -= 1;
    }
   }
  }

 // Create the nodes and edges...
  this->vertex = (Vertex*)malloc(this->vertex_count * sizeof(Vertex));
  this->edge_count /= 2;
  this->edge = (Edge*)malloc(this->edge_count * sizeof(Edge));
  
 // Quick pass through the edges to setup the SplitTag linked lists and reverse pointers...
  int i;
  for (i=0; i<this->edge_count; i++)
  {
   Edge * targ = &this->edge[i];

   targ->pos.reverse = &targ->neg;
   targ->neg.reverse = &targ->pos;
   
   targ->dummy.loc = targ;
   targ->dummy.next = &targ->dummy;
   targ->dummy.prev = &targ->dummy;
   targ->dummy.tag = NULL;
   targ->dummy.t = -1.0;
   targ->dummy.other = NULL;
   
   targ->source = -1;
  }

 // Second pass - fill in the nodes and edges...
  int out_edge = 0;
  for (y=0; y<height; y++)
  {
   for (x=0; x<width; x++)
   {
    int offset = y*width + x;
    if (mask[offset]!=0)
    {
     Vertex * targ = &this->vertex[index[offset]];

     // Set the actual vertex object to something sane...
      targ->incident = NULL;
      targ->x = x + 0.5; // 0.5 as correction to put it in the center of the pixel.
      targ->y = y + 0.5;
      targ->u = x + 0.5;
      targ->v = y + 0.5;
      targ->w = (radius!=NULL) ? radius[offset] : 0.5;
      targ->radius = targ->w;
      targ->density = (density!=NULL) ? density[offset] : 1.0;
      targ->weight = (weight!=NULL) ? weight[offset] : 1.0;
      targ->source = -1;

     // Calculate adjacency information...
      char p[8];
      p[0] = (x+1<width) ? mask[offset+delta[0]] : 0;
      p[1] = ((x+1<width)&&(y+1<height)) ? mask[offset+delta[1]] : 0;
      p[2] = (y+1<height) ? mask[offset+delta[2]] : 0;
      p[3] = ((x>0)&&(y+1<height)) ? mask[offset+delta[3]] : 0;
      p[4] = (x>0) ? mask[offset+delta[4]] : 0;
      p[5] = ((x>0)&&(y>0)) ? mask[offset+delta[5]] : 0;
      p[6] = (y>0) ? mask[offset+delta[6]] : 0;
      p[7] = ((x+1<width)&&(y>0)) ? mask[offset+delta[7]] : 0;

     // Go through and add all the edges...
      int i;
      HalfEdge * he[8]; // Of half-edges that are *leaving* this vertex.
      HalfEdge * prev = NULL; // Highest index he entry.

      for (i=0; i<8; i++)
      {
       he[i] = NULL;
       if (p[i]!=0)
       {
        if (i<4)
        {
         // Check we are not about to create a short cut...
          char ok = 1;
          
          if ((i==1)&&((p[0]!=0)||(p[2]!=0))) ok = 0;
          if ((i==3)&&((p[2]!=0)||(p[4]!=0))) ok = 0;
         
          if (ok!=0)
          {
           // Create the new half edge pair...
            he[i] = &this->edge[out_edge].pos;
            out_edge += 1;

            he[i]->dest = &this->vertex[index[offset+delta[i]]];
            he[i]->reverse->dest = targ;
          }
        }
        else
        {
         // Edge already exists - need to find it ready for the next step...
          HalfEdge * s = this->vertex[index[offset+delta[i]]].incident;
          HalfEdge * t = s;
          
          do
          {
           if (t->dest==targ)
           {
            he[i] = t->reverse;
            break;
           }
           
           t = t->reverse->next;
          }
          while (s!=t);
        }
        
        if (he[i]!=NULL)
        {
         if (targ->incident==NULL) targ->incident = he[i];
         prev = he[i];
        }
       }
      }

     // Second loop of the edges, to tie the wings together - we dont want them to fly away!..
      if (prev!=NULL)
      {
       for (i=0; i<8; i++)
       {
        if (he[i]!=NULL)
        {
         he[i]->prev = prev->reverse;
         he[i]->prev->next = he[i];

         prev = he[i];
        }
       }
      }
    }
   }
  }

 // Clean up...
  free(index);

 // Build the spatial indexing structure...
  LineGraph_new_spatial_index(this);
}


static PyObject * LineGraph_from_mask_py(LineGraph * self, PyObject * args)
{
 // Extract the parameters
  PyArrayObject * mask;
  PyArrayObject * radius = NULL;
  PyArrayObject * density = NULL;
  PyArrayObject * weight = NULL;
  if (!PyArg_ParseTuple(args, "O!|O!O!O!", &PyArray_Type, &mask, &PyArray_Type, &radius, &PyArray_Type, &density, &PyArray_Type, &weight)) return NULL;

 // Verify the arrays are suitable...
  if ((mask->nd!=2)||((radius!=NULL)&&(radius->nd!=2))||((density!=NULL)&&(density->nd!=2))||((weight!=NULL)&&(weight->nd!=2)))
  {
   PyErr_SetString(PyExc_TypeError, "All input arrays must be 2D");
   return NULL;
  }

  if (((radius!=NULL)&&((radius->dimensions[0]!=mask->dimensions[0])||(radius->dimensions[1]!=mask->dimensions[1]))) || ((density!=NULL)&&((density->dimensions[0]!=mask->dimensions[0])||(density->dimensions[1]!=mask->dimensions[1]))) || ((weight!=NULL)&&((weight->dimensions[0]!=weight->dimensions[0])||(weight->dimensions[1]!=weight->dimensions[1]))))
  {
   PyErr_SetString(PyExc_TypeError, "All input arrays must have the same sizes");
   return NULL;
  }

  if (mask->descr->kind!='b' || mask->descr->elsize!=sizeof(char))
  {
   PyErr_SetString(PyExc_TypeError, "mask must be of boolean type.");
   return NULL;
  }

  if ((radius!=NULL)&&(radius->descr->kind!='f' || radius->descr->elsize!=sizeof(float)))
  {
   PyErr_SetString(PyExc_TypeError, "radius must be a 32 bit float.");
   return NULL;
  }

  if ((density!=NULL)&&(density->descr->kind!='f' || density->descr->elsize!=sizeof(float)))
  {
   PyErr_SetString(PyExc_TypeError, "density must be a 32 bit float.");
   return NULL;
  }
  
  if ((weight!=NULL)&&(weight->descr->kind!='f' ||weight->descr->elsize!=sizeof(float)))
  {
   PyErr_SetString(PyExc_TypeError, "weight must be a 32 bit float.");
   return NULL;
  }

  if ((radius!=NULL)&&(radius->strides[0]!=(sizeof(float)*radius->dimensions[1])))
  {
   PyErr_SetString(PyExc_TypeError, "radius is not tightly packed.");
   return NULL;
  }

  if ((density!=NULL)&&(density->strides[0]!=(sizeof(float)*density->dimensions[1])))
  {
   PyErr_SetString(PyExc_TypeError, "density is not tightly packed.");
   return NULL;
  }
  
  if ((weight!=NULL)&&(weight->strides[0]!=(sizeof(float)*weight->dimensions[1])))
  {
   PyErr_SetString(PyExc_TypeError, "weight is not tightly packed.");
   return NULL;
  }

 // Extract the required pointers...
  char * mask_ptr = (char*)(void*)mask->data;
  float * radius_ptr = (radius!=NULL) ? (float*)(void*)radius->data : NULL;
  float * density_ptr = (density!=NULL) ? (float*)(void*)density->data : NULL;
  float * weight_ptr = (weight!=NULL) ? (float*)(void*)weight->data : NULL;

 // Call through to the C method that does the work...
  LineGraph_from_mask(self, mask->dimensions[1], mask->dimensions[0], mask_ptr, radius_ptr, density_ptr, weight_ptr);

 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}



// Helper - fills in an edge, linking it all up correctly (Gets the ability to go around a vertex right.)...
void Edge_init(Edge * this, Vertex * neg, Vertex * pos)
{
 // Fill in the easy to do stuff...
  this->pos.dest = pos;
  this->neg.dest = neg;
  
  this->pos.reverse = &this->neg;
  this->neg.reverse = &this->pos;
  
  this->dummy.loc = this;
  this->dummy.next = &this->dummy;
  this->dummy.prev = &this->dummy;
  this->dummy.tag = NULL;
  this->dummy.t = 2.0;
  this->dummy.other = NULL;
  
  this->source = -1;
 
 // Handle inserting the edge into the negative vertex edge ring... 
  if (neg->incident==NULL)
  {
   neg->incident = &this->pos;
   this->pos.prev = &this->neg;
   this->neg.next = &this->pos;
  }
  else
  {
   float angle = atan2(pos->y - neg->y, pos->x - neg->x); 
   
   HalfEdge * targ = neg->incident;
   while (1)
   {
    HalfEdge * next = targ->reverse->next;
    float targ_ang = atan2(targ->dest->y - neg->y, targ->dest->x - neg->x);
    float next_ang = atan2(next->dest->y - neg->y, next->dest->x - neg->x);
    
    float ang = angle - targ_ang;
    if (ang<0.0) ang += M_PI*2.0;
    
    next_ang -= targ_ang;
    if (next_ang<1e-12) next_ang += M_PI*2.0;
    
    if (ang<=next_ang)
    {
     this->pos.prev = targ->reverse;
     this->neg.next = next;
     
     this->pos.prev->next = &this->pos;
     this->neg.next->prev = &this->neg;
     
     break; 
    }
     
    targ = next;
   }
  }
 
 // Handle inserting the edge into the positive vertex edge ring...
  if (pos->incident==NULL)
  {
   pos->incident = &this->neg;
   this->pos.next = &this->neg;
   this->neg.prev = &this->pos;
  }
  else
  {
   float angle = atan2(neg->y - pos->y, neg->x - pos->x); 
   
   HalfEdge * targ = pos->incident;
   while (1)
   {
    HalfEdge * next = targ->reverse->next;
    float targ_ang = atan2(targ->dest->y - pos->y, targ->dest->x - pos->x);
    float next_ang = atan2(next->dest->y - pos->y, next->dest->x - pos->x);
    
    float ang = angle - targ_ang;
    if (ang<0.0) ang += M_PI*2.0;
    
    next_ang -= targ_ang;
    if (next_ang<1e-12) next_ang += M_PI*2.0;
    
    if (ang<=next_ang)
    {
     this->pos.next = next;
     this->neg.prev = targ->reverse;
     
     this->pos.next->prev = &this->pos;
     this->neg.prev->next = &this->neg;
     
     break; 
    }
     
    targ = next;
   }
  }
}



typedef struct VertSegList VertSegList;

struct VertSegList
{
 VertSegList * next;
 
 int vertex;
 int segment;
};

// Do not call unless there is a segmentation on the source. Returns a list of vertices and the segmentation on the other side - the user is responsible for calling free on each node in the returned list...
VertSegList * LineGraph_from_segment(LineGraph * this, LineGraph * source, int segment)
{
 VertSegList * ret = NULL;
 
 // Expunge any old data...
  LineGraph_dealloc(this);
   
 // Go through all the edges and count how many vertices and edges are needed...
  int i;
  for (i=0; i<source->edge_count; i++)
  {
   Edge * targ = &source->edge[i];
   
   int seg = targ->segment;
   
   // Negative vertex...
    if (seg==segment)
    {
     // Include this vertex - need to count it only once, hence only do so if we are the incident edge...
      if (targ->neg.dest->incident==&targ->pos)
      {
       this->vertex_count += 1;
      }
    }
   
   // The segments...
    SplitTag * st = targ->dummy.next;
    while (st!=&targ->dummy)
    {
     if ((st->tag==NULL)&&(st->other==NULL)&&(st->segment!=seg))
     {
      if (seg==segment) this->edge_count += 1;
      if ((seg==segment)||(st->segment==segment)) this->vertex_count += 1;
      seg = st->segment;
     }
      
     st = st->next;
    }
   
    if (seg==segment) this->edge_count += 1; // Last bit of edge.
   
   // Positive vertex...
    if (seg==segment)
    {
     // Include this vertex - need to count it only once, hence only do so if we are the incident edge...
      if (targ->pos.dest->incident==&targ->neg)
      {
       this->vertex_count += 1;
      }
    }
  }
 
 // Create the storage...
  this->vertex = (Vertex*)malloc(this->vertex_count * sizeof(Vertex));
  this->edge = (Edge*)malloc(this->edge_count * sizeof(Edge));
  
  int new_vert = 0;
  int new_edge = 0;
 
 // Create an index for the vertex array of this with the indexing of source, so we can reuse vertices as needed...
  Vertex ** vert_index = (Vertex**)malloc(source->vertex_count * sizeof(Vertex*));
  memset(vert_index, 0, source->vertex_count * sizeof(Vertex*));
 
 // Iterate all the edges again, this time filling in the structure as needed...
  for (i=0; i<source->edge_count; i++)
  {
   Edge * targ = &source->edge[i];
   
   int seg = targ->segment;
   Vertex * prev_vert = NULL;
   float prev_t = 0.0;
   SplitTag * prev_st = &targ->dummy;
   
   // Negative vertex...
    if (seg==segment)
    {
     // If this vertex has already been created fetch it, otherwise create it...
      int source_index = targ->neg.dest - source->vertex;
      prev_vert = vert_index[source_index];
      if (prev_vert==NULL)
      {
       prev_vert = &this->vertex[new_vert];
       new_vert += 1;
       vert_index[source_index] = prev_vert;
       
       prev_vert->incident = NULL;
       prev_vert->x = targ->neg.dest->x;
       prev_vert->y = targ->neg.dest->y;
       prev_vert->u = targ->neg.dest->u;
       prev_vert->v = targ->neg.dest->v;
       prev_vert->w = targ->neg.dest->w;
       prev_vert->radius = targ->neg.dest->radius;
       prev_vert->density = targ->neg.dest->density;
       prev_vert->weight = targ->neg.dest->weight;
       prev_vert->source = source_index;
      }
    }
    
   // The segments...
    SplitTag * st = targ->dummy.next;
    while (st!=&targ->dummy)
    {
     if ((st->tag==NULL)&&(st->other==NULL)&&(st->segment!=seg))
     {
      if ((seg==segment)||(st->segment==segment))
      {
       VertSegList * vsl = (VertSegList*)malloc(sizeof(VertSegList));
       vsl->next = ret;
       ret = vsl;
       vsl->vertex = new_vert;
       vsl->segment = (seg!=segment) ? seg : st->segment;
       
       Vertex * next_vert = &this->vertex[new_vert];
       new_vert += 1;
       float next_t = st->t;
       
       next_vert->incident = NULL;
       next_vert->x = targ->neg.dest->x * (1.0-st->t) + targ->pos.dest->x * st->t;
       next_vert->y = targ->neg.dest->y * (1.0-st->t) + targ->pos.dest->y * st->t;
       next_vert->u = targ->neg.dest->u * (1.0-st->t) + targ->pos.dest->u * st->t;
       next_vert->v = targ->neg.dest->v * (1.0-st->t) + targ->pos.dest->v * st->t;
       next_vert->w = targ->neg.dest->w * (1.0-st->t) + targ->pos.dest->w * st->t;
       next_vert->radius = targ->neg.dest->radius * (1.0-st->t) + targ->pos.dest->radius * st->t;
       next_vert->density = targ->neg.dest->density * (1.0-st->t) + targ->pos.dest->density * st->t;
       next_vert->weight = targ->neg.dest->weight * (1.0-st->t) + targ->pos.dest->weight * st->t;
       next_vert->source = -1;
       
       if (seg==segment)
       {
        Edge * ne = &this->edge[new_edge];
        new_edge += 1;
        
        Edge_init(ne, prev_vert, next_vert);
        ne->source = i;
        
        SplitTag * st_rev = st->prev;
        while (st_rev!=prev_st)
        {
         if ((st_rev->tag!=NULL)&&(st_rev->other==NULL)) // Only do basic tags.
         {
          // Copy this tag over to the new data structure...
           SplitTag * new_st = (SplitTag*)malloc(sizeof(SplitTag));
           
           new_st->loc = ne;
           
           new_st->next = ne->dummy.next;
           new_st->prev = &ne->dummy;
           new_st->next->prev = new_st;
           new_st->prev->next = new_st;
           
           new_st->tag = strdup(st_rev->tag);
           new_st->t = (st_rev->t - prev_t) / (next_t - prev_t);
           new_st->other = NULL;
         }
         
         if ((st_rev->tag==NULL)&&(st_rev->other!=NULL)) // Do links to connect seperate parts, but not splits, as they define what we are doing and should all be on the edge.
         {
          // Only do it if it has a destination (the other side is also part of this segment.)...
           int other_seg = st_rev->other->loc->segment;
           SplitTag * st_rev_other = st_rev->other->prev;
           while (st_rev_other!=&st_rev->other->loc->dummy)
           {
            if ((st_rev_other->tag==NULL)&&(st_rev_other->other==NULL))
            {
             other_seg = st_rev_other->segment;
             break; 
            }
            st_rev_other = st_rev_other->prev;
           }
           
           if (other_seg==segment)
           {
            SplitTag * new_st = (SplitTag*)malloc(sizeof(SplitTag));
           
            new_st->loc = ne;
           
            new_st->next = ne->dummy.next;
            new_st->prev = &ne->dummy;
            new_st->next->prev = new_st;
            new_st->prev->next = new_st;
           
            new_st->tag = NULL;
            new_st->t = (st_rev->t - prev_t) / (next_t - prev_t);
            
            if (st_rev->loc < st_rev->other->loc)
            {
             st_rev->other = new_st;
            }
            else
            {
             new_st->other = st_rev->other->other;
             
             new_st->other->other = new_st;
             st_rev->other->other = st_rev;
            }
           }
         }
          
         st_rev = st_rev->prev;
        }
       }
       
       prev_vert = next_vert;
       prev_t = next_t;
       prev_st = st;
      }

      seg = st->segment;
     }
      
     st = st->next;
    }
   
   // Positive vertex...
    if (seg==segment)
    {
     int source_index = targ->pos.dest - source->vertex;
     Vertex * next_vert = vert_index[source_index];
     float next_t = 1.0;
     
     if (next_vert==NULL)
     {
      next_vert = &this->vertex[new_vert];
      new_vert += 1;
      
      vert_index[source_index] = next_vert;
       
      next_vert->incident = NULL;
      next_vert->x = targ->pos.dest->x;
      next_vert->y = targ->pos.dest->y;
      next_vert->u = targ->pos.dest->u;
      next_vert->v = targ->pos.dest->v;
      next_vert->w = targ->pos.dest->w;
      next_vert->radius = targ->pos.dest->radius;
      next_vert->density = targ->pos.dest->density;
      next_vert->weight = targ->pos.dest->weight;
      next_vert->source = source_index;
     }
      
     Edge * ne = &this->edge[new_edge];
     new_edge += 1;
        
     Edge_init(ne, prev_vert, next_vert);
     ne->source = i;
        
     SplitTag * st_rev = targ->dummy.prev;
     while (st_rev!=prev_st)
     {
      if ((st_rev->tag!=NULL)&&(st_rev->other==NULL)) // Only do basic tags.
      {
       // Copy this tag over to the new data structure...
        SplitTag * new_st = (SplitTag*)malloc(sizeof(SplitTag));
        
        new_st->loc = ne;
        
        new_st->next = ne->dummy.next;
        new_st->prev = &ne->dummy;
        new_st->next->prev = new_st;
        new_st->prev->next = new_st;
           
        new_st->tag = strdup(st_rev->tag);
        new_st->t = (st_rev->t - prev_t) / (next_t - prev_t);
        new_st->other = NULL;
      }
      
      if ((st_rev->tag==NULL)&&(st_rev->other!=NULL)) // Do links to connect seperate parts, but not splits, as they define what we are doing and should all be on the edge.
      {
       // Only do it if it has a destination (the other side is also part of this segment.)...
        int other_seg = st_rev->other->loc->segment;
        SplitTag * st_rev_other = st_rev->other->prev;
        while (st_rev_other!=&st_rev->other->loc->dummy)
        {
         if ((st_rev_other->tag==NULL)&&(st_rev_other->other==NULL))
         {
          other_seg = st_rev_other->segment;
          break; 
         }
         st_rev_other = st_rev_other->prev;
        }
           
        if (other_seg==segment)
        {
         SplitTag * new_st = (SplitTag*)malloc(sizeof(SplitTag));
         
         new_st->loc = ne;
           
         new_st->next = ne->dummy.next;
         new_st->prev = &ne->dummy;
         new_st->next->prev = new_st;
         new_st->prev->next = new_st;
           
         new_st->tag = NULL;
         new_st->t = (st_rev->t - prev_t) / (next_t - prev_t);
         
         if (st_rev->loc < st_rev->other->loc)
         {
          st_rev->other = new_st;
         }
         else
         {
          new_st->other = st_rev->other->other;
            
          new_st->other->other = new_st;
          st_rev->other->other = st_rev;
         }
        }
      }
     
      st_rev = st_rev->prev;
     }
    }
  }

 // Clean up...
  free(vert_index);
  
 // Build the spatial indexing structure...
  LineGraph_new_spatial_index(this);
  
 return ret;
}


static PyObject * LineGraph_from_segment_py(LineGraph * self, PyObject * args)
{
 // Extract the parameters
  LineGraph * source;
  int segment;
  if (!PyArg_ParseTuple(args, "O!i", &LineGraphType, &source, &segment)) return NULL;
 
 // Check there is a valid segmentation...
  if (source->segments<0)
  {
   PyErr_SetString(PyExc_RuntimeError, "Source LineGraph does not have a valid segmentation.");
   return NULL;
  }
  
 // Do the work...
  VertSegList * vsl = LineGraph_from_segment(self, source, segment);
  
 // Create the return...
  PyObject * ret = PyList_New(0);
  
  VertSegList * targ = vsl;
  while (targ!=NULL)
  {
   PyList_Append(ret, Py_BuildValue("(i,i)", targ->vertex, targ->segment));
   targ = targ->next; 
  }
  
 // Clean up...
  while (vsl!=NULL)
  {
   targ = vsl;
   vsl = vsl->next;
   free(targ);
  }
  
 // Return...
  return ret;
}



// For the below function...
typedef struct EdgeDist EdgeDist;

struct EdgeDist
{
 HalfEdge * path; // Path we are following to get
 float dist; // Distance thus far from v1, to the end of the path.
 float cost; // Above plus the euclidean distance to the destination.
};


void edge_dist_heap_add(EdgeDist ed, EdgeDist * heap, int * heap_size)
{
 int targ = *heap_size;
 *heap_size += 1;
 
 while (targ!=0)
 {
  int parent = (targ-1)/2;
  if (heap[parent].cost<ed.cost) break;
  
  heap[targ] = heap[parent];
  targ = parent;
 }
 
 heap[targ] = ed;
}

void edge_dist_heap_rem(int targ, EdgeDist * heap, int * heap_size)
{
 *heap_size -= 1;
 
 while (1)
 {
  int c1 = 2*targ + 1;
  int c2 = 2*targ + 2;
  
  if (c1>=*heap_size) break;
  
  int lc;
  if (c2>=*heap_size) lc = c1;
  else
  {
   if (heap[c1].cost<heap[c2].cost) lc = c1;
   else lc = c2;    
  }
  
  if (heap[*heap_size].cost < heap[lc].cost) break;
  
  heap[targ] = heap[lc];
  targ = lc;
 }
 
 heap[targ] = heap[*heap_size];
}


// Returns 0 on error.
char LineGraph_from_path(LineGraph * this, LineGraph * source, int v1, int v2, int * out1, int * out2)
{
 // First we need to find the set of vertices to keep - its an A* implimentation...
  // Prepare the data structure...
   EdgeDist * heap = (EdgeDist*)malloc(source->edge_count * sizeof(EdgeDist));
   int heap_size = 0;
   
   int * prev = (int*)malloc(source->vertex_count * sizeof(int));
   
   int i;
   for (i=0; i<source->vertex_count; i++) prev[i] = -1;
   prev[v1] = -2;
   
  // Seed them with the starting edges...
   HalfEdge * he = source->vertex[v1].incident;
   if (he!=NULL)
   {
    do
    {
     EdgeDist ed;
     ed.path = he;
     
     float dx = he->dest->x - he->reverse->dest->x;
     float dy = he->dest->y - he->reverse->dest->y;
     ed.dist = sqrt(dx*dx + dy*dy);
     
     dx = source->vertex[v2].x - he->dest->x;
     dy = source->vertex[v2].y - he->dest->y;
     ed.cost = ed.dist + sqrt(dx*dx + dy*dy);
     
     edge_dist_heap_add(ed, heap, &heap_size);
     
     he = he->reverse->next;
    }
    while (he!=source->vertex[v1].incident);
   }
   
  // Run the A* algorithm to find the best path...
   while (heap_size!=0)
   {
    // Pop an item from the heap...
     EdgeDist targ = heap[0];
     edge_dist_heap_rem(0, heap, &heap_size);
     
    // Check if its going somewhere we have already explored, in which case ignore it - there was a quicker way of getting there...
     int dest_vert = targ.path->dest - source->vertex;
     if (prev[dest_vert]!=-1) continue;
     prev[dest_vert] = targ.path->reverse->dest - source->vertex;
     
    // Check if we have just reached our destination - if so stop...
     if (dest_vert==v2) break;
     
    // Iterate its edges - put entries into the heap for each edge that goes somewhere unexplored...
     he = targ.path->dest->incident;
     if (he!=NULL)
     {
      do
      {
       if (prev[he->dest - source->vertex]==-1)
       {
        EdgeDist ed;
        ed.path = he;
        
        float dx = he->dest->x - he->reverse->dest->x;
        float dy = he->dest->y - he->reverse->dest->y;
        ed.dist = targ.dist + sqrt(dx*dx + dy*dy);
        
        dx = source->vertex[v2].x - he->dest->x;
        dy = source->vertex[v2].y - he->dest->y;
        ed.cost = ed.dist + sqrt(dx*dx + dy*dy);
        
        edge_dist_heap_add(ed, heap, &heap_size);
       }
       
       he = he->reverse->next;
      }
      while (he!=targ.path->dest->incident);
     }
   }
   
   if (prev[v2]==-1) // This would mean the two vertices belong to disjoint islands.
   {
    free(prev);
    free(heap);
    return 0; 
   }


 // Trash the previous data...
  LineGraph_dealloc(this);


 // Count how many vertices and edges we will need - allocate storage for them...
  this->vertex_count = 1;
  
  int v = v2;
  while (prev[v]!=-2)
  {
   this->vertex_count += 1;
   this->edge_count += 1;
   v = prev[v]; 
  }
  
  this->vertex = (Vertex*)malloc(this->vertex_count * sizeof(Vertex));
  this->edge = (Edge*)malloc(this->edge_count * sizeof(Edge));


 // Follow the path and add in the vertices and edges as we go...
  int new_vert = 0;
  int new_edge = 0;
  
  this->vertex[new_vert].incident = NULL;
  this->vertex[new_vert].x = source->vertex[v2].x;
  this->vertex[new_vert].y = source->vertex[v2].y;
  this->vertex[new_vert].u = source->vertex[v2].u;
  this->vertex[new_vert].v = source->vertex[v2].v;
  this->vertex[new_vert].w = source->vertex[v2].w;
  this->vertex[new_vert].radius  = source->vertex[v2].radius;
  this->vertex[new_vert].density = source->vertex[v2].density;
  this->vertex[new_vert].weight  = source->vertex[v2].weight;  
  this->vertex[new_vert].source  = v2;
  
  if (out2!=NULL) *out2 = new_vert;
  new_vert += 1;
  
  v = v2;
  while (prev[v]!=-2)
  {
   // Create the destination vertex...
    this->vertex[new_vert].incident = NULL;
    this->vertex[new_vert].x = source->vertex[prev[v]].x;
    this->vertex[new_vert].y = source->vertex[prev[v]].y;
    this->vertex[new_vert].u = source->vertex[prev[v]].u;
    this->vertex[new_vert].v = source->vertex[prev[v]].v;
    this->vertex[new_vert].w = source->vertex[prev[v]].w;
    this->vertex[new_vert].radius  = source->vertex[prev[v]].radius;
    this->vertex[new_vert].density = source->vertex[prev[v]].density;
    this->vertex[new_vert].weight  = source->vertex[prev[v]].weight;  
    this->vertex[new_vert].source  = prev[v];
    
    if(prev[v]==v1)
    {
     if (out1!=NULL) *out1 = new_vert; 
    }
    new_vert += 1;
    
   // Create the edge connecting the destination/source...
    Edge_init(this->edge + new_edge, this->vertex + new_vert-2, this->vertex + new_vert-1);
    
    new_edge += 1;
   
   // To next...
    v = prev[v]; 
  }

   
 // Build the spatial structure...
  LineGraph_new_spatial_index(this);
 
 // Clean up...
  free(prev);
  free(heap);
  
 return 1;
}


static PyObject * LineGraph_from_path_py(LineGraph * self, PyObject * args)
{
 // Extract the parameters...
  LineGraph * source;
  int v1;
  int v2;
  if (!PyArg_ParseTuple(args, "O!ii", &LineGraphType, &source, &v1, &v2)) return NULL;
 
 // Verify the vertex numbers are valid...
  if ((v1<0)||(v1>=source->vertex_count)||(v2<0)||(v2>=source->vertex_count)||(v1==v2))
  {
   PyErr_SetString(PyExc_RuntimeError, "Not a valid pair of vertex indices with which to define a line");
   return NULL;
  }
  
 // Call through to the actual code...
  int out1, out2;
  if (LineGraph_from_path(self, source, v1, v2, &out1, &out2)==0)
  {
   PyErr_SetString(PyExc_RuntimeError, "vertices are on seperate islands - no path exists");
   return NULL;
  }
  
 // Return the input pair translated to the new structure...
  return Py_BuildValue("(i,i)", out1, out2);
}



void LineGraph_from_vertices(LineGraph * this, LineGraph * source, int count, int * indices)
{
 // First invert the indices vector, so we know where to put each vertex in source...
  int * dest = (int*)malloc(source->vertex_count * sizeof(int));
  
  int i;
  for (i=0; i<source->vertex_count; i++) dest[i] = -1;
  for (i=0; i<count; i++) dest[indices[i]] = i;
 
 // Terminate any previous content...
  LineGraph_dealloc(this);
 
 // Count how many edges we need...
  this->vertex_count = count;
  
  for (i=0; i<source->edge_count; i++)
  {
   Edge * targ = source->edge + i;
   if ((dest[targ->pos.dest - source->vertex]>=0)&&(dest[targ->neg.dest - source->vertex]>=0))
   {
    this->edge_count += 1;
   }
  }
  
 // Create the output...
  this->vertex = (Vertex*)malloc(this->vertex_count * sizeof(Vertex));
  this->edge = (Edge*)malloc(this->edge_count * sizeof(Edge));
 
 // Go through and copy over the vertices...
  for (i=0; i<source->vertex_count; i++)
  {
   if (dest[i]>=0)
   {
    Vertex * from = source->vertex + i;
    Vertex * to   = this->vertex + dest[i];
    
    to->incident = NULL;
    
    to->x = from->x;
    to->y = from->y;
    
    to->u = from->u;
    to->v = from->v;
    to->w = from->w;
    
    to->radius = from->radius;
    to->density = from->density;
    to->weight = from->weight;
    
    to->source = i;
   }
  }
 
 // Go through and copy over the edges...
  int edge_out = 0;
  for (i=0; i<source->edge_count; i++)
  {
   Edge * from = source->edge + i;
   if ((dest[from->pos.dest - source->vertex]>=0)&&(dest[from->neg.dest - source->vertex]>=0))
   {
    Vertex * neg = this->vertex + dest[from->neg.dest - source->vertex];
    Vertex * pos = this->vertex + dest[from->pos.dest - source->vertex];
    Edge_init(this->edge + edge_out, neg, pos);
    
    ++edge_out;
   }
  }
 
 // Create the spatial indexing structure...
  LineGraph_new_spatial_index(this);
  
 // Clean up...
  free(dest);
}


static PyObject * LineGraph_from_vertices_py(LineGraph * self, PyObject * args)
{
 // Extract the parameters...
  LineGraph * source;
  PyObject * vert_list;
  if (!PyArg_ParseTuple(args, "O!O!", &LineGraphType, &source, &PyList_Type, &vert_list)) return NULL;
 
 // Convert the list into an array...
  int * list = (int*)malloc(PyList_Size(vert_list) * sizeof(int));
  int i;
  for (i=0; i<PyList_Size(vert_list); i++)
  {
   PyObject * val = PyList_GetItem(vert_list, i);
   if (PyInt_Check(val)==0)
   {
    PyErr_SetString(PyExc_RuntimeError, "list of vertex indices contains something that is not a number.");
    free(list);
    return NULL; 
   }
   
   list[i] = PyInt_AsLong(val);
   
   if ((list[i]<0)||(list[i]>=source->vertex_count))
   {
    PyErr_SetString(PyExc_RuntimeError, "vertex index is out of range for the source line graph");
    free(list);
    return NULL; 
   }
  }
  
 // Call through to the method that does the work...
  LineGraph_from_vertices(self, source, PyList_Size(vert_list), list);
  
 // Clean up...
  free(list);
  
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}



static PyObject * LineGraph_as_dict_py(LineGraph * self, PyObject * args)
{
 // Create the vertex arrays...
  npy_intp dims = self->vertex_count;
  PyObject * vertex_x = PyArray_SimpleNew(1, &dims, NPY_FLOAT32);
  PyObject * vertex_y = PyArray_SimpleNew(1, &dims, NPY_FLOAT32);
  PyObject * vertex_u = PyArray_SimpleNew(1, &dims, NPY_FLOAT32);
  PyObject * vertex_v = PyArray_SimpleNew(1, &dims, NPY_FLOAT32);
  PyObject * vertex_w = PyArray_SimpleNew(1, &dims, NPY_FLOAT32);
  PyObject * vertex_radius  = PyArray_SimpleNew(1, &dims, NPY_FLOAT32);
  PyObject * vertex_density = PyArray_SimpleNew(1, &dims, NPY_FLOAT32);
  PyObject * vertex_weight = PyArray_SimpleNew(1, &dims, NPY_FLOAT32);
  
  int i;
  for (i=0; i<self->vertex_count; i++)
  {
   *(float*)PyArray_GETPTR1(vertex_x, i) = self->vertex[i].x;
   *(float*)PyArray_GETPTR1(vertex_y, i) = self->vertex[i].y;
   *(float*)PyArray_GETPTR1(vertex_u, i) = self->vertex[i].u;
   *(float*)PyArray_GETPTR1(vertex_v, i) = self->vertex[i].v;
   *(float*)PyArray_GETPTR1(vertex_w, i) = self->vertex[i].w;
   *(float*)PyArray_GETPTR1(vertex_radius, i)  = self->vertex[i].radius;
   *(float*)PyArray_GETPTR1(vertex_density, i) = self->vertex[i].density;
   *(float*)PyArray_GETPTR1(vertex_weight, i) = self->vertex[i].weight;
  }
 
 // Create the edge array...
  dims = self->edge_count;
  PyObject * edge_from = PyArray_SimpleNew(1, &dims, NPY_INT32);
  PyObject * edge_to   = PyArray_SimpleNew(1, &dims, NPY_INT32);
  
  for (i=0; i<self->edge_count; i++)
  {
   *(int*)PyArray_GETPTR1(edge_from, i) = self->edge[i].neg.dest - self->vertex;
   *(int*)PyArray_GETPTR1(edge_to, i)   = self->edge[i].pos.dest - self->vertex;
  }
 
 // Pass over the edges to count the various meta information...
  int split_count = 0;
  int tag_count = 0;
  int link_count = 0;
  int link_tag_count = 0;
  
  for (i=0; i<self->edge_count; i++)
  {
   Edge * targ = &self->edge[i];
   
   SplitTag * st = targ->dummy.next;
   while(st!=&targ->dummy)
   {
    if (st->other==NULL)
    {
     if (st->tag==NULL) split_count += 1;
                   else tag_count += 1;
    }
    else
    {
     if (st < st->other) // Count each link once
     {
      if (st->tag==NULL) link_count += 1;
                    else link_tag_count += 1;
     }
    }
    
    st = st->next; 
   }
  }
 
 // Create the arrays for the meta data...
  dims = split_count;
  PyObject * split_edge = PyArray_SimpleNew(1, &dims, NPY_INT32);
  PyObject * split_t    = PyArray_SimpleNew(1, &dims, NPY_FLOAT32);
  
  dims = tag_count;
  PyObject * tag_edge = PyArray_SimpleNew(1, &dims, NPY_INT32);
  PyObject * tag_t    = PyArray_SimpleNew(1, &dims, NPY_FLOAT32);
  PyObject * tag_text = PyArray_SimpleNew(1, &dims, NPY_OBJECT);
  
  dims = link_count;
  PyObject * link_from_edge = PyArray_SimpleNew(1, &dims, NPY_INT32);
  PyObject * link_from_t    = PyArray_SimpleNew(1, &dims, NPY_FLOAT32);
  PyObject * link_to_edge   = PyArray_SimpleNew(1, &dims, NPY_INT32);
  PyObject * link_to_t      = PyArray_SimpleNew(1, &dims, NPY_FLOAT32);
  
  dims = link_tag_count;
  PyObject * link_tag_from_edge = PyArray_SimpleNew(1, &dims, NPY_INT32);
  PyObject * link_tag_from_t    = PyArray_SimpleNew(1, &dims, NPY_FLOAT32);
  PyObject * link_tag_to_edge   = PyArray_SimpleNew(1, &dims, NPY_INT32);
  PyObject * link_tag_to_t      = PyArray_SimpleNew(1, &dims, NPY_FLOAT32);
  PyObject * link_tag_text      = PyArray_SimpleNew(1, &dims, NPY_OBJECT);
  
 // Pass over the edges to fill in the meta data...
  split_count = 0;
  tag_count = 0;
  link_count = 0;
  link_tag_count = 0;
  
  for (i=0; i<self->edge_count; i++)
  {
   Edge * targ = &self->edge[i];
   
   SplitTag * st = targ->dummy.next;
   while(st!=&targ->dummy)
   {
    if (st->other==NULL)
    {
     if (st->tag==NULL)
     {
      *(int*)PyArray_GETPTR1(split_edge, split_count) = i;
      *(float*)PyArray_GETPTR1(split_t, split_count) = st->t;
      
      split_count += 1;
     }
     else
     {
      *(int*)PyArray_GETPTR1(tag_edge, tag_count) = i;
      *(float*)PyArray_GETPTR1(tag_t, tag_count) = st->t;
      *(PyObject**)PyArray_GETPTR1(tag_text, tag_count) = PyString_FromString(st->tag);
      
      tag_count += 1;
     }
    }
    else
    {
     if (st < st->other) // Process each link once
     {
      if (st->tag==NULL)
      {
       *(int*)PyArray_GETPTR1(link_from_edge, link_count) = i;
       *(float*)PyArray_GETPTR1(link_from_t, link_count) = st->t;
       *(int*)PyArray_GETPTR1(link_to_edge, link_count) = st->other->loc - self->edge;
       *(float*)PyArray_GETPTR1(link_to_t, link_count) = st->other->t;
      
       link_count += 1;      
      }
      else
      {  
       *(int*)PyArray_GETPTR1(link_tag_from_edge, link_tag_count) = i;
       *(float*)PyArray_GETPTR1(link_tag_from_t, link_tag_count) = st->t;
       *(int*)PyArray_GETPTR1(link_tag_to_edge, link_tag_count) = st->other->loc - self->edge;
       *(float*)PyArray_GETPTR1(link_tag_to_t, link_tag_count) = st->other->t;
       *(PyObject**)PyArray_GETPTR1(link_tag_text, link_tag_count) = PyString_FromString(st->tag);
      
       link_tag_count += 1;
      }
     }
    }
    
    st = st->next; 
   }
  }
  
 
 // Return the resulting dictionary - fiddly...
  PyObject * vertex_dict = Py_BuildValue("{s:N,s:N,s:N,s:N,s:N,s:N,s:N,s:N}", "x", vertex_x, "y", vertex_y, "u", vertex_u, "v", vertex_v, "w", vertex_w, "radius", vertex_radius, "density", vertex_density, "weight", vertex_weight);
  
  PyObject * edge_dict = Py_BuildValue("{s:N,s:N}", "from", edge_from, "to", edge_to);
  
  PyObject * split_dict = Py_BuildValue("{s:N,s:N}", "edge", split_edge, "t", split_t);
  
  PyObject * tag_dict = Py_BuildValue("{s:N,s:N,s:N}", "edge", tag_edge, "t", tag_t, "text", tag_text);
  
  PyObject * link_dict = Py_BuildValue("{s:N,s:N,s:N,s:N}", "from_edge", link_from_edge, "from_t", link_from_t, "to_edge", link_to_edge, "to_t", link_to_t);
  
  PyObject * link_tag_dict = Py_BuildValue("{s:N,s:N,s:N,s:N,s:N}", "from_edge", link_tag_from_edge, "from_t", link_tag_from_t, "to_edge", link_tag_to_edge, "to_t", link_tag_to_t, "text", link_tag_text);
 
  PyObject * element_dict = Py_BuildValue("{s:N,s:N,s:N,s:N,s:N,s:N}", "vertex", vertex_dict, "edge", edge_dict, "split", split_dict, "tag", tag_dict, "link", link_dict, "link_tag", link_tag_dict);
 
  return Py_BuildValue("{s:N}", "element", element_dict);
}



// After calling this the LineGraph segmentation will be valid; if it is already valid it will do nothing...
void LineGraph_segment(LineGraph * this)
{
 // If the segmentation is valid do nothing...
  if (this->segments>=0) return;
  
 // Assign everything to its own segment, and make sure the SplitTags for each edge are sorted...
  int i;
  int next_segment = 0;
  for (i=0; i<this->edge_count; i++)
  {
   this->edge[i].segment = next_segment; 
   next_segment += 1;
   
   int sorted = 1;
   float prev = -1.0;
   
   SplitTag * st = this->edge[i].dummy.next;
   while (st!=&this->edge[i].dummy)
   {
    if ((st->tag==NULL)&&(st->other==NULL))
    {
     st->segment = next_segment;
     next_segment += 1;
    }
    else st->segment = -1;
    
    if (prev > st->t) sorted = 0;
    prev = st->t;
    
    st = st->next; 
   }
   
   if (sorted==0)
   {
    // Need to sort the SplitTag's - this is quite rare (edges with more than 1 SplitTag are very unusual) and always for small numbers, so don't care too much for efficiency, hence using insertion sort on a linked list doesn't bother me...
    
    // Create singularly linked list and reset dummy, setting its t to be high...
     SplitTag * eat_me = this->edge[i].dummy.next;
     this->edge[i].dummy.prev->next = NULL;
     
     this->edge[i].dummy.next = &this->edge[i].dummy;
     this->edge[i].dummy.prev = &this->edge[i].dummy;
     this->edge[i].dummy.t = 2.0;
    
    // Consume the list, doing an insertion sort on each...
     while (eat_me!=NULL)
     {
      SplitTag * st = &this->edge[i].dummy;
      while (st->next->t < eat_me->t) st = st->next;
      
      SplitTag * d = eat_me;
      eat_me = eat_me->next;
      
      d->prev = st;
      d->next = st->next;
      
      d->next->prev = d;
      d->prev->next = d;
     }
   }
  }
  
 // Go through and keep merging until a merge pass does nothing...
  int cont = 1;
  int prev_cont;
  int dir = -1;
  do
  {
   prev_cont = cont;
   cont = 0;
   dir *= -1;
   
   i = (dir>0) ? 0 : (this->edge_count-1);
   while(1)
   {
    // Get the edge...
     Edge * targ = &this->edge[i];
    
    // Do the negative side merges...
     int min_seg = targ->segment;
     
     HalfEdge * he = targ->neg.dest->incident;
     do
     {
      Edge * he_edge = HalfToEdge(he);
      
      int s = he_edge->segment;
      if (targ->neg.dest==he_edge->pos.dest) 
      {
       SplitTag * st = he_edge->dummy.prev;
       while (st!=&he_edge->dummy)
       {
        if ((st->tag==NULL)&&(st->other==NULL))
        {
         s = st->segment;
         break;
        }
        st = st->prev; 
       }
      }
      if (s<min_seg) min_seg = s;
      
      he = he->reverse->next;
     }
     while (he!=targ->neg.dest->incident);
     
     if (min_seg!=targ->segment)
     {
      targ->segment = min_seg;
      cont = 1;
     }
    
    // Loop the SplitTag's, performing merges for every segment on it...
     SplitTag * prev_split = NULL;
     
     SplitTag * st = targ->dummy.next;
     while (st!=&targ->dummy)
     {
      if (st->tag==NULL)
      {
       if (st->other!=NULL)
       {
        // Its a bare link - send information along it...
         int min_seg = (prev_split==NULL) ? targ->segment : prev_split->segment;
         
         SplitTag * ohe = st->other->prev;
         while ((ohe!=&st->other->loc->dummy)&&((ohe->tag!=NULL)||(ohe->other!=NULL)))
         {
          ohe = ohe->prev; 
         }
         
         if (ohe==&st->other->loc->dummy)
         {
          // Linked to the other edge...
           if (st->other->loc->segment<min_seg) min_seg = st->other->loc->segment;
         }
         else
         {
          if (ohe->segment<min_seg) min_seg = ohe->segment;
         }
         
         if (prev_split==NULL)
         {
          if (min_seg!=targ->segment)
          {
           targ->segment = min_seg;
           cont = 1;
          }
         }
         else
         {
          if (min_seg!=prev_split->segment)
          {
           prev_split->segment = min_seg;
           cont = 1;
          }
         }
       }
       else
       {
        // Its a split - record so we can mess with it...
         prev_split = st;
       }
      }
      st = st->next; 
      
     }
    
    // Do the positive side merges...
     min_seg = (prev_split==NULL) ? targ->segment : prev_split->segment;
     
     he = targ->pos.dest->incident;
     do
     {
      Edge * he_edge = HalfToEdge(he);
      
      int s = he_edge->segment;
      if (targ->pos.dest==he_edge->pos.dest)
      {
       SplitTag * st = he_edge->dummy.prev;
       while (st!=&he_edge->dummy)
       {
        if ((st->tag==NULL)&&(st->other==NULL))
        {
         s = st->segment;
         break;
        }
        st = st->prev; 
       }
      }
      if (s<min_seg) min_seg = s;
      
      he = he->reverse->next;
     }
     while (he!=targ->pos.dest->incident);
     
     if (prev_split==NULL)
     {
      if (min_seg!=targ->segment)
      {
       targ->segment = min_seg;
       cont = 1;
      }
     }
     else
     {
      if (min_seg!=prev_split->segment)
      {
       prev_split->segment = min_seg;
       cont = 1;
      }
     }
   
    // Move to next...
     i += dir;
     if ((i<0)||(i>=this->edge_count)) break;
   }
  }
  while ((prev_cont!=0)&&(cont!=0));
  
  
 // Go find all the segments, assigning a new sequential number to each...
  int * new_segment = (int*)malloc(next_segment * sizeof(int));
  for (i=0; i<next_segment; i++) new_segment[i] = -1;

  next_segment = 0;
  for (i=0; i<this->edge_count; i++)
  {
   if (new_segment[this->edge[i].segment]==-1)
   {
    new_segment[this->edge[i].segment] = next_segment;
    next_segment += 1;
   }

   SplitTag * st = this->edge[i].dummy.next;
   while (st!=&this->edge[i].dummy)
   {
    if ((st->tag==NULL)&&(st->other==NULL))
    {
     if (new_segment[st->segment]==-1)
     {
      new_segment[st->segment] = next_segment;
      next_segment += 1;
     }
    }
    
    st = st->next; 
   }
  }
  
 // Make the transformation to packed segment numbers...
  for (i=0; i<this->edge_count; i++)
  {
   this->edge[i].segment = new_segment[this->edge[i].segment];

   SplitTag * st = this->edge[i].dummy.next;
   while (st!=&this->edge[i].dummy)
   {
    if ((st->tag==NULL)&&(st->other==NULL))
    {
     st->segment = new_segment[st->segment];
    }
    
    st = st->next; 
   }
  }
  
  free(new_segment);
  
 // Mark the segmentation as being valid, in a variable that encodes how many we have...
  this->segments = next_segment;
}


static PyObject * LineGraph_segment_py(LineGraph * self, PyObject * args)
{
 // Do the segmentation...
  LineGraph_segment(self);
 
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}



static PyObject * LineGraph_get_bounds_py(LineGraph * self, PyObject * args)
{
 // Parse the parameters...
  int segment = -1;
  if (!PyArg_ParseTuple(args, "|i", &segment)) return NULL;
  
 // If its the simple scenario, of the bounds of everything, we already have it compliments of the spatial indexing structure - return it...
  if (segment==-1)
  {
   if (self->root!=NULL) return Py_BuildValue("(f,f,f,f)", self->root->min_x, self->root->max_x, self->root->min_y, self->root->max_y);
   else
   {
    Py_INCREF(Py_None);
    return Py_None;
   }
  }
  
 // Check we have a valid segmentation...
  if (self->segments<0)
  {
   PyErr_SetString(PyExc_RuntimeError, "Request for bounds of segment when segmentation is invalid");
   return NULL;
  }
 
 // More complicated scenario - we just have to loop all edges and note the ranges seen as we go...
  float min_x = 0.0, max_x = 0.0, min_y = 0.0, max_y = 0.0;
  char valid = 0;
  
  int i;
  for (i=0; i<self->edge_count; i++)
  {
   Edge * targ = &self->edge[i];
   int seg = targ->segment;
   
   // Start of edge...
    if (seg==segment)
    {
     float low_x = targ->neg.dest->x - targ->neg.dest->radius;
     float high_x = targ->neg.dest->x + targ->neg.dest->radius;
     float low_y = targ->neg.dest->y - targ->neg.dest->radius;
     float high_y = targ->neg.dest->y + targ->neg.dest->radius;
      
     if (valid!=0)
     {
      if (low_x<min_x) min_x = low_x;
      if (high_x>max_x) max_x = high_x;
      if (low_y<min_y) min_y = low_y;
      if (high_y>max_y) max_y = high_y;
     }
     else
     {
      min_x = low_x;
      max_x = high_x;
      min_y = low_y;
      max_y = high_y;
      valid = 1; 
     }
    }
    
   // Splits, with segment transitions...
    SplitTag * st = targ->dummy.next;
    while (st!=&targ->dummy)
    {
     if ((st->tag==NULL)&&(st->other==NULL))
     {
      // Check if this defines a point in the segment, in which case update the bounds with it (Only bother if it is a transition)...
       if ((seg!=st->segment)&&((seg==segment)||(st->segment==segment)))
       {
        float x = targ->neg.dest->x * (1.0-st->t) + targ->pos.dest->x * st->t;
        float y = targ->neg.dest->y * (1.0-st->t) + targ->pos.dest->y * st->t;
        float radius = targ->neg.dest->radius * (1.0-st->t) + targ->pos.dest->radius * st->t;
         
        float low_x = x - radius;
        float high_x = x + radius;
        float low_y = y - radius;
        float high_y = y + radius;
        
        if (valid!=0)
        {
         if (low_x<min_x) min_x = low_x;
         if (high_x>max_x) max_x = high_x;
         if (low_y<min_y) min_y = low_y;
         if (high_y>max_y) max_y = high_y;
        }
        else
        {
         min_x = low_x;
         max_x = high_x;
         min_y = low_y;
         max_y = high_y;
         valid = 1; 
        }
       }
      
      // Move to next segment...
       seg = st->segment; 
     }
      
     st = st->next;
    }
   
   // End of edge...
    if (seg==segment)
    {
     float low_x = targ->pos.dest->x - targ->pos.dest->radius;
     float high_x = targ->pos.dest->x + targ->pos.dest->radius;
     float low_y = targ->pos.dest->y - targ->pos.dest->radius;
     float high_y = targ->pos.dest->y + targ->pos.dest->radius;
      
     if (valid!=0)
     {
      if (low_x<min_x) min_x = low_x;
      if (high_x>max_x) max_x = high_x;
      if (low_y<min_y) min_y = low_y;
      if (high_y>max_y) max_y = high_y;
     }
     else
     {
      min_x = low_x;
      max_x = high_x;
      min_y = low_y;
      max_y = high_y;
      valid = 1; 
     }
    }
  }
  
  if (valid!=0) return Py_BuildValue("(f,f,f,f)", min_x, max_x, min_y, max_y);
  else
  {
   Py_INCREF(Py_None);
   return Py_None;
  }
}



static PyObject * LineGraph_get_vertex_py(LineGraph * self, PyObject * args)
{
 // Extract the parameters
  int i;
  if (!PyArg_ParseTuple(args, "i", &i)) return NULL;
  if ((i<0)||(i>=self->vertex_count))
  {
   PyErr_SetString(PyExc_IndexError, "Index out of range.");
   return NULL;
  }

 // Fetch the item...
  Vertex * targ = &self->vertex[i];
  
 // Handle the source...
  PyObject * source = Py_None;
  if (targ->source<0) Py_INCREF(Py_None);
  else source = PyInt_FromLong(targ->source);
  
 // Return it...
  return Py_BuildValue("(f,f,f,f,f,f,f,f,N)", targ->x, targ->y, targ->u, targ->v, targ->w, targ->radius, targ->density, targ->weight, source);
}



static PyObject * LineGraph_get_edge_py(LineGraph * self, PyObject * args)
{
 // Extract the parameters
  int i;
  if (!PyArg_ParseTuple(args, "i", &i)) return NULL;
  if ((i<0)||(i>=self->edge_count))
  {
   PyErr_SetString(PyExc_IndexError, "Index out of range.");
   return NULL;
  }

 // Fetch the edge...
  Edge * targ = &self->edge[i];

 // Build the return item - gets complicated because of the tags/splits...
  // Count the tags/splits...
   int count = 0;
   SplitTag * st = targ->dummy.next;
   while (st!=&targ->dummy)
   {
    count += 1;
    st = st->next;
   }
   
  // Create a list, fill it...
   PyObject * list = PyList_New(count);
   count = 0;

   st = targ->dummy.next;
   while (st!=&targ->dummy)
   {
    PyObject * pair = Py_None;
    if (st->other==NULL) Py_INCREF(Py_None);
    else pair = Py_BuildValue("(i,f)", st->other->loc - self->edge, st->other->t);
      
    PyList_SetItem(list, count, Py_BuildValue("(f,s,N)", st->t, st->tag, pair));
    
    count += 1;
    st = st->next;
   }
   
  // Handle the source...
   PyObject * source = Py_None;
   if (targ->source<0) Py_INCREF(Py_None);
   else source = PyInt_FromLong(targ->source);

  // Create the tuple of information and return it...
   int pv = targ->pos.dest - self->vertex;
   int nv = targ->neg.dest - self->vertex;
   return Py_BuildValue("(i,i,N,N)", nv, pv, list, source);
}



static PyObject * LineGraph_get_point_py(LineGraph * self, PyObject * args)
{
 // Extract the parameters
  int i;
  float t;
  if (!PyArg_ParseTuple(args, "if", &i, &t)) return NULL;
  if ((i<0)||(i>=self->edge_count))
  {
   PyErr_SetString(PyExc_IndexError, "Index out of range.");
   return NULL;
  }
  
 // Calculate the return values - simple linear interpolation...
  Edge * targ = &self->edge[i];
  float mt = 1.0 - t;
 
  float x = targ->neg.dest->x * mt + targ->pos.dest->x * t;
  float y = targ->neg.dest->y * mt + targ->pos.dest->y * t;
  float u = targ->neg.dest->u * mt + targ->pos.dest->u * t;
  float v = targ->neg.dest->v * mt + targ->pos.dest->v * t;
  float w = targ->neg.dest->w * mt + targ->pos.dest->w * t;
  float radius  = targ->neg.dest->radius  * mt + targ->pos.dest->radius  * t;
  float density = targ->neg.dest->density * mt + targ->pos.dest->density * t;
  float weight = targ->neg.dest->weight * mt + targ->pos.dest->weight * t;
 
 // Return the tuple...
  return Py_BuildValue("(f,f,f,f,f,f,f,f)", x, y, u, v, w, radius, density, weight);
}



static PyObject * LineGraph_get_segment_py(LineGraph * self, PyObject * args)
{
 // Extract the parameters
  int i;
  float t;
  if (!PyArg_ParseTuple(args, "if", &i, &t)) return NULL;
  if ((i<0)||(i>=self->edge_count))
  {
   PyErr_SetString(PyExc_IndexError, "Index out of range.");
   return NULL;
  }
  
 // Make sure we have a valid segmentation...
  LineGraph_segment(self);
  
 // Go through the SplitTag for the edge and find the relevent segment number...
  Edge * targ = &self->edge[i];
  
  int segment = targ->segment;
  SplitTag * st = targ->dummy.next;
  while (st!=&targ->dummy)
  {
   if (st->t>t) break;
   if ((st->tag==NULL)&&(st->other==NULL)) segment = st->segment; 
   st = st->next; 
  }
  
 // Return it...
  return Py_BuildValue("i", segment);
}


static PyObject * LineGraph_get_segs_py(LineGraph * self, PyObject * args)
{
 // Create the numpy array...
  npy_intp length = self->vertex_count;
  PyObject * ret = PyArray_SimpleNew(1, &length, NPY_INT32);
  
 // Make sure we have a valid segmentation...
  LineGraph_segment(self);
  
 // Fill it in...
  int i;
  for (i=0; i<length; i++)
  {
   int segment = -1;
   if (self->vertex[i].incident!=NULL)
   {
    Edge * edge = HalfToEdge(self->vertex[i].incident);
    segment = edge->segment;
    if (self->vertex[i].incident!=&edge->neg)
    {
     SplitTag * st = edge->dummy.next;
     while (st!=&edge->dummy)
     {
      if ((st->tag==NULL)&&(st->other==NULL)) segment = st->segment; 
      st = st->next; 
     }
    }
   }
   
   *(int*)PyArray_GETPTR1(ret, i) = segment;
  }
  
 // Return the numpy array...
  return ret;
}



static PyObject * LineGraph_vertex_to_edges_py(LineGraph * self, PyObject * args)
{
 // Extract the parameter...
  int vert;
  if (!PyArg_ParseTuple(args, "i", &vert)) return NULL;
  if ((vert<0)||(vert>=self->vertex_count))
  {
   PyErr_SetString(PyExc_IndexError, "Index out of range.");
   return NULL;
  }
  
 // Create the list and loop the vertices incident edges...
  PyObject * ret = PyList_New(0);
  
  HalfEdge * targ = self->vertex[vert].incident;
  do
  {
   // Add this edge...
    Edge * e = HalfToEdge(targ);
    
    PyObject * tup = Py_BuildValue("(i,O)", e - self->edge, (&e->neg==targ) ? Py_True : Py_False);
    PyList_Append(ret, tup);
    Py_DECREF(tup);
    
   // Move to next...
    targ = targ->reverse->next;
    
  } while (targ!=self->vertex[vert].incident);
 
 // Return the list...
  return ret;
}



void LineGraph_add_split_tag(LineGraph * this, Edge * e, float t, char * tag)
{
 SplitTag * nst = (SplitTag*)malloc(sizeof(SplitTag));
 
 nst->loc = e;
 
 nst->next = &e->dummy;
 nst->prev = e->dummy.prev;
 
 nst->next->prev = nst;
 nst->prev->next = nst;
 
 nst->tag = (tag!=NULL) ? strdup(tag) : NULL;
 nst->t = t;
 nst->other = NULL;
 
 this->segments = -1;
}

void LineGraph_add_link(LineGraph * this, Edge * a, float ta, Edge * b, float tb, char * tag)
{
 if (a==b) return; // I am not coding for this kind of craziness.
 
 SplitTag * sta = (SplitTag*)malloc(sizeof(SplitTag));
 SplitTag * stb = (SplitTag*)malloc(sizeof(SplitTag));
 
 sta->loc = a;
 stb->loc = b;
 
 sta->next = &a->dummy;
 sta->prev = a->dummy.prev;
 sta->next->prev = sta;
 sta->prev->next = sta;
 
 stb->next = &b->dummy;
 stb->prev = b->dummy.prev;
 stb->next->prev = stb;
 stb->prev->next = stb;
 
 sta->tag = (tag!=NULL) ? strdup(tag) : NULL;
 stb->tag = sta->tag;
 
 sta->t = ta;
 stb->t = tb;
 
 sta->other = stb;
 stb->other = sta;
 
 this->segments = -1;
}

// Remove the split/tag/link that is closest to the given given t. If edge has none it does nothing.
void LineGraph_rem(LineGraph * this, Edge * e, float t)
{
 // Search for the closest...
  float best_d = 1e100;
  SplitTag * best = NULL;
 
  SplitTag * targ = e->dummy.next;
  while (targ!=&e->dummy)
  {
   float d = fabs(targ->t - t);
   if (d<best_d)
   {
    best_d = d;
    best = targ;
   }
   targ = targ->next;
  }
 
 // Terminate it...
  if (best!=NULL)
  {
   this->segments = -1;
   SplitTag_free(best);
  }
}


static PyObject * LineGraph_add_split_tag_py(LineGraph * self, PyObject * args)
{
 // Extract the parameters
  int i;
  float t;
  char * tag = NULL;
  
  if (!PyArg_ParseTuple(args, "if|z", &i, &t, &tag)) return NULL;
  if ((i<0)||(i>=self->edge_count))
  {
   PyErr_SetString(PyExc_IndexError, "Index out of range.");
   return NULL;
  }
  
 // Get the edge object...
  Edge * e = &self->edge[i];
  
 // Add the entity...
  LineGraph_add_split_tag(self, e, t, tag);
  
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject * LineGraph_add_link_py(LineGraph * self, PyObject * args)
{
 // Extract the parameters
  int ai, bi;
  float at, bt;
  char * tag = NULL;
  
  if (!PyArg_ParseTuple(args, "ifif|z", &ai, &at, &bi, &bt, &tag)) return NULL;
  
  if ((ai<0)||(ai>=self->edge_count))
  {
   PyErr_SetString(PyExc_IndexError, "Index out of range - first edge.");
   return NULL;
  }
  if ((bi<0)||(bi>=self->edge_count))
  {
   PyErr_SetString(PyExc_IndexError, "Index out of range - second edge.");
   return NULL;
  }
  
 // Get the edge objects...
  Edge * a = &self->edge[ai];
  Edge * b = &self->edge[bi];
  
 // Add the entity...
  LineGraph_add_link(self, a, at, b, bt, tag);
  
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject * LineGraph_rem_py(LineGraph * self, PyObject * args)
{
 // Extract the parameters
  int i;
  float t;
  
  if (!PyArg_ParseTuple(args, "if", &i, &t)) return NULL;
  if ((i<0)||(i>=self->edge_count))
  {
   PyErr_SetString(PyExc_IndexError, "Index out of range.");
   return NULL;
  }
  
 // Get the edge object...
  Edge * e = &self->edge[i];
  
 // Release the assassin...
  LineGraph_rem(self, e, t);
  
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}



static PyObject * LineGraph_vertex_stats_py(LineGraph * self, PyObject * args)
{
 // Loop the vertices and collate the counts...
  int ret[4] = {0, 0, 0, 0};
  
  int i;
  for (i=0; i<self->vertex_count; i++)
  {
   Vertex * targ = &self->vertex[i];
   
   int count = 0;
   HalfEdge * he = targ->incident;
   if (he!=NULL)
   {
    do
    {
     count += 1;
     he = he->reverse->next;
    }
    while (he!=targ->incident);
   }
   
   if (count<4) ret[count] += 1;
   else ret[3] += 1;
  }
  
 // Return the relevant tuple...
  return Py_BuildValue("(i,i,i,i)", ret[0], ret[1], ret[2], ret[3]);
}



static PyObject * LineGraph_chains_py(LineGraph * self, PyObject * args)
{
 PyObject * ret = PyList_New(0);
 
 // Storage so we can detect loops...
  char * used = (char*)malloc(self->vertex_count * sizeof(char));
  memset(used, 0, self->vertex_count * sizeof(char));
 
 // Normal edge segments...
  int i;
  for (i=0; i<self->vertex_count; i++)
  {
   Vertex * targ = &self->vertex[i];
  
   // Only if the vertex doesn't have two edges do we consider it...
    if ((targ->incident!=NULL)&&((targ->incident->reverse->next==targ->incident)||(targ->incident->reverse->next->reverse->next!=targ->incident)))
    {
     used[i] = 1;
     
     // Its an interesting vertex - it doesn't have two edges leaving it, therefore consider it as the start of a chain - iterate all edges that leave it, investigating each...
      HalfEdge * targ_he = targ->incident;
      do
      {
       // Create the chain...
        PyObject * chain = PyList_New(1);
        PyList_SetItem(chain, 0, PyInt_FromLong(i));
       
       // Fill it until we hit a branch/tail...
        HalfEdge * he = targ_he;
        while (1)
        {
         // Store the destination vertex index...
          PyObject * num = PyInt_FromLong(he->dest - self->vertex);
          PyList_Append(chain, num);
          Py_DECREF(num);
          
          used[he->dest - self->vertex] = 1;
        
         // Check if the destination vertex is complicated, in which case end the chain...
          if (he->next==he->reverse) break;
          if (he->next->reverse->next!=he->reverse) break;
        
         // Move to the next...
          he = he->next;
        }
       
       // Store the chain if the end has a larger pointer than the start - code creates every chain twice, so this avoids repetition...
        if (he->dest>targ)
        {
         PyList_Append(ret, chain);
        }
        Py_DECREF(chain);
       
       // Move to the next edge that is leaving this interesting vertex...
        targ_he = targ_he->reverse->next;
      }
      while (targ_he!=targ->incident);
    }
  }
 
 // Loops...
  for (i=0; i<self->vertex_count; i++)
  {
   if ((used[i]==0)&&(self->vertex[i].incident!=NULL))
   {
    used[i] = 1;
    // Unused vertex that has edges - only possible explanation is its part of a loop - extract it...
     // Create the chain...
      PyObject * chain = PyList_New(1);
      PyList_SetItem(chain, 0, PyInt_FromLong(i));
     
     // Extract its members...
      HalfEdge * he = self->vertex[i].incident;
      while (1)
      {
       // Store the destination vertex index...
        PyObject * num = PyInt_FromLong(he->dest - self->vertex);
        PyList_Append(chain, num);
        Py_DECREF(num);
          
        used[he->dest - self->vertex] = 1;
        
       // Check if we have followed the loop...
        if (he->dest==(self->vertex+i)) break;
        
       // To next...
        he = he->next; 
      }
      
     // Store it...
      PyList_Append(ret, chain);
      Py_DECREF(chain);
   }
  }
 
 // Clean up and return...
  free(used);
  return ret;
}



static PyObject * LineGraph_get_tails_py(LineGraph * self, PyObject * args)
{
 PyObject * ret = PyList_New(0);
 
 int i;
 for (i=0; i<self->vertex_count; i++)
 {
  Vertex * targ = &self->vertex[i];
  if ((targ->incident!=NULL)&&(targ->incident->reverse->next==targ->incident))
  {
   PyObject * num = PyInt_FromLong(i);
   PyList_Append(ret, num);
   Py_DECREF(num);
  }
 }
 
 return ret;
}



static PyObject * LineGraph_get_tags_py(LineGraph * self, PyObject * args)
{
 // Fetch the parameters...
  int seg = -1;
  if (!PyArg_ParseTuple(args, "|i", &seg)) return NULL;
  
 // Throw an error if the segmentation is invalid...
  if ((seg!=-1)&&(self->segments<0))
  {
   PyErr_SetString(PyExc_RuntimeError, "Request for segment when segmentation is invalid");
   return NULL;
  }

 // First count how many tags that we care about exist in the graph...
  int i;
  int count = 0;
  for (i=0; i<self->edge_count; i++)
  {
   Edge * te = &self->edge[i];
   
   int segment = te->segment;
   SplitTag * se = te->dummy.next;
   while (se!=&te->dummy)
   {
    // Check if this is a tag we intend to record or not...
     if ((se->tag!=NULL)&&((se->other==NULL)||(se<se->other)))
     {
      if ((seg==-1)||(segment==seg)) count += 1;
     }
     else
     {
      // Handle moving to the next segment...
       if (se->other==NULL) segment = se->segment;
     }
     
    se = se->next; 
   }
  }
  
 // Create the return value - a list...
  PyObject * ret = PyList_New(count);
  
 // Second pass to collect them all into the list...
  count = 0;
  for (i=0; i<self->edge_count; i++)
  {
   Edge * te = &self->edge[i];
   
   int segment = te->segment;
   SplitTag * se = te->dummy.next;
   while (se!=&te->dummy)
   {
    // Check if this is a tag we intend to record or not...
     if ((se->tag!=NULL)&&((se->other==NULL)||(se<se->other)))
     {
      if ((seg==-1)||(segment==seg))
      {
       // Record it - different tuple depending if its a tag or link...
        if (se->other!=NULL)
        {
         // Link with tag... 
          PyList_SetItem(ret, count, Py_BuildValue("(s,i,f,i,f)", se->tag, i, se->t, se->other->loc - self->edge, se->other->t));
        }
        else
        {
         // Tag...
          PyList_SetItem(ret, count, Py_BuildValue("(s,i,f)", se->tag, i, se->t));
        }
       count += 1; 
      }
     }
     else
     {
      // Handle moving to the next segment...
       if (se->other==NULL) segment = se->segment;
     }
     
    se = se->next; 
   }
  }
  
 // Return...
  return ret;
}



static PyObject * LineGraph_get_splits_py(LineGraph * self, PyObject * args)
{
 // Fetch the parameters...
  int seg = -1;
  if (!PyArg_ParseTuple(args, "|i", &seg)) return NULL;
  
 // Throw an error if the segmentation is invalid...
  if ((seg!=-1)&&(self->segments<0))
  {
   PyErr_SetString(PyExc_RuntimeError, "Request for segment when segmentation is invalid");
   return NULL;
  }

 // First count how many tags that we care about exist in the graph...
  int i;
  int count = 0;
  for (i=0; i<self->edge_count; i++)
  {
   Edge * targ = &self->edge[i];
   
   int segment = targ->segment;
   SplitTag * st = targ->dummy.next;
   while (st!=&targ->dummy)
   {
    // Check if this is a tag we intend to record or not...
     if ((st->tag==NULL)&&((st->other==NULL)||(st<st->other)))
     {
      if ((seg==-1)||(segment==seg)||((st->other==NULL)&&(st->segment==seg))) count += 1;
      if (st->other==NULL) segment = st->segment;
     }
     
    st = st->next; 
   }
  }
  
 // Create the return value - a list...
  PyObject * ret = PyList_New(count);
  
 // Second pass to collect them all into the list...
  count = 0;
  for (i=0; i<self->edge_count; i++)
  {
   Edge * targ = &self->edge[i];
   
   int segment = targ->segment;
   SplitTag * st = targ->dummy.next;
   while (st!=&targ->dummy)
   {
    // Check if this is a tag we intend to record or not...
     if ((st->tag==NULL)&&((st->other==NULL)||(st<st->other)))
     {
      if ((seg==-1)||(segment==seg)||((st->other==NULL)&&(st->segment==seg)))
      {
       // We need to record this tag - different code if its a split or a link... 
        if (st->other==NULL)
        {
         // Split... 
          float dir = 0.0;
          if (segment!=st->segment)
          {
           if (segment==seg) dir = -1.0;
                        else dir = 1.0;
          }
          
          PyList_SetItem(ret, count, Py_BuildValue("(i,f,f)", i, st->t, dir));
        }
        else
        {
         // Link...
          PyList_SetItem(ret, count, Py_BuildValue("(i,f,i,f)", i, st->t, st->other->loc - self->edge, st->other->t));
        }
        
        count += 1;
      }
      
      if (st->other==NULL) segment = st->segment;
     }
     
    st = st->next; 
   }
  }
  
 // Return...
  return ret;
}



// Sticks into out and out_dist the nearest tag after arriving at a vertex via origin, having already traveled dist - out must start as NULL so it can be replaced with the corrcet value, which may be updated repeatedly during the call...
void HalfEdge_NearTag(HalfEdge * origin, float dist, SplitTag ** out, float * out_dist, float limit)
{
 if (dist>limit) return;
 
 // Eat as much chain as we dare, breaking if we find a tag to factor in...
  while (1)
  {
   // Check we are on a chain - break if not...
    if (origin->next->reverse==origin) return; // Its a tail!
    if (origin->next->reverse->next!=origin->reverse) break; // Not a chain.
    
   // Whatever happens we are going to need its length...
    float dx = origin->next->dest->x - origin->dest->x;
    float dy = origin->next->dest->y - origin->dest->y;
    float len = sqrt(dx*dx + dy*dy);
   
   // Check the next edge in the chain for relevant tags...
    Edge * e = HalfToEdge(origin->next);
    float b = 0.0;
    float m = 1.0;
    if (&e->neg==origin->next)
    {
     b = 1.0;
     m = -1.0;
    }
    
    SplitTag * targ = e->dummy.next;
    while (targ!=&e->dummy)
    {
     if ((targ->tag!=NULL)&&(targ->other==NULL))
     {
      float td = dist + len * (b + m*targ->t);
      
      if ((*out==NULL) || (*out_dist > td))
      {
       *out = targ;
       *out_dist = td;
      }
     }
     
     targ = targ->next;
    }
    
    if (*out!=NULL) return; // Found one - done.
   
   // Move to next point in chain...
    origin = origin->next;
    dist += len;
    if (dist>limit) return;
  }
  
 // If we have eatten to a junction check all exits, then recurse them if not done...
  // Check the junction...
   HalfEdge * exit = origin->next;
   while (exit!=origin->reverse)
   {
    float dx = exit->dest->x - origin->dest->x;
    float dy = exit->dest->y - origin->dest->y;
    float len = sqrt(dx*dx + dy*dy);
    
    Edge * e = HalfToEdge(exit);
    float b = 0.0;
    float m = 1.0;
    if (&e->neg==exit)
    {
     b = 1.0;
     m = -1.0;
    }
    
    SplitTag * targ = e->dummy.next;
    while (targ!=&e->dummy)
    {
     if ((targ->tag!=NULL)&&(targ->other==NULL))
     {
      float td = dist + len * (b + m*targ->t);
      
      if ((*out==NULL) || (*out_dist > td))
      {
       *out = targ;
       *out_dist = td;
      }
     }
     
     targ = targ->next;
    }
    
    exit = exit->reverse->next;
   }
  
   if (*out!=NULL) return; // Found one - done.
   
  // Recurse the junction...
   exit = origin->next;
   while (exit!=origin->reverse)
   {
    float dx = exit->dest->x - origin->dest->x;
    float dy = exit->dest->y - origin->dest->y;
    float len = sqrt(dx*dx + dy*dy);
    
    HalfEdge_NearTag(exit, dist + len, out, out_dist, limit);
    
    exit = exit->reverse->next; 
   }
}


static PyObject * LineGraph_between_py(LineGraph * self, PyObject * args)
{
 // Extract the parameters
  int i;
  float t;
  float limit = 1024.0;
  
  if (!PyArg_ParseTuple(args, "if|f", &i, &t, &limit)) return NULL;
  if ((i<0)||(i>=self->edge_count))
  {
   PyErr_SetString(PyExc_IndexError, "Edge index out of range.");
   return NULL;
  }
  
 // Get the edge object...
  Edge * e = &self->edge[i];
  
 // Check if local tags satisfy the query...
  SplitTag * before = NULL;
  SplitTag * after = NULL;
  
  float before_dist = -1.0;
  float after_dist = -1.0;
  
  SplitTag * targ = e->dummy.next;
  while (targ!=&e->dummy)
  {
   if ((targ->tag!=NULL)&&(targ->other==NULL))
   {
    // Its a tag - must be relevent to one of the directions...
     float dx = e->pos.dest->x - e->neg.dest->x;
     float dy = e->pos.dest->y - e->neg.dest->y;
     float dist = sqrt(dx*dx + dy*dy) * fabs(t - targ->t);
     
     if (targ->t<t)
     {
      // Relevant to before... 
       if ((before==NULL)||(before_dist>dist))
       {
        before = targ;
        before_dist = dist;
       }
     }
     else
     {
      // Relevant to after...
       if ((after==NULL)||(after_dist>dist))
       {
        after = targ;
        after_dist = dist;
       }
     }
   }
    
   targ = targ->next;
  }
  
 // Use recursion to satisfy any unanswered queries...
  if (before==NULL)
  {
   HalfEdge_NearTag(&e->neg, t, &before, &before_dist, limit);
  }
  
  if (after==NULL)
  {
   HalfEdge_NearTag(&e->pos, 1.0-t, &after, &after_dist, limit);
  }
  
 // Return the required tuple...
  PyObject * before_obj;
  if (before==NULL)
  {
   before_obj = Py_None;
   Py_INCREF(Py_None);
  }
  else
  {
   before_obj = Py_BuildValue("(f,s,i,f)", before_dist, before->tag, before->loc - self->edge, before->t);
  }
  
  PyObject * after_obj;
  if (after==NULL)
  {
   after_obj = Py_None;
   Py_INCREF(Py_None);
  }
  else
  {
   after_obj = Py_BuildValue("(f,s,i,f)", after_dist, after->tag, after->loc - self->edge, after->t);
  }
  
  return Py_BuildValue("(N,N)", before_obj, after_obj);
}



static PyObject * LineGraph_chain_feature_py(LineGraph * self, PyObject * args)
{
 int i;
 
 // Extract the one possible parameter...
  int bin_count = 8;
  float rad_mult = 1.0;
  float den_mult = 1.0;
  if (!PyArg_ParseTuple(args, "|iff", &bin_count, &rad_mult, &den_mult)) return NULL;
 
 // Create the numpy array that is the return value...
  npy_intp feat_size = bin_count * 4;
  PyObject * feat = PyArray_SimpleNew(1, &feat_size, NPY_FLOAT32);
  
 // Create storage for the weights thus far, for incrimental means...
  float * weight = (float*)malloc(bin_count * 4 * sizeof(int));
  for (i=0; i<bin_count * 4; i++)
  {
   *(float*)PyArray_GETPTR1(feat, i) = 0.0;
   weight[i] = 0.0;
  }
  
 // Calculate the total length of the line...
  float total = 0.0;
  for (i=1; i<self->vertex_count; i++)
  {
   float dx = self->vertex[i].x - self->vertex[i-1].x; 
   float dy = self->vertex[i].y - self->vertex[i-1].y;
   total += sqrt(dx*dx + dy*dy);
  }
 
 // Iterate and process each vertex in turn...
  float dist = 0.0;
  for (i=0; i<self->vertex_count; i++)
  {
   Vertex * targ = self->vertex + i;
   
   // Calculate a normalised orientation vector...
    int low = (i>0) ? (i-1) : (0);
    int high = (i<(self->vertex_count-1)) ? (i+1) : (self->vertex_count-1);
    float dir_x = self->vertex[high].x - self->vertex[low].x;
    float dir_y = self->vertex[high].y - self->vertex[low].y;
    
    float len = sqrt(dir_x*dir_x + dir_y*dir_y);
    if (len<1e-6) continue;
    dir_x /= len;
    dir_y /= len;
    
   // Calculate the bin distribution of the vertex...
    float t = bin_count * (dist / total);
    int lb = (int)floor(t);
    t -= lb;
    
    if (lb+1>=bin_count)
    {
     lb -= 1;
     t += 1.0;
    }
    
    float omt = 1.0 - t;

   // Update the feature vector...
    if (omt>1e-3)
    {
     float * out = (float*)PyArray_GETPTR1(feat, lb);
     weight[lb] += omt;
     *out += omt * (dir_x - *out) / weight[lb];
     
     out = (float*)PyArray_GETPTR1(feat, bin_count + lb);
     weight[bin_count + lb] += omt;
     *out += omt * (dir_y - *out) / weight[bin_count + lb];
     
     out = (float*)PyArray_GETPTR1(feat, 2*bin_count + lb);
     weight[2*bin_count + lb] += omt;
     *out += omt * (rad_mult*targ->radius - *out) / weight[2*bin_count + lb];
     
     out = (float*)PyArray_GETPTR1(feat, 3*bin_count + lb);
     weight[3*bin_count + lb] += omt;
     *out += omt * (den_mult*targ->density - *out) / weight[3*bin_count + lb];
    }

    if (t>1e-3)
    {
     float * out = (float*)PyArray_GETPTR1(feat, lb + 1);
     weight[lb + 1] += t;
     *out += t * (dir_x - *out) / weight[lb + 1];
     
     out = (float*)PyArray_GETPTR1(feat, bin_count + lb + 1);
     weight[bin_count + lb + 1] += t;
     *out += t * (dir_y - *out) / weight[bin_count + lb + 1];
     
     out = (float*)PyArray_GETPTR1(feat, 2*bin_count + lb + 1);
     weight[2*bin_count + lb + 1] += t;
     *out += t * (rad_mult*targ->radius - *out) / weight[2*bin_count + lb + 1];
     
     out = (float*)PyArray_GETPTR1(feat, 3*bin_count + lb + 1);
     weight[3*bin_count + lb + 1] += t;
     *out += t * (den_mult*targ->density - *out) / weight[3*bin_count + lb + 1];
    }

   // To next vertex, in terms of distance...
    if (i+1<self->vertex_count)
    {
     float dx = self->vertex[i+1].x - self->vertex[i].x;
     float dy = self->vertex[i+1].y - self->vertex[i].y;
     dist += sqrt(dx*dx + dy*dy);
    }
  }
  
 // Clean up...
  free(weight);
  
 // Return the feature vector...
  return feat;
}



// Helper method - given a half edge (Starting from the origin vertex of it) and travel distance this travels that distance in the positive direction of the half edge, outputing the half edge and t value of where it ends up. If it hits a junction or the end of a tail it stops there...
void HalfEdge_travel(HalfEdge * start, float distance, HalfEdge ** out, float * out_t)
{
 // Calculate the length...
  float dx = start->dest->x - start->reverse->dest->x;
  float dy = start->dest->y - start->reverse->dest->y; 
  float length = sqrt(dx*dx + dy*dy);
 
 // Check if the output is a location on this HalfEdge...
  if (length>distance)
  {
   *out = start;
   *out_t = distance/length;
   return;
  }
 
 // Verify that the end of this half edge leads to one other, and is not a tail or junction - both are exit early cases...
  if ((start->next==start->reverse) || (start->next->reverse->next!=start->reverse))
  {
   *out = start;
   *out_t = 1.0;
   return; 
  }
 
 // Tail recurse to the next half edge...
  HalfEdge_travel(start->next, distance - length, out, out_t);
}



// Helper for the below - contains all the details needed...
typedef struct FeatureSpec FeatureSpec;

struct FeatureSpec
{
 // Parameters...
  float dir_travel;
  float travel_max;
  int travel_bins;
  float travel_ratio;
  int pos_bins;
  float pos_ratio;
  int radius_bins;
  int density_bins;
  
 // Details for the point being parsed...
  Vertex * vert;
  float ox; // Orientation; unit length.
  float oy; // "
  
 // Bin centres, so they only have to be calculated once; pointers are null if there is only one bin for that class...
  float * travel;
  float * pos_x;
  float * pos_y;
  float * radius;
  float * density;
};



// This runs a path, for the below method, which is to say follows the HalfEdge's out to the given distance and updates the provided feature vector, noting that some care is taken to handle the orientation ambiguity. Recursion is used to handle splits, with a depth limit...
void HalfEdge_feature(FeatureSpec * fs, HalfEdge * targ, float weight, float distance, int rec_limit, float * out)
{
 // Loop until we run out of chain (tail or junction) or distance...
 while (1)
 {
  // Move the length of the edge...
   float dx = targ->dest->x - targ->reverse->dest->x;
   float dy = targ->dest->y - targ->reverse->dest->y;
   distance += sqrt(dx*dx + dy*dy);
   
  // Check if we are done due to having consumed all distance...
   if (distance>fs->travel_max) return;
  
  // Add the vertex at the tail of the half edge to the feature vector...
   // Calculate the bin for distance...
    int travel_base;
    float travel_t;
    int travel_count;
    
    if (fs->travel==NULL)
    {
     travel_base = 0;
     travel_t = 0.0;
     travel_count = 1;
    }
    else
    {
     int low = 0;
     int high = fs->travel_bins-1;
     
     while (low+1<high)
     {
      int half = (low + high) / 2;
      if (fs->travel[half]<=distance) low = half;
                                 else high = half;
     }
     
     travel_base = low;
     travel_t = (distance - fs->travel[travel_base]) / (fs->travel[travel_base+1] - fs->travel[travel_base]);
     travel_count = 2;
    }
    
   // Calculate the x bin for relative position...
    int pos_x_base;
    float pos_x_t;
    int pos_x_count;
    
    if (fs->pos_x==NULL)
    {
     pos_x_base = 0;
     pos_x_t = 0.0;
     pos_x_count = 1;
    }
    else
    {
     int low = 0;
     int high = fs->pos_bins-1;
     
     float pos_x = fs->ox * (targ->dest->x - fs->vert->x) + fs->oy * (targ->dest->y - fs->vert->y);
     
     while (low+1<high)
     {
      int half = (low + high) / 2;
      if (fs->pos_x[half]<=pos_x) low = half;
                             else high = half;
     }
     
     pos_x_base = low;
     pos_x_t = (pos_x - fs->pos_x[pos_x_base]) / (fs->pos_x[pos_x_base+1] - fs->pos_x[pos_x_base]);
     pos_x_count = 2;
    }
    
   // Calculate the y bin for relative position...
    int pos_y_base;
    float pos_y_t;
    int pos_y_count;
    
    if (fs->pos_y==NULL)
    {
     pos_y_base = 0;
     pos_y_t = 0.0;
     pos_y_count = 1;
    }
    else
    {
     int low = 0;
     int high = fs->pos_bins-1;
     
     float pos_y = fs->oy * (targ->dest->x - fs->vert->x) - fs->ox * (targ->dest->y - fs->vert->y);
     
     while (low+1<high)
     {
      int half = (low + high) / 2;
      if (fs->pos_y[half]<=pos_y) low = half;
                             else high = half;
     }
     
     pos_y_base = low;
     pos_y_t = (pos_y - fs->pos_y[pos_y_base]) / (fs->pos_y[pos_y_base+1] - fs->pos_y[pos_y_base]);
     pos_y_count = 2;
    }
   
   // Calculate the bin for radius...
    int radius_base;
    float radius_t;
    int radius_count;
    
    if (fs->radius==NULL)
    {
     radius_base = 0;
     radius_t = 0.0;
     radius_count = 1;
    }
    else
    {
     float r = fs->vert->radius;
     if (r<1e-3) r = 1e-3;
     float radius = targ->dest->radius / r;
     
     if (radius<=fs->radius[0])
     {
      radius_base = 0;
      radius_t = 0.0;
      radius_count = 1;
     }
     else
     {
      if (radius>=fs->radius[fs->radius_bins-1])
      {
       radius_base = fs->radius_bins-1;
       radius_t = 0.0;
       radius_count = 1; 
      }
      else
      {
       int low = 0;
       int high = fs->radius_bins-1;

       while (low+1<high)
       {
        int half = (low + high) / 2;
        if (fs->radius[half]<=radius) low = half;
                               else high = half;
       }
     
       radius_base = low;
       radius_t = (radius - fs->radius[radius_base]) / (fs->radius[radius_base+1] - fs->radius[radius_base]);
       radius_count = 2;
      }
     }
    }
   
   // Calculate the bin for density...
    int density_base;
    float density_t;
    int density_count;
    
    if (fs->density==NULL)
    {
     density_base = 0;
     density_t = 0.0;
     density_count = 1;
    }
    else
    {
     int low = 0;
     int high = fs->density_bins-1;
     
     float density = targ->dest->density - fs->vert->density;
     
     while (low+1<high)
     {
      int half = (low + high) / 2;
      if (fs->density[half]<=density) low = half;
                                 else high = half;
     }
     
     density_base = low;
     density_t = (density - fs->density[density_base]) / (fs->density[density_base+1] - fs->density[density_base]);
     density_count = 2;
     
     if (density_t<0.0) density_t = 0.0;
     else
     {
      if (density_t>1.0) density_t = 1.0;
     }
    }
   
   // Add the weight to the relevant bins, with linear interpolation between them...
    int travel_i;
    for (travel_i=0; travel_i<travel_count; travel_i++)
    {
     float travel_weight = weight * ((travel_i==0) ? (1.0-travel_t) : travel_t);
     int travel_index = travel_base + travel_i;
     
     int pos_x_i;
     for (pos_x_i=0; pos_x_i<pos_x_count; pos_x_i++)
     {
      float pos_x_weight = travel_weight * ((pos_x_i==0) ? (1.0-pos_x_t) : pos_x_t);
      int pos_x_index = travel_index * fs->pos_bins + pos_x_base + pos_x_i;
      
      int pos_y_i;
      for (pos_y_i=0; pos_y_i<pos_y_count; pos_y_i++)
      {
       float pos_y_weight = pos_x_weight * ((pos_y_i==0) ? (1.0-pos_y_t) : pos_y_t);
       int pos_y_index = pos_x_index * fs->pos_bins + pos_y_base + pos_y_i;
       
       int radius_i;
       for (radius_i=0; radius_i<radius_count; radius_i++)
       {
        float radius_weight = pos_y_weight * ((radius_i==0) ? (1.0-radius_t) : radius_t);
        int radius_index = pos_y_index * fs->radius_bins + radius_base + radius_i; 
        
        int density_i;
        for (density_i=0; density_i<density_count; density_i++)
        {
         float density_weight = radius_weight * ((density_i==0) ? (1.0-density_t) : density_t);
         int density_index = radius_index * fs->density_bins + density_base + density_i; 
         
         if (density_weight>0.0)
         {
          out[density_index] += density_weight;
         }
        }
       }
      }
     }
    }
   
  // Check if we have hit a tail or junction... 
   if (targ->next==targ->reverse) return; // At a tail - done
   int at_junction = targ->next->reverse->next != targ->reverse;
   if (at_junction && (rec_limit>0)) break; // At a junction
   
  // Move to next in chain...
   if (at_junction)
   {
    // We have stopped diffusing and are now doing a random walk, so get random. All hail the giant squid of high heals!..
     // Count the options...
      int count = 0;
      HalfEdge * loop = targ->next->reverse;
      while (loop!=targ)
      {
       ++count;
       loop = loop->next->reverse;
      }
      
     // Select a random one to grab...
      float r = drand48();
      float sub = 1.0 / count;
     
     // Go select it...
      loop = targ->next->reverse;
      while (loop!=targ)
      {
       r -= sub;
       if (r<0.0)
       {
        targ = loop;
        break; 
       }
       loop = loop->next->reverse;
      }
   }
   else
   {
    // On a chain, so only one location to travel to - easy...
     targ = targ->next;
   }
 }
   
 // Ok, we have a junction - follow all the paths via recursion...
  int count = 0;
  HalfEdge * loop = targ->next->reverse;
  while (loop!=targ)
  {
   ++count;
   loop = loop->next->reverse;
  }
  
  weight *= 1.0 / count;
  
  loop = targ->next->reverse;
  while (loop!=targ)
  {
   HalfEdge_feature(fs, loop->reverse, weight, distance, rec_limit - 1, out);
   loop = loop->next->reverse;
  }
}



static PyObject * LineGraph_features_py(LineGraph * self, PyObject * args, PyObject * kw)
{
 int i;
 
 // Handle the parameters - quite a lot of them...
  FeatureSpec fs;
  fs.dir_travel = 4.0;
  fs.travel_max = 32.0;
  fs.travel_bins = 4;
  fs.travel_ratio = 0.75;
  fs.pos_bins = 3;
  fs.pos_ratio = 0.75;
  fs.radius_bins = 2;
  fs.density_bins = 2;
  
  int rec_depth = 12;
  
  static char * kw_list[] = {"dir_travel", "travel_max", "travel_bins", "travel_ratio", "pos_bins", "pos_ratio", "radius_bins", "density_bins", "rec_depth", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kw, "|ffififiii", kw_list, &fs.dir_travel, &fs.travel_max, &fs.travel_bins, &fs.travel_ratio, &fs.pos_bins, &fs.pos_ratio, &fs.radius_bins, &fs.density_bins, &rec_depth)) return NULL;
  
  if (fs.travel_bins<1) fs.travel_bins = 1;
  if (fs.pos_bins<1) fs.pos_bins = 1;
  if (fs.radius_bins<1) fs.radius_bins = 1;
  if (fs.density_bins<1) fs.density_bins = 1;
  if (rec_depth<0) rec_depth = 0;
  
 // Create the return object - a big fat array...
  npy_intp feats_size[2] = {self->vertex_count, fs.travel_bins * fs.pos_bins * fs.pos_bins * fs.radius_bins * fs.density_bins};
  PyObject * feats = PyArray_SimpleNew(2, feats_size, NPY_FLOAT32);

 // Put bin centres into the fs data structure...
  // Memory allocation - if its uni-bin we ...
   float * bin_stuff = (float*)malloc((fs.travel_bins + 2 * fs.pos_bins + fs.radius_bins + fs.density_bins) * sizeof(float));
   fs.travel = bin_stuff;
   fs.pos_x = fs.travel + fs.travel_bins;
   fs.pos_y = fs.pos_x + fs.pos_bins;
   fs.radius = fs.pos_y + fs.pos_bins;
   fs.density = fs.radius + fs.radius_bins;
   
   if (fs.travel_bins<2) fs.travel = NULL;
   if (fs.pos_bins<2) {fs.pos_x = NULL; fs.pos_y = NULL;}
   if (fs.radius_bins<2) fs.radius = NULL;
   if (fs.density_bins<2) fs.density = NULL;
  
  // Travel distance...
   if (fs.travel!=NULL)
   {
    fs.travel[0] = 0.0;
    float scale = 1.0;
    for (i=1; i<fs.travel_bins; i++)
    {
     fs.travel[i] = fs.travel[i-1] + scale;
     scale /= fs.travel_ratio;
    }
  
    scale = fs.travel_max / fs.travel[fs.travel_bins-1];
    for (i=1; i<fs.travel_bins; i++) fs.travel[i] *= scale;
   }
   
  // x position - the direction of travel - bit strange as we expect to travel far, as most lines are roughly straight, so the bias is to make traveling in a straight direction lead to bins of higher resolution...
   if (fs.pos_x!=NULL)
   {
    fs.pos_x[0] = 0.0;
    float scale = 1.0;
    for (i=1; i<fs.pos_bins; i++)
    {
     fs.pos_x[i] = fs.pos_x[i-1] + scale;
     scale *= fs.pos_ratio;
    }
    
    scale = 2.0 * fs.travel_max / fs.pos_x[fs.pos_bins-1];
    for (i=0; i<fs.pos_bins; i++) fs.pos_x[i] = fs.pos_x[i] * scale - fs.travel_max;
   }
   
  // y position - the tangential variation from the direction of travel - symmetric with a size bias towards the centre, just to make calculating it extra fun...
   if (fs.pos_y!=NULL)
   {
    if ((fs.pos_bins%2)==0)
    {
     // Even bin count...
      int mid_low = (fs.pos_bins-1) / 2;
      fs.pos_y[mid_low] = -0.5;
      fs.pos_y[mid_low+1] = 0.5;
      
      float scale = 1.0 / fs.pos_ratio;
      for (i=1; i<=mid_low; i++)
      {
       float value = fs.pos_y[mid_low+i] + scale;
       fs.pos_y[mid_low+1+i] = value;
       fs.pos_y[mid_low-i] = -value;
       scale /= fs.pos_ratio;
      }
    }
    else
    {
     // Odd bin count...
      int middle_bin = fs.pos_bins / 2;
      fs.pos_y[middle_bin] = 0.0;
      
      float scale = 1.0;
      for (i=1; i<=middle_bin; i++)
      {
       float value = fs.pos_y[middle_bin+i-1] + scale;
       fs.pos_y[middle_bin+i] =  value;
       fs.pos_y[middle_bin-i] = -value;
       scale /= fs.pos_ratio;
      }
    }
    
    float mult = -fs.travel_max / fs.pos_y[0];
    for (i=0; i<fs.pos_bins; i++) fs.pos_y[i] *= mult;
   }
  
  // Bins for radius - as this is unbounded we use the radius divided by the radius of the start, and clamp at twice as big/half the size. Factor in the multiplicative effect to the bin centres, i.e. use log space, to make it more fun...
   if (fs.radius!=NULL)
   {
    float base = log(0.5);
    float offset = log(1.5) - base;
    
    for (i=0; i<fs.radius_bins; i++)
    {
     fs.radius[i] = exp(base + (offset * i) / (float)(fs.radius_bins-1));
    }
   }
   
  // Offset of density - maxes out at 1...
   if (fs.density!=NULL)
   {
    for (i=0; i<fs.density_bins; i++)
    {
     fs.density[i] = -1.0 + (2.0 * i) / (float)(fs.density_bins-1);
    }
   }
  
 // Calculate the features for each vertex in turn...
  int vert;
  for (vert=0; vert<PyArray_DIMS(feats)[0]; vert++)
  {
   // Zero the output to start with...
    int feat;
    for (feat=0; feat<PyArray_DIMS(feats)[1]; feat++)
    {
     *(float*)PyArray_GETPTR2(feats, vert, feat) = 0.0;
    }
   
   // Run the paths and sum them into the feature vector - once for each half edge leaving the vertex...
    fs.vert = self->vertex + vert;
    
    HalfEdge * targ = fs.vert->incident;
    do
    {
     // Calculate the position after traveling the direction distance...
      HalfEdge * dest;
      float dest_t;
      HalfEdge_travel(targ, fs.dir_travel, &dest, &dest_t);
      
      float ex = (1.0-dest_t) * dest->reverse->dest->x + dest_t * dest->dest->x;
      float ey = (1.0-dest_t) * dest->reverse->dest->y + dest_t * dest->dest->y;
     
     // Take the diference from the start to the destination, and store as the orientation...
      fs.ox = ex - fs.vert->x;
      fs.oy = ey - fs.vert->y;
      
     // Normalise the orientation to a unit vector...
      float o_len = sqrt(fs.ox*fs.ox + fs.oy*fs.oy);
      if (o_len>1e-6)
      {
       fs.ox /= o_len;
       fs.oy /= o_len;
      
       // Follow the graph and factor in the vectors into this feature vector...
        HalfEdge_feature(&fs, targ, 1.0, 0.0, rec_depth, (float*)PyArray_GETPTR2(feats, vert, 0));
      }
     
     // To next half edge...
      targ = targ->reverse->next;
    }
    while (targ!=fs.vert->incident);
    
   // Normalise, and apply the sqrt trick as its a histogram and we care about the distance between them...
    float sum = 0.0;
    for (feat=0; feat<PyArray_DIMS(feats)[1]; feat++)
    {
     sum += *(float*)PyArray_GETPTR2(feats, vert, feat);
    }
    
    if (sum<1e-6) sum = 1e-6;
    for (feat=0; feat<PyArray_DIMS(feats)[1]; feat++)
    {
     float * targ = (float*)PyArray_GETPTR2(feats, vert, feat);
     *targ = sqrt(*targ / sum);
    }
  }
  
 // Clean up...
  free(bin_stuff);

 // Return the array of feature vectors...
  return feats; 
}



static PyObject * LineGraph_pos_py(LineGraph * self, PyObject * args)
{
 // Extract the parameters...
  PyArrayObject * hg = NULL;
  if (!PyArg_ParseTuple(args, "|O!", &PyArray_Type, &hg)) return NULL;
  
 // Verify it is a suitable homography...
  if ((hg!=NULL)&&((hg->nd!=2)||(hg->dimensions[0]!=3)||(hg->dimensions[1]!=3)||(hg->descr->kind!='f')||(hg->descr->elsize!=sizeof(float))))
  {
   PyErr_SetString(PyExc_TypeError, "Homography must be a 3x3 matrix of 32 bit floats.");
   return NULL;
  }
 
 // Create a return array...
  npy_intp shape[2] = {self->vertex_count, 2};
  PyObject * ret = PyArray_SimpleNew(2, shape, NPY_FLOAT32);
 
 // Loop and fill the array with kittens...
  int i;
  for (i=0; i<self->vertex_count; i++)
  {
   float x = self->vertex[i].x;
   float y = self->vertex[i].y;
   
   if (hg!=NULL)
   {
    float tx = (*(float*)PyArray_GETPTR2(hg, 0, 0)) * x + (*(float*)PyArray_GETPTR2(hg, 0, 1)) * y + (*(float*)PyArray_GETPTR2(hg, 0, 2));
    float ty = (*(float*)PyArray_GETPTR2(hg, 1, 0)) * x + (*(float*)PyArray_GETPTR2(hg, 1, 1)) * y + (*(float*)PyArray_GETPTR2(hg, 1, 2));
    float tw = (*(float*)PyArray_GETPTR2(hg, 2, 0)) * x + (*(float*)PyArray_GETPTR2(hg, 2, 1)) * y + (*(float*)PyArray_GETPTR2(hg, 2, 2));
    
    x = tx / tw;
    y = ty / tw;
   }
   
   *(float*)PyArray_GETPTR2(ret, i, 0) = x;
   *(float*)PyArray_GETPTR2(ret, i, 1) = y;
  }
 
 // Release the kittens...
  return ret;
}



static PyObject * LineGraph_adjacent_py(LineGraph * self, PyObject * args)
{
 // Extract the segment index...
  int segment;
  if (!PyArg_ParseTuple(args, "i", &segment)) return NULL;
  
 // Check the segmentation is valid...
  if (self->segments<0)
  {
   PyErr_SetString(PyExc_RuntimeError, "Request for adjacent segments when segmentation is invalid");
   return NULL;
  }
  
 // Create a list to return...
  PyObject * ret = PyList_New(0);
 
 // Iterate all edges and find the relevant points...
  int i;
  for (i=0; i<self->edge_count; i++)
  {
   Edge * targ = &self->edge[i];
   
   int seg = targ->segment;
   SplitTag * st = targ->dummy.next;
   while (st!=&targ->dummy)
   {
    // Check if this is a split or a tagged link...
     if ((st->tag==NULL)&&(st->other==NULL))
     {
      // A split...
       if ((st->segment!=seg)&&((st->segment==segment)||(seg==segment)))
       {
        int other_seg = (seg!=segment) ? seg : st->segment;
        PyObject * tup = Py_BuildValue("(i,i,f)", other_seg, i, st->t);
        PyList_Append(ret, tup);
        Py_DECREF(tup);
       }
      
      seg = st->segment;
     }
     else
     {
      if ((st->tag!=NULL)&&(st->other!=NULL)&&(seg==segment))
      {
       // A tagged link...
        // Find out what is on the other side...
         Edge * other = st->other->loc;
         int other_seg = other->segment;
         SplitTag * ost = other->dummy.next;
         while (ost!=st->other)
         {
          if ((ost->tag==NULL)&&(ost->other==NULL))
          {
           other_seg = ost->segment;  
          }
          
          ost = ost->next; 
         }
         
        // Check if its something we are going to record, and if so record it...
         if (other_seg!=segment)
         {
          PyObject * tup = Py_BuildValue("(i,i,f,s,i,f)", other_seg, i, st->t, st->tag, other-self->edge, ost->t);
          PyList_Append(ret, tup);
          Py_DECREF(tup);            
         }
      }
     }
     
    st = st->next; 
   }
  }
  
 // Return...
  return ret;
}



static PyObject * LineGraph_merge_py(LineGraph * self, PyObject * args)
{
 // Extract the segment indices...
  int seg_a;
  int seg_b;
  if (!PyArg_ParseTuple(args, "ii", &seg_a, &seg_b)) return NULL;
  
 // Check the segmentation is valid...
  if (self->segments<0)
  {
   PyErr_SetString(PyExc_RuntimeError, "Request to merge segments when segmentation is invalid");
   return NULL;
  }
  
 // Iterate the edges and do the terminations...
  int kill_count = 0;
  
  int i;
  for (i=0; i<self->edge_count; i++)
  {
   Edge * targ = &self->edge[i];
   
   int seg = targ->segment;
   
   SplitTag * st = targ->dummy.next;
   while (st!=&targ->dummy)
   {
    // Check if this is a split...
     if ((st->tag==NULL)&&(st->other==NULL))
     {
      if (((seg==seg_a)&&(st->segment==seg_b))||((seg==seg_b)&&(st->segment==seg_a)))
      {
       // This split is dividing our two segments - terminate it...
        seg = st->segment;
        kill_count += 1;
        
        SplitTag * next = st->next;
        SplitTag_free(st);
        st = next;
      }
      else
      {
       seg = st->segment; 
       st = st->next;
      }
     }
     else
     {
      st = st->next; 
     }
   }
  }
  
 // If we have terminated splits then the segmentaiton is no longer valid...
  if (kill_count>0) self->segments = -1;
  
 // Return how many splits died...
  return Py_BuildValue("i", kill_count);
}



// Smooths the positions, sizes, densities and weights of points that have precisly two neighbours - interpolates them by strength towards being the half way point of their neighbours. Updates UV's, and you can ask it to repeat a number of times...
void LineGraph_smooth(LineGraph * this, float strength, int repeat)
{
 int i,r;
 float * offset = (float*)malloc(sizeof(float) * 5 * this->vertex_count);
 
 for (r=0; r<repeat; r++)
 {
  // First pass to calculate the offsets...
   for (i=0; i<this->vertex_count; i++)
   {
    Vertex * targ = &this->vertex[i];
    HalfEdge * leave1 = targ->incident;
    HalfEdge * leave2 = leave1->reverse->next;
    if ((leave1!=leave2)&&(leave2->reverse->next==leave1))
    {
     float nx = 0.5 * (leave1->dest->x + leave2->dest->x);
     float ny = 0.5 * (leave1->dest->y + leave2->dest->y);
     float nr = 0.5 * (leave1->dest->radius  + leave2->dest->radius);
     float nd = 0.5 * (leave1->dest->density + leave2->dest->density);
     float nw = 0.5 * (leave1->dest->weight + leave2->dest->weight);
     
     nx = strength * nx + (1.0-strength) * targ->x;
     ny = strength * ny + (1.0-strength) * targ->y;
     nr = strength * nr + (1.0-strength) * targ->radius;
     nd = strength * nd + (1.0-strength) * targ->density;
     nw = strength * nw + (1.0-strength) * targ->weight;
     
     offset[i*5+0] = nx - targ->x;
     offset[i*5+1] = ny - targ->y;
     offset[i*5+2] = nr - targ->radius;
     offset[i*5+3] = nd - targ->density;
     offset[i*5+4] = nw - targ->weight;
    }
    else
    {
     offset[i*5+0] = 0.0;
     offset[i*5+1] = 0.0;
     offset[i*5+2] = 0.0;
     offset[i*5+3] = 0.0;
     offset[i*5+4] = 0.0;
    }
   }
 
  // Second pass to apply the offsets...
   for (i=0; i<this->vertex_count; i++)
   {
    Vertex * targ = &this->vertex[i];
    
    targ->x += offset[i*5+0];
    targ->u += offset[i*5+0];
    targ->y += offset[i*5+1];
    targ->v += offset[i*5+1];
    
    targ->w += offset[i*5+2];
    targ->radius  += offset[i*5+2];
    targ->density += offset[i*5+3];
    targ->weight += offset[i*5+4];
   }
 }
 
 free(offset);
 
 // Build the spatial indexing structure...
  LineGraph_new_spatial_index(this);
}


static PyObject * LineGraph_smooth_py(LineGraph * self, PyObject * args)
{
 // Extract the parameters...
  float strength = 1.0;
  int repeat = 1;

  if (!PyArg_ParseTuple(args, "|fi", &strength, &repeat)) return NULL;

 // Do it...
  LineGraph_smooth(self, strength, repeat);

 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}



// Applys a homography to every vertex in the LineGraph; assumes row major order. Adjusts the radius only if drad is not 0...
void LineGraph_homography_float(LineGraph * this, float * hg, char drad)
{
 float px_x = (hg[0] + hg[2]) / (hg[6] + hg[8]) - hg[2] / hg[8];
 float px_y = (hg[3] + hg[5]) / (hg[6] + hg[8]) - hg[5] / hg[8];
 float py_x = (hg[1] + hg[2]) / (hg[6] + hg[8]) - hg[2] / hg[8];
 float py_y = (hg[4] + hg[5]) / (hg[6] + hg[8]) - hg[5] / hg[8];
 
 float scale = 0.5 * (sqrt(px_x*px_x + px_y*px_y) + sqrt(py_x*py_x + py_y*py_y));
 
 int i;
 for (i=0; i<this->vertex_count; i++)
 {
  float px = this->vertex[i].x;
  float py = this->vertex[i].y;
  
  float x = px*hg[0] + py*hg[1] + hg[2];
  float y = px*hg[3] + py*hg[4] + hg[5];
  float w = px*hg[6] + py*hg[7] + hg[8];
  
  this->vertex[i].x = x / w;
  this->vertex[i].y = y / w;
  
  if (drad!=0) this->vertex[i].radius *= scale;
 }
 
 LineGraph_new_spatial_index(this);
}

void LineGraph_homography_double(LineGraph * this, double * hg, char drad)
{
 double px_x = (hg[0] + hg[2]) / (hg[6] + hg[8]) - hg[2] / hg[8];
 double px_y = (hg[3] + hg[5]) / (hg[6] + hg[8]) - hg[5] / hg[8];
 double py_x = (hg[1] + hg[2]) / (hg[6] + hg[8]) - hg[2] / hg[8];
 double py_y = (hg[4] + hg[5]) / (hg[6] + hg[8]) - hg[5] / hg[8];
 
 double scale = 0.5 * (sqrt(px_x*px_x + px_y*px_y) + sqrt(py_x*py_x + py_y*py_y));
 
 int i;
 for (i=0; i<this->vertex_count; i++)
 {
  double px = this->vertex[i].x; 
  double py = this->vertex[i].y;
  
  double x = px*hg[0] + py*hg[1] + hg[2];
  double y = px*hg[3] + py*hg[4] + hg[5];
  double w = px*hg[6] + py*hg[7] + hg[8];
  
  this->vertex[i].x = x / w;
  this->vertex[i].y = y / w;
  
  if (drad!=0) this->vertex[i].radius *= scale;
 }
 
 LineGraph_new_spatial_index(this);
}


static PyObject * LineGraph_transform_py(LineGraph * self, PyObject * args)
{
 // Extract the parameters...
  PyArrayObject * hg;
  PyObject * do_radius = Py_False;
  if (!PyArg_ParseTuple(args, "O!|O", &PyArray_Type, &hg, &do_radius)) return NULL;
  
  int drad = PyObject_IsTrue(do_radius);
  
 // Verify it is a suitable array...
  if ((hg->nd!=2)||(hg->dimensions[0]!=3)||(hg->dimensions[1]!=3)||(hg->descr->kind!='f'))
  {
   PyErr_SetString(PyExc_TypeError, "Homography must be a 3x3 real matrix.");
   return NULL;
  }
  
 // Do the operation, if we can...
  if (hg->descr->elsize==sizeof(float))
  {
   float packed[9];
   int r,c;
   for (r=0; r<3; r++)
   {
    for (c=0; c<3; c++) packed[r*3 + c] = *(float*)PyArray_GETPTR2(hg, r, c);
   }
   
   LineGraph_homography_float(self, packed, drad);
  }
  else
  {
   if (hg->descr->elsize==sizeof(double))
   {
    double packed[9];
    int r,c;
    for (r=0; r<3; r++)
    {
     for (c=0; c<3; c++) packed[r*3 + c] = *(double*)PyArray_GETPTR2(hg, r, c);
    }
   
    LineGraph_homography_double(self, packed, drad);
   }
   else
   {
    PyErr_SetString(PyExc_TypeError, "Homography must be either float (32 bits) or double (64 bits).");
    return NULL;
   }
  }
 
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}



static PyObject * LineGraph_scale_py(LineGraph * self, PyObject * args)
{
 // Extract the parameters...
  float mult_radius = 1.0;
  float mult_density = 1.0;
  if (!PyArg_ParseTuple(args, "|ff", &mult_radius, &mult_density)) return NULL;
 
 // Do the scaling... 
  int i;
  for (i=0; i<self->vertex_count; i++)
  {
   self->vertex[i].radius  *= mult_radius; 
   self->vertex[i].density *= mult_density;
  }
  
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}



typedef struct VertMatch VertMatch;

struct VertMatch
{
 VertMatch * next;
 int index_lhs;
 float weight; // For rhs, for lhs its 1 minus this.
 int index_rhs;
};

// Blends both line graphs to match the other, given a set of matches with blending weights...
// (If soft==0 it does linear interpolation; if soft!=0 it biases with a cosine curve.)
void LineGraph_blend(LineGraph * lhs, LineGraph * rhs, VertMatch * matches, char soft)
{
 // Current version is written roughly on the assumption that there are two matches either end of a simple line - probably won't work very well otherwise, though it will do 'something'...
 // (Also not very efficient.)
 
 // Count how many matches exist...
  int mCount = 0;
  VertMatch * targ = matches;
  while (targ!=NULL)
  {
   mCount += 1;
   targ = targ->next;
  }
  if (mCount<2) return; // Its just not going to work - give up.
  
 
 // Create distance arrays...
  int i, j;
  
  float * lhs_d = (float*)malloc(lhs->vertex_count * mCount * sizeof(float));
  for (i=0; i<lhs->vertex_count; i++)
  {
   for (j=0; j<mCount; j++) lhs_d[i*mCount + j] = -1.0;
  }
 
  float * rhs_d = (float*)malloc(rhs->vertex_count * mCount * sizeof(float));
  for (i=0; i<rhs->vertex_count; i++)
  {
   for (j=0; j<mCount; j++) rhs_d[i*mCount + j] = -1.0;
  }
  
 // Set the zero distances, plus store the weights for later lookup...
  targ = matches;
  i = 0;
  float * weight = (float*)malloc(mCount * sizeof(float));
  
  while (targ!=NULL)
  {
   lhs_d[targ->index_lhs*mCount + i] = 0.0;
   weight[i] = targ->weight;
   rhs_d[targ->index_rhs*mCount + i] = 0.0;
   
   i += 1;
   targ = targ->next;
  }
 
 // Fill in the distance arrays, using a simple fill approach (Not very efficient.)...
  // lhs...
   int dir = -1;
   while (1)
   {
    int changes = 0;
    
    dir *= -1;
    i = (dir>0) ? (0) : (lhs->vertex_count-1);
    
    while ((i>=0)&&(i<lhs->vertex_count))
    {
     Vertex * targ = &lhs->vertex[i];
     HalfEdge * he = targ->incident;
     if (he!=NULL)
     {
      do
      {
       // Get the other side...
        Vertex * source = he->dest;
        j = source - lhs->vertex;
       
       // Calculate the distance...
        float dx = source->x - targ->x;
        float dy = source->y - targ->y;
        float dist = sqrt(dx*dx + dy*dy);
       
       // Update all the distances...
        int k;
        for (k=0; k<mCount; k++)
        {
         if (lhs_d[j*mCount + k]>=0.0)
         {
          float d = lhs_d[j*mCount + k] + dist;
          if ((lhs_d[i*mCount + k]<0.0)||(d < lhs_d[i*mCount + k]))
          {
           changes += 1;
           lhs_d[i*mCount + k] = d;
          }
         }
        }
        
       he = he->reverse->next;
      }
      while(he!=targ->incident);
     }
      
     i += dir; 
    }

    if (changes==0) break;
   }
 
  // rhs...
   dir = -1;
   while (1)
   {
    int changes = 0;
    
    dir *= -1;
    i = (dir>0) ? (0) : (rhs->vertex_count-1);
    
    while ((i>=0)&&(i<rhs->vertex_count))
    {
     Vertex * targ = &rhs->vertex[i];
     HalfEdge * he = targ->incident;
     if (he!=NULL)
     {
      do
      {
       // Get the other side...
        Vertex * source = he->dest;
        j = source - rhs->vertex;
       
       // Calculate the distance...
        float dx = source->x - targ->x;
        float dy = source->y - targ->y;
        float dist = sqrt(dx*dx + dy*dy);
       
       // Update all the distances...
        int k;
        for (k=0; k<mCount; k++)
        {
         if (rhs_d[j*mCount + k]>=0.0)
         {
          float d = rhs_d[j*mCount + k] + dist;
          if ((rhs_d[i*mCount + k]<0.0)||(d < rhs_d[i*mCount + k]))
          {
           changes += 1;
           rhs_d[i*mCount + k] = d;
          }
         }
        }
        
       he = he->reverse->next;
      }
      while(he!=targ->incident);
     }
      
     i += dir; 
    }

    if (changes==0) break;
   }

 // Normalise the distance matrices to create a coordinate system we can search to find a match for each vertex - zero out all but the two closest and negate the rest of the coordinates. If there is a match point at each complex vertex this should work...
  // lhs...
   for (i=0; i<lhs->vertex_count; i++)
   {
    // Find the indices of the best and second best...
     int first;
     int second;
     if (lhs_d[i*mCount + 0] < lhs_d[i*mCount + 1])
     {
      first = 0;
      second = 1;
     }
     else
     {
      first = 1;
      second = 0;
     }
    
     for (j=2; j<mCount; j++)
     {
      if (lhs_d[i*mCount + j]<lhs_d[i*mCount + second])
      {
       if (lhs_d[i*mCount + j]<lhs_d[i*mCount + first])
       {
        second = first;
        first = j;
       }
       else
       {
        second = j; 
       }
      }
     }
   
    // Set all other indices to -2...
     for (j=0; j<mCount; j++)
     {
      if ((j!=first)&&(j!=second)) lhs_d[i*mCount + j] = -2.0;
     }
   
    // Normalise the two vertices we have kept so they sum to 1...
     float div = lhs_d[i*mCount + first] + lhs_d[i*mCount + second];
     lhs_d[i*mCount + first] /= div;
     lhs_d[i*mCount + second] /= div;
   }
  
  // rhs...
   for (i=0; i<rhs->vertex_count; i++)
   {
    // Find the indices of the best and second best...
     int first;
     int second;
     if (rhs_d[i*mCount + 0] < rhs_d[i*mCount + 1])
     {
      first = 0;
      second = 1;
     }
     else
     {
      first = 1;
      second = 0;
     }
    
     for (j=2; j<mCount; j++)
     {
      if (rhs_d[i*mCount + j]<rhs_d[i*mCount + second])
      {
       if (rhs_d[i*mCount + j]<rhs_d[i*mCount + first])
       {
        second = first;
        first = j;
       }
       else
       {
        second = j; 
       }
      }
     }
   
    // Set all other indices to -2...
     for (j=0; j<mCount; j++)
     {
      if ((j!=first)&&(j!=second)) rhs_d[i*mCount + j] = -2.0;
     }
   
    // Normalise the two vertices we have kept so they sum to 1...
     float div = rhs_d[i*mCount + first] + rhs_d[i*mCount + second];
     rhs_d[i*mCount + first] /= div;
     rhs_d[i*mCount + second] /= div;
   }
  
 // Create arrays of coordinate in the other system, by finding the closest match out of the vertices and then trying its edges for an even closer point...
 // (Current implimentaiton is n^2 - not good.)
  float * lhs_c = (float*)malloc(lhs->vertex_count * 3 * sizeof(float));
  float * rhs_c = (float*)malloc(rhs->vertex_count * 3 * sizeof(float));
    
  // lhs...
   for (i=0; i<lhs->vertex_count; i++)
   {
    // Find the closest distance vector in the other line graph...
     int best = 0;
     float best_dist_sqr = 1e100;
     int m;
     for (j=0; j<rhs->vertex_count; j++)
     {
      float dist_sqr = 0.0;
      for (m=0; m<mCount; m++)
      {
       float delta = lhs_d[i*mCount + m] - rhs_d[j*mCount + m];
       dist_sqr += delta*delta;
      }
      
      if (dist_sqr<best_dist_sqr)
      {
       best = j; 
       best_dist_sqr = dist_sqr;
      }
     }
     
    // Record what we have thus far...
     Vertex * other = &rhs->vertex[best];
     lhs_c[i*3+0] = other->x;
     lhs_c[i*3+1] = other->y;
     lhs_c[i*3+2] = other->radius;
     
    // Consider its edges to see if we can find a closer point on them...
     HalfEdge * he = other->incident;
     
     if (he!=NULL)
     {
      do
      {
       int dest = he->dest - rhs->vertex;
       
       // Solve for t...
        float lenSqr = 0.0;
        for (m=0; m<mCount; m++)
        {
         float delta = rhs_d[dest*mCount + m] - rhs_d[best*mCount + m];
         lenSqr += delta*delta;
        }
        
        float t = 0.0;
        for (m=0; m<mCount; m++)
        {
         float dir = (rhs_d[dest*mCount + m] - rhs_d[best*mCount + m]) / lenSqr; 
         float point = lhs_d[i*mCount + m] - rhs_d[best*mCount + m];
         t += dir * point;
        }
        
        if ((t>0.0)&&(t<1.0))
        {
         // Check if the point is closer - if it is note and record the result...
          float ds = 0.0;
          for (m=0; m<mCount; m++)
          {
           float delta = lhs_d[i*mCount + m] - ((1.0-t)*rhs_d[best*mCount + m] + t*rhs_d[dest*mCount + m]);
           ds += delta*delta;
          }
          
          if (ds<best_dist_sqr)
          {
           best_dist_sqr = ds;
           lhs_c[i*3+0] = other->x * (1.0-t) + he->dest->x * t;
           lhs_c[i*3+1] = other->y * (1.0-t) + he->dest->y * t;
           lhs_c[i*3+2] = other->radius * (1.0-t) + he->dest->radius * t;
          }
        }
       
       he = he->reverse->next;
      }
      while (he!=other->incident);
     }
   }
  
  // rhs...
   for (i=0; i<rhs->vertex_count; i++)
   {
    // Find the closest distance vector in the other line graph...
     int best = 0;
     float best_dist_sqr = 1e100;
     int m;
     for (j=0; j<lhs->vertex_count; j++)
     {
      float dist_sqr = 0.0;
      for (m=0; m<mCount; m++)
      {
       float delta = rhs_d[i*mCount + m] - lhs_d[j*mCount + m];
       dist_sqr += delta*delta;
      }
      
      if (dist_sqr<best_dist_sqr)
      {
       best = j; 
       best_dist_sqr = dist_sqr;
      }
     }
     
    // Record what we have thus far...
     Vertex * other = &lhs->vertex[best];
     rhs_c[i*3+0] = other->x;
     rhs_c[i*3+1] = other->y;
     rhs_c[i*3+2] = other->radius;
     
    // Consider its edges to see if we can find a closer point on them...
     HalfEdge * he = other->incident;
     
     if (he!=NULL)
     {
      do
      {
       int dest = he->dest - lhs->vertex;
       
       // Solve for t...
        float lenSqr = 0.0;
        for (m=0; m<mCount; m++)
        {
         float delta = lhs_d[dest*mCount + m] - lhs_d[best*mCount + m];
         lenSqr += delta*delta;
        }
        
        float t = 0.0;
        for (m=0; m<mCount; m++)
        {
         float dir = (lhs_d[dest*mCount + m] - lhs_d[best*mCount + m]) / lenSqr; 
         float point = rhs_d[i*mCount + m] - lhs_d[best*mCount + m];
         t += dir * point;
        }
        
        if ((t>0.0)&&(t<1.0))
        {
         // Check if the point is closer - if it is note and record the result...
          float ds = 0.0;
          for (m=0; m<mCount; m++)
          {
           float delta = rhs_d[i*mCount + m] - ((1.0-t)*lhs_d[best*mCount + m] + t*lhs_d[dest*mCount + m]);
           ds += delta*delta;
          }
          
          if (ds<best_dist_sqr)
          {
           best_dist_sqr = ds;
           rhs_c[i*3+0] = other->x * (1.0-t) + he->dest->x * t;
           rhs_c[i*3+1] = other->y * (1.0-t) + he->dest->y * t;
           rhs_c[i*3+2] = other->radius * (1.0-t) + he->dest->radius * t;
          }
        }
       
       he = he->reverse->next;
      }
      while (he!=other->incident);
     }
   }
   
 // Use the other coordinates combined with weight interpolation to correct all the coordinates...
  // lhs...
   for (i=0; i<lhs->vertex_count; i++)
   {
    // Calculate the weight, such that 0.0 biases towards lhs...
     float w = 0.0;
     for (j=0; j<mCount; j++)
     {
      if (lhs_d[i*mCount + j]>0.0) w += (1.0-lhs_d[i*mCount + j]) * weight[j];
     }
     
    // Check if its a fixed weight one - if so the above code will not work and we need to correct...
     targ = matches;
     j = 0;
     while (targ!=NULL)
     {
      if (targ->index_lhs==i)
      {
       w = weight[j]; 
       break;
      }
      
      j += 1;
      targ = targ->next; 
     }
     
    // If soft bias the weight...
     if (soft!=0)
     {
      w = 0.5 - 0.5 * cos(w * M_PI);
     }

    // Use the weight to do the interpolation...
     lhs->vertex[i].x = lhs->vertex[i].x*(1.0-w) + lhs_c[i*3+0]*w;
     lhs->vertex[i].y = lhs->vertex[i].y*(1.0-w) + lhs_c[i*3+1]*w;
     lhs->vertex[i].radius = lhs->vertex[i].radius*(1.0-w) + lhs_c[i*3+2]*w;
   }
  
  // rhs...
   for (i=0; i<rhs->vertex_count; i++)
   {
    // Calculate the weight, such that 0.0 biases towards lhs...
     float w = 0.0;
     for (j=0; j<mCount; j++)
     {
      if (rhs_d[i*mCount + j]>0.0) w += (1.0-rhs_d[i*mCount + j]) * weight[j];
     }
     
    // Check if its a fixed weight one - if so the above code will not work and we need to correct...
     targ = matches;
     j = 0;
     while (targ!=NULL)
     {
      if (targ->index_rhs==i)
      {
       w = weight[j]; 
       break;
      }
      
      j += 1;
      targ = targ->next; 
     }
     
    // If soft bias the weight...
     if (soft!=0)
     {
      w = 0.5 - 0.5 * cos(w * M_PI);
     }
     
    // Use the weight to do the interpolation...
     rhs->vertex[i].x = rhs->vertex[i].x*w + rhs_c[i*3+0]*(1.0-w);
     rhs->vertex[i].y = rhs->vertex[i].y*w + rhs_c[i*3+1]*(1.0-w);
     rhs->vertex[i].radius = rhs->vertex[i].radius*w + rhs_c[i*3+2]*(1.0-w);
   }
 
 // Clean up...
  free(rhs_c);
  free(lhs_c);
  free(weight);
  free(rhs_d);
  free(lhs_d);
 
 // Recalculate the spatial indexing structure...
  LineGraph_new_spatial_index(lhs);
  LineGraph_new_spatial_index(rhs);
}


static PyObject * LineGraph_blend_py(LineGraph * self, PyObject * args)
{
 // Extract the parameters...
  LineGraph * other;
  PyObject * list;
  PyObject * softer = Py_False;
  if (!PyArg_ParseTuple(args, "O!O!|O", &LineGraphType, &other, &PyList_Type, &list, &softer)) return NULL;
  
 // Convert the list of matches from Tuples to an actual struct...
  Py_ssize_t len = PyList_Size(list);
  VertMatch * vm = (VertMatch*)malloc(len * sizeof(VertMatch));
  
  int i;
  for (i=0; i<len; i++)
  {
   PyObject * tup = PyList_GetItem(list, i);
   if ((PyTuple_Check(tup)==0)||(PyTuple_Size(tup)!=3))
   {
    PyErr_SetString(PyExc_RuntimeError, "Matches must be represented by 3-tuples of (this vertex index, weight, other vertex index).");
    free(vm);
    return NULL; 
   }
   
   vm[i].next = &vm[i+1];
   
   vm[i].index_lhs = PyInt_AsLong(PyTuple_GetItem(tup, 0));
   vm[i].weight = PyFloat_AsDouble(PyTuple_GetItem(tup, 1));
   vm[i].index_rhs = PyInt_AsLong(PyTuple_GetItem(tup, 2));
   
   if ((vm[i].index_lhs<0)||(vm[i].index_lhs>=self->vertex_count)||(vm[i].weight<0.0)||(vm[i].weight>1.0)||(vm[i].index_rhs<0)||(vm[i].index_rhs>=other->vertex_count))
   {
    PyErr_SetString(PyExc_RuntimeError, "Match contains an out-of-bounds number.");
    free(vm);
    return NULL; 
   }
  }
  vm[len-1].next = NULL;
 
 // Do the actual blend...
  char soft = (softer==Py_False) ? 0 : 1;
  LineGraph_blend(self, other, vm, soft);
 
 // Clean up...
  free(vm);
 
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}



void LineGraph_morph_to(LineGraph * this, LineGraph * other, int vert_count, int * vert, float weight)
{
 if (this->vertex_count<2) return;
 if (vert_count<2) return;
 
 int i;
 float dx;
 float dy;
 
 // First calculate the length of this segment...
  float len_this = 0.0;
  for (i=1; i<this->vertex_count; i++)
  {
   dx = this->vertex[i].x - this->vertex[i-1].x;
   dy = this->vertex[i].y - this->vertex[i-1].y;
   len_this += sqrt(dx*dx + dy*dy);
  }
 
 // Now calculate the length of the target segment...
  float len_other = 0.0;
  for (i=1; i<vert_count; i++)
  {
   dx = other->vertex[vert[i]].x - other->vertex[vert[i-1]].x;
   dy = other->vertex[vert[i]].y - other->vertex[vert[i-1]].y;
   len_other += sqrt(dx*dx + dy*dy);
  }
 
 // Iterate the vertices of this and update each in turn - can walk the other vertices once by steping forward only when required...
  int oi = 0;
  float oi_start = 0.0;
  dx = other->vertex[vert[1]].x - other->vertex[vert[0]].x;
  dy = other->vertex[vert[1]].y - other->vertex[vert[0]].y;
  float oi_end = sqrt(dx*dx + dy*dy);
  
  float dist = 0.0;
  for (i=0; i<this->vertex_count; i++)
  {
   float dist_o = dist * len_other / len_this;
   // Move oi forward until the current vertex is covered by the current range...
    while ((oi<(vert_count-2))&&(dist_o>oi_end))
    {
     ++oi;
     oi_start = oi_end;
     dx = other->vertex[vert[oi+1]].x - other->vertex[vert[oi]].x;
     dy = other->vertex[vert[oi+1]].y - other->vertex[vert[oi]].y;
     oi_end += sqrt(dx*dx + dy*dy);
    }
    
   // Work out the required t value...
    float t = (dist_o - oi_start) / (oi_end - oi_start);
    float omt = 1.0 - t;
    
   // Increase dist to get to the next point...
    if (i+1<this->vertex_count)
    {
     dx = this->vertex[i+1].x - this->vertex[i].x;
     dy = this->vertex[i+1].y - this->vertex[i].y;
     dist += sqrt(dx*dx + dy*dy);
    }
    
   // Do the interpolation to find out the values at the target location...
    float x = omt * other->vertex[vert[oi]].x + t * other->vertex[vert[oi+1]].x;
    float y = omt * other->vertex[vert[oi]].y + t * other->vertex[vert[oi+1]].y;
    float radius = omt * other->vertex[vert[oi]].radius + t * other->vertex[vert[oi+1]].radius;
   
   // Use weight to update the vertex position accordingly...
    this->vertex[i].x = (1.0-weight) * this->vertex[i].x + weight * x;
    this->vertex[i].y = (1.0-weight) * this->vertex[i].y + weight * y;
    this->vertex[i].radius = (1.0-weight) * this->vertex[i].radius + weight * radius;
  }
}


static PyObject * LineGraph_morph_to_py(LineGraph * self, PyObject * args)
{
 // Extract the parameters...
  LineGraph * other;
  PyObject * vert_list;
  float weight = 1.0;
  if (!PyArg_ParseTuple(args, "O!O!|f", &LineGraphType, &other, &PyList_Type, &vert_list, &weight)) return NULL;
 
 // Convert the list into an array...
  int * list = (int*)malloc(PyList_Size(vert_list) * sizeof(int));
  int i;
  for (i=0; i<PyList_Size(vert_list); i++)
  {
   PyObject * val = PyList_GetItem(vert_list, i);
   if (PyInt_Check(val)==0)
   {
    PyErr_SetString(PyExc_RuntimeError, "list of vertex indices contains something that is not a number.");
    free(list);
    return NULL; 
   }
   
   list[i] = PyInt_AsLong(val);
   
   if ((list[i]<0)||(list[i]>=other->vertex_count))
   {
    PyErr_SetString(PyExc_RuntimeError, "vertex index is out of range");
    free(list);
    return NULL;
   }
  }
  
 // Call through to the method that will perform the work...
  LineGraph_morph_to(self, other, PyList_Size(vert_list), list, weight);

 // Clean up...
  free(list);
  
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}



static Region * Region_within(Region * this, float min_x, float max_x, float min_y, float max_y, Region * prev)
{
 // Check if this region is entirly outside the search region...
  if ((min_x>this->max_x)||(max_x<this->min_x)||(min_y>this->max_y)||(max_y<this->min_y))
  {
   return prev;
  }

 // Check if it is entirly inside or it has no children - if so there is no point recursing, just add this entire region to the list...
  if ((this->child_low==NULL)||((min_x<this->min_x)&&(max_x>this->max_x)&&(min_y<this->min_y)&&(max_y>this->max_y)))
  {
   this->next = prev;
   return this;
  }

 // The search region overlaps with the current region, so recurse into its children...
  prev = Region_within(this->child_low,  min_x, max_x, min_y, max_y, prev);
  prev = Region_within(this->child_high, min_x, max_x, min_y, max_y, prev);

  return prev;
}

// Returns a region that is at the front of a linked list of regions, that defines the set of edges that are within the given region. Note that the pointers will be wiped as a result of calling this again, or calling several other operations - use the result immediatly...
Region * LineGraph_within(LineGraph * this, float min_x, float max_x, float min_y, float max_y)
{
 return Region_within(this->root, min_x, max_x, min_y, max_y, NULL);
}


static PyObject * LineGraph_within_py(LineGraph * self, PyObject * args)
{
 // Extract the parameters...
  float min_x;
  float max_x;
  float min_y;
  float max_y;

  if (!PyArg_ParseTuple(args, "ffff", &min_x, &max_x, &min_y, &max_y)) return NULL;

 // Create the region list...
  Region * first = LineGraph_within(self, min_x, max_x, min_y, max_y);

 // Count its length...
  int length = 0;
  Region * targ = first;
  while (targ!=NULL)
  {
   length += 1;
   targ = targ->next;
  }

 // Create a tuple, fill in each entry with a slice for the edge range involved...
  PyObject * ret = PyTuple_New(length);

  length = 0;
  targ = first;
  while (targ!=NULL)
  {
   PyObject * start = PyInt_FromLong(targ->begin);
   PyObject * stop = PyInt_FromLong(targ->end);
   PyObject * slice = PySlice_New(start, stop, NULL);
   PyTuple_SetItem(ret, length, slice);

   Py_DECREF(start);
   Py_DECREF(stop);

   length += 1;
   targ = targ->next;
  }

 // Return the created tuple...
  return ret;
}



// Outputs the closest and furthest distance of the set of points defined by the region to the given 2D point...
void Region_distance(Region * this, float x, float y, float * out_min_dist, float * out_max_dist)
{
 float min_dx = 0.0;
 if (x<this->min_x) min_dx = this->min_x - x;
 else
 {
  if (x>this->max_x) min_dx = x - this->max_x;
 }
 
 float min_dy = 0.0;
 if (y<this->min_y) min_dy = this->min_y - y;
 else
 {
  if (y>this->max_y) min_dy = y - this->max_y;
 }
 
 float a = fabs(x - this->min_x);
 float b = fabs(x - this->max_x);
 float max_dx = (a>b) ? a : b;
 
 a = fabs(y - this->min_y);
 b = fabs(y - this->max_y);
 float max_dy = (a>b) ? a : b;
 
 if (out_min_dist!=NULL)
 {
  *out_min_dist = sqrt(min_dx*min_dx + min_dy*min_dy);
 }
 
 if (out_max_dist!=NULL)
 {
  *out_max_dist = sqrt(max_dx*max_dx + max_dy*max_dy);
 }
}

// Given a point finds the closets point on all of the edges in a region - simple brute force (Merges with previously stored best if overwrite is 0)...
void LineGraph_distance_region_edge(LineGraph * this, Region * r, float x, float y, float * out_distance, Edge ** out_edge, float * out_t, char overwrite)
{
 int i;
 
 float closest;
 if (overwrite!=0) closest = 1e100;
 else closest = *out_distance;
 
 for (i=r->begin; i<r->end; i++)
 {
  Edge * targ = &this->edge[i];
  
  // Calculate a normal vector for the edge...
   float nx = targ->pos.dest->x - targ->neg.dest->x;
   float ny = targ->pos.dest->y - targ->neg.dest->y;
   float l = sqrt(nx*nx + ny*ny);
  
  // A length of zero is possible, for a degenerate edge - treat it as a point...
   if (l<1e-6)
   {
    float dx = x - 0.5 * (targ->pos.dest->x + targ->neg.dest->x);
    float dy = y - 0.5 * (targ->pos.dest->y + targ->neg.dest->y);
    float d = sqrt(dx*dx + dy*dy);
    
    if (d<closest)
    {
     closest = d;
     if (out_distance!=NULL) *out_distance = d;
     if (out_edge!=NULL) *out_edge = targ;
     if (out_t!=NULL) *out_t = 0.5;
    }
    
    continue;
   }
   
  // Normalise the normal vector...
   nx /= l;
   ny /= l;
   
  // Dot product to get the travel distance that puts us at the closest point, on the infinite line...
   float dp = nx * (x - targ->neg.dest->x) + ny * (y - targ->neg.dest->y);
   
  // Factor in the fact that the line is finite, finding the intersect point and t accordingly...
   float t, ix, iy;
   
   if (dp<0.0)
   {
    t = 0.0;
    ix = targ->neg.dest->x;
    iy = targ->neg.dest->y;
   }
   else
   {
    if (dp>l)
    {
     t = 1.0;
     ix = targ->pos.dest->x;
     iy = targ->pos.dest->y;
    }
    else
    {
     t = dp / l;
     ix = targ->neg.dest->x + nx*dp;
     iy = targ->neg.dest->y + ny*dp;
    }
   }

  // Calculate the actual distance - if closer than previous points store it...
   float dx = x - ix;
   float dy = y - iy;
   float d = sqrt(dx*dx + dy*dy);
   
   if (d<closest)
   {
    closest = d;
    if (out_distance!=NULL) *out_distance = d;
    if (out_edge!=NULL) *out_edge = targ;
    if (out_t!=NULL) *out_t = t;
   }  
 }
}

// Finds the closest location on the graph to the given 2D point. Makes use of the spatial support structure...
void Linegraph_nearest(LineGraph * this, float x, float y, float * out_distance, Edge ** out_edge, float * out_t)
{
 // Create a linked list of jobs to do and a record of the closest found thus far...
  Region * work = this->root;
  work->next = NULL;
  
  *out_distance = 1e100;
  
 // Eat the work queue until all done - designed to quickly find a tight bound to help minimise the search effort, whilst avoiding complex data structures... (Could be faster, but fast enough for me.)
  while (work!=NULL)
  {
   // Pop the work item off...
    Region * targ = work;
    work = targ->next;
    
   // Check if its worth considering further...
    float min_dist;
    Region_distance(targ, x, y, &min_dist, NULL);
    if (min_dist>*out_distance) continue;
    
   // If it has no children brute force it, otherwise iterate into the children, being careful to choose the order they are pushed onto the stack so the best one gets processed first...
    if (targ->child_low==NULL)
    {
     // No children - brute force its contents...
      LineGraph_distance_region_edge(this, targ, x, y, out_distance, out_edge, out_t, 0);
    }
    else
    {
     // Find out which child is the best...
      float min_low;
      Region_distance(targ->child_low, x, y, &min_low, NULL);
      
      float min_high;
      Region_distance(targ->child_high, x, y, &min_high, NULL);
      
     // Put the best child onto the stack second, so it gets popped first...
      if (min_low<min_high)
      {
       targ->child_high->next = work;
       work = targ->child_high;
       
       targ->child_low->next = work;
       work = targ->child_low;
      }
      else
      {       
       targ->child_low->next = work;
       work = targ->child_low;
       
       targ->child_high->next = work;
       work = targ->child_high;
      }
    }
  }
}


static PyObject * Linegraph_nearest_py(LineGraph * self, PyObject * args)
{
 // Extract the parameters...
  float x;
  float y;

  if (!PyArg_ParseTuple(args, "ff", &x, &y)) return NULL;
  
 // Run the operation...
  float distance;
  Edge * edge = NULL;
  float t;
  
  Linegraph_nearest(self, x, y, &distance, &edge, &t);
  
 // Generate and return the result...
  if (edge==NULL)
  {
   Py_INCREF(Py_None);
   return Py_None; 
  }
  else
  {
   int e = (edge - self->edge);
   return Py_BuildValue("(f,i,f)", distance, e, t);
  }
}



// Returns 1 if the segment intercepts the region, 0 if it misses it...
char Region_intersect_segment(Region * this, float sx, float sy, float ex, float ey)
{
 // The edges of the region divide space into 9 regions - classify which region each of the line end points is in...
  char sxc = 1;
  if (sx<this->min_x) sxc = 0;
  else
  {
   if (sx>this->max_x) sxc = 2; 
  }
  
  char syc = 1;
  if (sy<this->min_y) syc = 0;
  else
  {
   if (sy>this->max_y) syc = 2; 
  }
  
  char exc = 1;
  if (ex<this->min_x) exc = 0;
  else
  {
   if (ex>this->max_x) exc = 2; 
  }
  
  char eyc = 1;
  if (ey<this->min_y) eyc = 0;
  else
  {
   if (ey>this->max_y) eyc = 2; 
  }
  
 // Check a bunch of cases, that will give us an instant answer based on the classification. These handle most cases most of the time, and do so quickly...
  if ((sxc==exc)&&(syc==eyc))
  {
   // Segment is in a single square - 1,1 means intercept, all others are a miss...
    if ((sxc==1)&&(syc==1)) return 1;
    else return 0;
  }
  
  if (sxc==exc)
  {
   if (sxc==1) return 1;
   else return 0;
  }
  
  if (syc==eyc)
  {
   if (syc==1) return 1;
   else return 0;
  }
  
 // Line is unfortunatly in some complex relationship with the region - do intercept tests with all the edges to see if it does intercept...
  float ox = ex - sx;
  float oy = ey - sy;
  
  // x-axis, minimum...
   if ((sxc==0)||(exc==0))
   {
    float idist = (this->min_x - sx) / ox; 
    float int_y = sy + idist * oy;
    if ((int_y>(this->min_y-1e-3))&&(int_y<(this->max_y+1e-3))) return 1;
   }
   
  // x-axis, maximum...
   if ((sxc==2)||(exc==2))
   {
    float idist = (this->max_x - sx) / ox;
    float int_y = sy + idist * oy;
    if ((int_y>(this->min_y-1e-3))&&(int_y<(this->max_y+1e-3))) return 1;
   }

  // y-axis, minimum...
   if ((syc==0)||(eyc==0))
   {
    float idist = (this->min_y - sy) / oy; 
    float int_x = sx + idist * ox;
    if ((int_x>(this->min_x-1e-3))&&(int_x<(this->max_x+1e-3))) return 1;
   }

  // y-axis, maximum...
   if ((syc==2)||(eyc==2))
   {
    float idist = (this->max_y - sy) / oy; 
    float int_x = sx + idist * ox;
    if ((int_x>(this->min_x-1e-3))&&(int_x<(this->max_x+1e-3))) return 1;
   }
  
  // If missed all edges - its a miss...
   return 0;
}

// Intersect a line segment with all edges in a region, returning a list of those that intersect, with t-values for both parts. Takes ownership of the tail, to which it adds edge intersects, and returns a linked list of edge intersects, which the caller is responsible for deleting...
EdgeInt * LineGraph_intersect_region(LineGraph * this, Region * r, float sx, float sy, float ex, float ey, EdgeInt * tail)
{
 // Convert the provided line into the standard ax + by = c form...
  float a1 = ey - sy;
  float b1 = sx - ex;
  float c1 = a1*sx + b1*sy;
  
 // Iterate and consider each edge in turn...
  int i;
  for (i=r->begin; i<r->end; i++)
  {
   Edge * targ = &this->edge[i];
   
   // Convert the line to the ax + by = c form...
    float a2 = targ->pos.dest->y - targ->neg.dest->y;
    float b2 = targ->neg.dest->x - targ->pos.dest->x;
    float c2 = a2*targ->neg.dest->x + b2*targ->neg.dest->y;
    
   // Intersect the two lines...
    float det = a1*b2 - a2*b1;
    if (fabs(det)<1e-6) continue; // Parallel - ignore.
      
    float ix = (b2*c1 - b1*c2) / det;
    float iy = (a1*c2 - a2*c1) / det;
    
   // Find the t values for both lines, verifying they are in intersect range...
    float t1;
    if (fabs(ex-sx)>fabs(ey-sy))
    {
     t1 = (ix - sx) / (ex - sx);
    }
    else
    {
     t1 = (iy - sy) / (ey - sy);
    }
    if ((t1<0.0)||(t1>1.0)) continue;
   
    float t2;
    if (fabs(targ->pos.dest->x - targ->neg.dest->x)>fabs(targ->pos.dest->y - targ->neg.dest->y))
    {
     t2 = (ix - targ->neg.dest->x) / (targ->pos.dest->x - targ->neg.dest->x);
    }
    else
    {
     t2 = (iy - targ->neg.dest->y) / (targ->pos.dest->y - targ->neg.dest->y); 
    }
    if ((t2<0.0)||(t2>1.0)) continue;
      
   // We have an intersection - store the relevant information into the linked list...
    EdgeInt * ei = (EdgeInt*)malloc(sizeof(EdgeInt));

    ei->edge = targ;
    ei->edge_t = t2;
    ei->other_t = t1;
    
    ei->next = tail;
    tail = ei;
  }
 
 return tail; 
}

// Returns a list of intersections for the provided line segment (from (sx,sy) to (ex,ey).). You are responsible for deleting the intersections list when done.
EdgeInt * LineGraph_intersect(LineGraph * this, float sx, float sy, float ex, float ey)
{
 Region * work = this->root;
 work->next = NULL;
 EdgeInt * ret = NULL;
 
 while (work!=NULL)
 {
  // Pop the work...
   Region * targ = work;
   work = work->next;
  
  // Check this work item is worth dealing with - does the segment intersect with it?..
   if (Region_intersect_segment(targ, sx, sy, ex, ey)==0) continue;
   
  // Either push its children onto the work stack or do the intersection...
   if (targ->child_low==NULL)
   {
    ret = LineGraph_intersect_region(this, targ, sx, sy, ex, ey, ret);
   }
   else
   {
    targ->child_low->next = work;
    work = targ->child_low;
    
    targ->child_high->next = work;
    work = targ->child_high;
   }
 }
 
 return ret;
}


static PyObject * LineGraph_intersect_py(LineGraph * self, PyObject * args)
{
 // Extract the parameters...
  float sx, sy, ex, ey;
  if (!PyArg_ParseTuple(args, "ffff", &sx, &sy, &ex, &ey)) return NULL;
  
 // Find the intersections...
  EdgeInt * start = LineGraph_intersect(self, sx, sy, ex, ey);
  
 // Convert them into a list of tuples...
  int count = 0;
  EdgeInt * targ = start;
  while (targ!=NULL)
  {
   count += 1;
   targ = targ->next; 
  }
  
  PyObject * ret = PyTuple_New(count);
  
  count = 0;
  targ = start;
  while (targ!=NULL)
  {
   int edge_i = targ->edge - self->edge;
   PyTuple_SetItem(ret, count, Py_BuildValue("(i,f,f)", edge_i, targ->edge_t, targ->other_t));
   
   count += 1;
   targ = targ->next; 
  }
  
 // Clean up...
  EdgeInt_free(start);
  
 // Return...
  return ret;
}



static PyObject * LineGraph_intersect_links_py(LineGraph * self, PyObject * args)
{
 // Extract the parameters...
  float sx, sy, ex, ey;
  if (!PyArg_ParseTuple(args, "ffff", &sx, &sy, &ex, &ey)) return NULL;
 
 // Convert the provided line into the standard ax + by = c form...
  float a1 = ey - sy;
  float b1 = sx - ex;
  float c1 = a1*sx + b1*sy;
  
 // Create the return list...
  PyObject * ret = PyList_New(0);

 // Iterate all edges, to find all links...
  int i;
  for (i=0; i<self->edge_count; i++)
  {
   Edge * targ = &self->edge[i]; 
   
   SplitTag * st = targ->dummy.next;
   while (st!=&targ->dummy)
   {
    // Only process links, and only do each one once...
     if ((st->other!=NULL)&&(st->other<st))
     {
      // Get the start and end coordinate for the link... 
       float lsx = targ->neg.dest->x * (1.0 - st->t) + targ->pos.dest->x * st->t;
       float lsy = targ->neg.dest->y * (1.0 - st->t) + targ->pos.dest->y * st->t;
       float lex = st->other->loc->neg.dest->x * (1.0 - st->other->t) + st->other->loc->pos.dest->x * st->other->t;
       float ley = st->other->loc->neg.dest->y * (1.0 - st->other->t) + st->other->loc->pos.dest->y * st->other->t;
      
      // Convert into a suitable parametric form...
       float a2 = ley - lsy;
       float b2 = lsx - lex;
       float c2 = a2*lsx + b2*lsy;
    
      // Intersect the two lines...
       float det = a1*b2 - a2*b1;

       if (fabs(det)>1e-6) // Only continue if not parallel.
       {
        float ix = (b2*c1 - b1*c2) / det;
        float iy = (a1*c2 - a2*c1) / det;
    
        // Find the t values for both lines, verifying they are within intersect range...
         float t1;
         if (fabs(ex-sx)>fabs(ey-sy))
         {
          t1 = (ix - sx) / (ex - sx);
         }
         else
         {
          t1 = (iy - sy) / (ey - sy);
         }
         
         if ((t1>=0.0)&&(t1<=1.0)) // Only continue if its in range.
         {
          float t2;
          if (fabs(lex - lsx)>fabs(ley - lsy))
          {
           t2 = (ix - lsx) / (lex - lsx);
          }
          else
          {
           t2 = (iy - lsy) / (ley - lsy); 
          }

          if ((t2>=0.0)&&(t2<=1.0)) // Only continue if its in range.
          {
           // At this point we have a collision, so add the intercept to the return list...
            PyObject * tup = Py_BuildValue("(f,i,f,i,f,f,s)", t1, targ-self->edge, st->t, st->other->loc-self->edge, st->other->t, t2, st->tag);
            PyList_Append(ret, tup);
            Py_DECREF(tup);
          }
         }
       }
     }
    
    st = st->next; 
   }
  } 
  
 // Return the list...
  return ret;
}



static PyObject * LineGraph_from_dict_py(LineGraph * self, PyObject * args)
{
 // Get the dictionary that has been passed in...
  PyObject * dict;
  if (!PyArg_ParseTuple(args, "O!", &PyDict_Type, &dict)) return NULL;
  
 // Index the dictionary to fetch the important arrays of data...
  PyObject * element = PyDict_GetItemString(dict, "element");
  if (element==NULL)
  {
   PyErr_SetString(PyExc_RuntimeError, "Provided dictionary does not include the 'element' entry, which is expected to contain all of the data.");
   return NULL;
  }
  
  PyObject * vertex = PyDict_GetItemString(element, "vertex");
  if (vertex==NULL)
  {
   PyErr_SetString(PyExc_RuntimeError, "Provided dictionary does not include 'vertex' under 'element', which is required.");
   return NULL;
  }
  
  PyObject * vertex_x = PyDict_GetItemString(vertex, "x");
  PyObject * vertex_y = PyDict_GetItemString(vertex, "y");
  PyObject * vertex_u = PyDict_GetItemString(vertex, "u");
  PyObject * vertex_v = PyDict_GetItemString(vertex, "v");
  PyObject * vertex_w = PyDict_GetItemString(vertex, "w");
  PyObject * vertex_radius = PyDict_GetItemString(vertex, "radius");
  PyObject * vertex_density = PyDict_GetItemString(vertex, "density");
  PyObject * vertex_weight = PyDict_GetItemString(vertex, "weight");
  if ((vertex_x==NULL)||(vertex_y==NULL))
  {
   PyErr_SetString(PyExc_RuntimeError, "Vertices must, at a minimum, have x and y coordinates.");
   return NULL;
  }
  
  PyObject * edge = PyDict_GetItemString(element, "edge");
  if (edge==NULL)
  {
   PyErr_SetString(PyExc_RuntimeError, "Provided dictionary does not include 'edge' under 'element', which is required.");
   return NULL;
  }
  
  PyObject * edge_from = PyDict_GetItemString(edge, "from");
  PyObject * edge_to = PyDict_GetItemString(edge, "to");
  if ((edge_from==NULL)||(edge_to==NULL))
  {
   PyErr_SetString(PyExc_RuntimeError, "Edges must have 'from' and 'to' vertex indices.");
   return NULL;
  }
  
 // Empty the current structure...
  LineGraph_dealloc(self);
  
 // Create the vertices...
  self->vertex_count = PyArray_DIMS(vertex_x)[0];
  self->vertex = (Vertex*)malloc(self->vertex_count * sizeof(Vertex));
  
  int i;
  for (i=0; i<self->vertex_count; i++)
  {
   Vertex * targ = &self->vertex[i];
   targ->incident = NULL;
   targ->source = -1;
   
   targ->x = *(float*)PyArray_GETPTR1(vertex_x, i);
   targ->y = *(float*)PyArray_GETPTR1(vertex_y, i);
   targ->u = (vertex_u!=NULL) ? (*(float*)PyArray_GETPTR1(vertex_u, i)) : (targ->x);
   targ->v = (vertex_v!=NULL) ? (*(float*)PyArray_GETPTR1(vertex_v, i)) : (targ->y);
   targ->radius = (vertex_radius!=NULL) ? (*(float*)PyArray_GETPTR1(vertex_radius, i)) : (1.0);
   targ->w = (vertex_w!=NULL) ? (*(float*)PyArray_GETPTR1(vertex_w, i)) : (targ->radius);
   targ->density = (vertex_density!=NULL) ? (*(float*)PyArray_GETPTR1(vertex_density, i)) : (1.0);
   targ->weight = (vertex_weight!=NULL) ? (*(float*)PyArray_GETPTR1(vertex_weight, i)) : (1.0);
  }
  
 // Create the edges...
  self->edge_count = PyArray_DIMS(edge_from)[0];
  self->edge = (Edge*)malloc(self->edge_count * sizeof(Edge));
  
  for (i=0; i<self->edge_count; i++)
  {
   Edge * targ = &self->edge[i];
   
   int from = *(int*)PyArray_GETPTR1(edge_from, i);
   int to = *(int*)PyArray_GETPTR1(edge_to, i);
   
   Edge_init(targ, &self->vertex[from], &self->vertex[to]);
  }
  
 // Load the splits...
  PyObject * split = PyDict_GetItemString(element, "split");
  if (split!=NULL)
  {
   PyObject * split_edge = PyDict_GetItemString(split, "edge");
   PyObject * split_t = PyDict_GetItemString(split, "t");
  
   if (split_edge!=NULL)
   {
    for (i=0; i<PyArray_DIMS(split_edge)[0]; i++)
    {
     int edge = *(int*)PyArray_GETPTR1(split_edge, i);
     float t = (split_t!=NULL) ? (*(float*)PyArray_GETPTR1(split_t, i)) : (0.5);
    
     LineGraph_add_split_tag(self, &self->edge[edge], t, NULL);
    }
   }
  }

 // Load the tags...
  PyObject * tag = PyDict_GetItemString(element, "tag");
  if (tag!=NULL)
  {
   PyObject * tag_edge = PyDict_GetItemString(tag, "edge");
   PyObject * tag_t = PyDict_GetItemString(tag, "t");
   PyObject * tag_text = PyDict_GetItemString(tag, "text");
  
   if ((tag_edge!=NULL)&&(tag_text!=NULL))
   {
    for (i=0; i<PyArray_DIMS(tag_edge)[0]; i++)
    {
     int edge = *(int*)PyArray_GETPTR1(tag_edge, i);
     float t = (tag_t!=NULL) ? (*(float*)PyArray_GETPTR1(tag_t, i)) : (0.5);
     PyObject * text = *(PyObject**)PyArray_GETPTR1(tag_text, i);
    
     LineGraph_add_split_tag(self, &self->edge[edge], t, PyString_AsString(text));
    }
   }
  }
  
 // Load the links...
  PyObject * link = PyDict_GetItemString(element, "link");
  if (link!=NULL)
  {
   PyObject * link_from_edge = PyDict_GetItemString(link, "from_edge"); 
   PyObject * link_from_t = PyDict_GetItemString(link, "from_t"); 
   PyObject * link_to_edge = PyDict_GetItemString(link, "to_edge"); 
   PyObject * link_to_t = PyDict_GetItemString(link, "to_t"); 
  
   if ((link_from_edge!=NULL)&&(link_to_edge!=NULL))
   {
    for (i=0; i<PyArray_DIMS(link_from_edge)[0]; i++)
    {
     int from_edge = *(int*)PyArray_GETPTR1(link_from_edge, i);
     float from_t = (link_from_t!=NULL) ? (*(float*)PyArray_GETPTR1(link_from_t, i)) : (0.5);
     int to_edge = *(int*)PyArray_GETPTR1(link_to_edge, i);
     float to_t = (link_to_t!=NULL) ? (*(float*)PyArray_GETPTR1(link_to_t, i)) : (0.5);
    
     LineGraph_add_link(self, &self->edge[from_edge], from_t, &self->edge[to_edge], to_t, NULL);
    }
   }
  }
  
 // Load the link_tags...
  PyObject * link_tag = PyDict_GetItemString(element, "link_tag");
  if (link_tag!=NULL)
  {
   PyObject * link_tag_from_edge = PyDict_GetItemString(link_tag, "from_edge"); 
   PyObject * link_tag_from_t = PyDict_GetItemString(link_tag, "from_t"); 
   PyObject * link_tag_to_edge = PyDict_GetItemString(link_tag, "to_edge"); 
   PyObject * link_tag_to_t = PyDict_GetItemString(link_tag, "to_t");
   PyObject * link_tag_text = PyDict_GetItemString(link_tag, "text");
  
   if ((link_tag_from_edge!=NULL)&&(link_tag_to_edge!=NULL)&&(link_tag_text!=NULL))
   {
    for (i=0; i<PyArray_DIMS(link_tag_from_edge)[0]; i++)
    {
     int from_edge = *(int*)PyArray_GETPTR1(link_tag_from_edge, i);
     float from_t = (link_tag_from_t!=NULL) ? (*(float*)PyArray_GETPTR1(link_tag_from_t, i)) : (0.5);
     int to_edge = *(int*)PyArray_GETPTR1(link_tag_to_edge, i);
     float to_t = (link_tag_to_t!=NULL) ? (*(float*)PyArray_GETPTR1(link_tag_to_t, i)) : (0.5);
     PyObject * text = *(PyObject**)PyArray_GETPTR1(link_tag_text, i);
    
     LineGraph_add_link(self, &self->edge[from_edge], from_t, &self->edge[to_edge], to_t, PyString_AsString(text));
    }
   }
  }
 
 // Build the spatial indexing structure...
  LineGraph_new_spatial_index(self);

 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}



static PyMemberDef LineGraph_members[] =
{
 {"vertex_count", T_INT, offsetof(LineGraph, vertex_count), READONLY, "Number of vertices in the graph."},
 {"edge_count", T_INT, offsetof(LineGraph, edge_count), READONLY, "Number of edges in the graph."},
 {"segments", T_INT, offsetof(LineGraph, segments), READONLY, "Number of segments that have been assigned to the graph - they will be indexed using the natural numbers less than this. If no segmentation exists this will be negative."},
 {NULL}
};



static PyMethodDef LineGraph_methods[] =
{
 {"clear", (PyCFunction)LineGraph_clear_py, METH_VARARGS, "Empties the object, deleting all information and setting it to contain no edges/vertices etc."},
 
 {"from_many",(PyCFunction)LineGraph_from_many_py, METH_VARARGS, "Merges together an arbitrary number of LineGraphs into one LineGraph, applying homographies to each individual LineGraph as it is merged. You provide a list of arguments containing both homographies (3x3 numpy arrays) and LineGraphs. Homographies are multiplied together as they appear, and then used by the first LineGraph to follow them. Each time a LineGraph is seen all homographies are removed and the transformation starts again, from the identity. If homographies are not provided then it uses the identity. Can be used as a copy constructor."},
 {"from_mask", (PyCFunction)LineGraph_from_mask_py, METH_VARARGS, "Replaces the contents of the object by extracting vertices/edges using 8-way connectivity in a mask (A 2D numpy array). You can optionally provide radius, density and weight arrays after the mask."},
 {"from_segment", (PyCFunction)LineGraph_from_segment_py, METH_VARARGS, "Given a LineGraph that has a valid segmentation and an integer to select a segment - this fills in this LineGraph with just that segment. Must not be the same LineGraph as the source. origin indices for the vertices and edges will be recorded. Returns a list of tuples indicating the cuts that have been made, and the vertices introduced as a result: (vertex #, segment # for the other side.)."},
 {"from_path", (PyCFunction)LineGraph_from_path_py, METH_VARARGS, "Given another line graph and two vertices in that line graph this works out the shortest path between them and copies everything on that path. End result is a chain rather than a graph, which is useful, for instance, for the blend method, as other shapes can be a problem for it. Does not copy splits/tags/links over, as any given tag could appear on the culled or not culled geometry. Returns the indices of the passed in vertices in the new structure as a tuple (v1, v2)"},
 {"from_vertices", (PyCFunction)LineGraph_from_vertices_py, METH_VARARGS, "Given a line graph and a list of vertices in that line graph sets this line graph to the vertices and all edges that connect them. The order of the vertices in the list defines the order in the output. Splits, tags and links are not copied over."},

 {"from_dict", (PyCFunction)LineGraph_from_dict_py, METH_VARARGS, "Replaces the current contents with a line graph loaded from a dictionary in the style generated by the as_dict method. This is of course what the ply2 loader will provide."},
 {"as_dict", (PyCFunction)LineGraph_as_dict_py, METH_VARARGS, "Returns a dictionary of numpy arrays that represents the state of the LineGraph - the same format that the ply2 i/o library uses."},
 
 {"segment", (PyCFunction)LineGraph_segment_py, METH_VARARGS, "Segments the line graph, assigning an (arbitrary) integer to every location, such that connected locations have the same integer. Connections are created by the edges obviously, but splits will break the flow, and links without tags will connect otherwise disparate areas. This segmentation is automatically invalidated if a split/tag/link is added/removed, and added if it is needed by another method call (That does not take segments as a parameter.), so typically you would not call this method directly. There is absolutly no guarantee of consistancy between segmentations after a change of splits/tags/links."},
 
 {"get_bounds", (PyCFunction)LineGraph_get_bounds_py, METH_VARARGS, "Returns the bounds of the line graph, taking into account the radii of the lines - e.g. the region you have to draw to see it all. Return is the tuple (min_x, max_x, min_y, max_y). If you provide a segment index it will return the bounding box of that segment, though note that this is a slow operation, whilst bounds for everything is a simple lookup. Will return None if bounds are not defined."},
 {"get_vertex", (PyCFunction)LineGraph_get_vertex_py, METH_VARARGS, "Given the index of a vertex returns a tuple of infomation about it: (x, y, u, v, w, radius, density, weight, index of vertex in origin LineGraph, or None if no origin.)"},
 {"get_edge", (PyCFunction)LineGraph_get_edge_py, METH_VARARGS, "Given the index of an edge returns a tuple of infomation about it: (from vertex_index, to vertex index, list of tags/splits, each as a tuple (t, string if it is a tag, None if it is a split), None if it is not paired, or a tuple of (edge index, t) for its partner if it is paired with another split/tag, index of edge in origin LineGraph, or None if no origin.)"},
 {"get_point", (PyCFunction)LineGraph_get_point_py, METH_VARARGS, "Given an edge index and a t value returns the details of that point, a tuple: (x, y, u, v, w, radius, density, weight). Note that radius, density and weight are all subject to linear interpolation."},
 {"get_segment", (PyCFunction)LineGraph_get_segment_py, METH_VARARGS, "Given an edge and t value returns the segment integer that is assigned to it. If the segmentation is not valid it is recalculated."},
 {"get_segs", (PyCFunction)LineGraph_get_segs_py, METH_VARARGS, "Returns a numpy array, indexed by vertex number, that gives the segment index the given vertex is in. Note that a lone vertex, with no connected edges, will have a negative number assigned, as segments are only assigned to edges."},
 
 {"vertex_to_edges", (PyCFunction)LineGraph_vertex_to_edges_py, METH_VARARGS, "Given the index of a vertex returns a list of edge indices that are connected to it. The indices are actually tuples (index, False if the edge starts at the vertex, True if it ends at the vertex.)."},
 
 {"add_split", (PyCFunction)LineGraph_add_split_tag_py, METH_VARARGS, "Given the index of an edge and a t value adds a split at that point."},
 {"add_tag", (PyCFunction)LineGraph_add_split_tag_py, METH_VARARGS, "Given the index of an edge, a t value  and a string adds a tag at that point."},
 {"add_link", (PyCFunction)LineGraph_add_link_py, METH_VARARGS, "Given 4 or 5 parameters - (index of edge a, t value of edge a, index of edge b, t value of edge b [, tag string]). It creates a link that links the two locations, with an optional tag. Note that if the tag is not defined the link will be followed when growing segments."},
 {"rem", (PyCFunction)LineGraph_rem_py, METH_VARARGS, "Given the index of an edge and a t value this removes the closest entity, be it split, tag or a link."},
 
 {"vertex_stats", (PyCFunction)LineGraph_vertex_stats_py, METH_VARARGS, "Returns a 4-tuple: (Number of vertices with no edges, number of vertices with 1 edge, number of vertices with 2 edges, number of vertices with 3 or more edges."},
 {"chains", (PyCFunction)LineGraph_chains_py, METH_VARARGS, "Returns a list of lists, where each inner list is the indices of vertices that form a chain of vertices, without any branches (The vertex at the start and end will be branches of some kind, unless its a loop.)."},
 {"get_tails", (PyCFunction)LineGraph_get_tails_py, METH_VARARGS, "Returns a list of the indices of the vertices that only have one edge, i.e. the ends of lines."},
 {"get_tags", (PyCFunction)LineGraph_get_tags_py, METH_VARARGS, "Returns a list of all tags in the LineGraph if no parameters are provided, or if an integer is provided all tags for a specific segment. An error will be thrown if the segments have not been calculated when a segment number is provided. Note that this does not include the splits, but does include tagged links, though not bare links. Each tag is represented as a tuple, for normal tags its (tag string, edge index, t), for tagged links its (tag string, edge index 1, t 1, edge index 2, t 2). In segment mode the tagged link might have one end outside the segment."},
 {"get_splits", (PyCFunction)LineGraph_get_splits_py, METH_VARARGS, "Returns a list of all splits in the LineGraph if no parameters are provided, or if an integer is provided all splits for a specific segment, including those that delinate it. An error will be thrown if the segments have not been calculated when a segment number is provided. This also includes links without tags. Returns a list of tuples - each split gets: (edge #, t, dir), whilst each link gets: (edge # 1, t 1, edge # 2, t 2). For a split dir is the direction along the edge of the segment - -1 if the segment is in the negative direction, 1 if its in the positive direction, 0.0 if both sides belong to the segment."},
 
 {"between", (PyCFunction)LineGraph_between_py, METH_VARARGS, "Given a location, as an edge index then a t value along that edge this returns the tags its between as two tuples within another: ((before distance, before text, before edge, before t), (after distance, after text, after edge, after t)) - before distance how far you have to travel along the graph to get to the closest tag, before text is the string of the closest tag in the negative t direction, before edge its edge and before t its position on that edge. after is the same again except going in the positive t direction. Internally it actually uses a depth first search, so be warned that it can be very slow. Can return None instead of a tuple for an entry if their are no tags in a given direction. Will follow links, counting them as length 0. Note that this method does predispose a valid segmentation. A third optional parameter, which defaults to 1024.0, is a distance limit - it will stop recursing after this depth to avoid infinite loops (side effect of inefficient implimentation)"},
 
 {"chain_feature", (PyCFunction)LineGraph_chain_feature_py, METH_VARARGS, "Given a bin count (Defaults to 8) returns a feature vector (Length related to bin count, currently 4*) that represents the line graph, under the assumption that it is a chain, with the vertices in sequence. A second optional parameter multiplies all the radii before their inclusion in the feature vector, so you can account for different width impliments. Third optional parameter does density."},
 {"features", (PyCFunction)LineGraph_features_py, METH_KEYWORDS | METH_VARARGS, "Calculates features for every vertex in the line graph - returns an array <# vertices> X <feature vector length> - it can potentially be quite large. Based on a random walk of the line graph from each starting point, and the probability of the relationship between the destination and origin - the feature is a histogram over the quantised space of relationships with the sqrt trick applied. The feature vector and its length depends on the parameters of the request: (dir_travel = 4.0 - The distance traveled to assign an orientation along a chain; travel_max = 32.0 - Maximum distance to travel along the line graph for the random walk; travel_bins = 4 - Number of bins to use for travel distance; travel_ratio = 0.75 - How much smaller the one nearer distance bin is than its neigbour, so that the bins are smaller nearer the origin; pos_bins = 3 - Number of bins to split relative position in; pos_ratio = 0.75 - Same as travel_ratio, but for the relative position; radius_bins = 2 - Number of bins to use for the radius difference; density_bins = 2 - Number of bins to use for the density difference; rec_depth = 12 - If it hits this many junctions, such that it has to search down all paths of each it swicthes from diffusion (perfect answer) to an actual random walk (answer with error) to save on computation. Note that 2^rec_depth is the number of potential branches explored, for each path leaving from the current vertex.)."},
 
 {"pos", (PyCFunction)LineGraph_pos_py, METH_VARARGS, "Returns an array [vertex, {0=x, 1=y}] - the position of every vertex in the data structure. You can optionally pass in a 3x3 homography, which will be applied to the vertices before they are returned."},
 
 {"adjacent", (PyCFunction)LineGraph_adjacent_py, METH_VARARGS, "Given a segment index this returns a list of adjacent segments, as tuples (seg #, edge #, t). If there are multiple adjacencies with a segment then they are all included, and seg # can be repeated. Tagged links are also included as adjacencies, except they get a longer tuple: (seg #, in seg edge #, in seg t, tag, other seg edge #, other seg t)."},
 {"merge", (PyCFunction)LineGraph_merge_py, METH_VARARGS, "Given two segment indices this removes all splits that directly seperate them, returning how many splits it has returned. If the return is greater than 0 then the segmentation will be marked as invalid."},
 
 {"smooth", (PyCFunction)LineGraph_smooth_py, METH_VARARGS, "Smooths the positions, radii, density and weights of all vertices that have precisly two neighbours. A simple interpolation with the average of the neighbours parameters. Two optional parameters: The first is a strength, from 0 to 1, of each iteration, the second how many iterations to do. They both default to 1."},
 {"transform", (PyCFunction)LineGraph_transform_py, METH_VARARGS, "Given a homography, as a 3x3 numpy matrix (float or double), this applies it to the LineGraph. Note that by default it does not adjust the radius parameter - if you want this pass True as its second parameter."},
 {"scale", (PyCFunction)LineGraph_scale_py, METH_VARARGS, "Allows you to scale the radius (1st parameter) and density (2nd parameter) by an arbitrary multipland."},
 {"blend", (PyCFunction)LineGraph_blend_py, METH_VARARGS, "Given another LineGraph this adjusts both this and the others vertex positions and radii such that they overlap. In addition to the other LineGraph you also provide a list of tuples: (this index, weight, other index), each being vertices to match, with weight indicating the bias of linear interpolation - 0.0 for the this index position, 1.0 for the other index position. These match tuples must be provided for all vertices that don't have 2 edges, so the weight can be blended along the edges linking them. The weight values of the LineGraphs are set by the provided and interpolated weights, with this being one minus those of other. A further optional parameter allows you to control the interpolation - False for linear, True for biased with a cosine curve, which produces a softer change in perceived velocity."},
 {"morph_to", (PyCFunction)LineGraph_morph_to_py, METH_VARARGS, "Given another line graph, a list of vertices in it and a weight parameter this morphs the current line graph to the given one, assuming the vertices in this line graph form a linear chain, as do the list of vertices in the other line graph. Basically it is designed to work with the chains and from_vertices methods. Chains don't have to be the same length, and the third parameter, the weight, defaults to 1, for 100% morphed."},
 
 {"within", (PyCFunction)LineGraph_within_py, METH_VARARGS, "Given 4 floating point values (min x, max x, min y, max y) this returns all the edges within that region. Return value is a tuple of slices, where the edges indexed by each slice constitute the set."},
 {"nearest", (PyCFunction)Linegraph_nearest_py, METH_VARARGS, "Given 2 floating point values x and y this returns the closest point in the line graph to this coordinate. The return is (distance, edge index, t value), or None if you have been silly enough to call this on a graph with no edges."},
 {"intersect", (PyCFunction)LineGraph_intersect_py, METH_VARARGS, "Given 4 floating point values, representing the start and end points of a line segment, this intersects it with the edges in the line graph, and returns all intersections. Return is a tuple, where each entry is an intersection. Each intersection is represented as a 3-tuple - an edge index, a t-value for where on that edge the intersection occurs and finally a t-value for the provided line segment."},
 {"intersect_links", (PyCFunction)LineGraph_intersect_links_py, METH_VARARGS, "Given 4 floating point values, representing the start and end points of a line segment, this intersects it with the links in the line graph, and returns all intersections. Return is a list, where each entry is an intersection. Each intersection is represented as a 7-tuple: (t value for input line, edge of link start, t value of link start, edge of link end, t value of link end, t value for point along link of intersection, tag of link, or None if it does not exist.)"},
 {NULL}
};



static PyTypeObject LineGraphType =
{
 PyObject_HEAD_INIT(NULL)
 0,                                /*ob_size*/
 "line_graph_c.LineGraph",         /*tp_name*/
 sizeof(LineGraph),                /*tp_basicsize*/
 0,                                /*tp_itemsize*/
 (destructor)LineGraph_dealloc_py, /*tp_dealloc*/
 0,                                /*tp_print*/
 0,                                /*tp_getattr*/
 0,                                /*tp_setattr*/
 0,                                /*tp_compare*/
 0,                                /*tp_repr*/
 0,                                /*tp_as_number*/
 0,                                /*tp_as_sequence*/
 0,                                /*tp_as_mapping*/
 0,                                /*tp_hash */
 0,                                /*tp_call*/
 0,                                /*tp_str*/
 0,                                /*tp_getattro*/
 0,                                /*tp_setattro*/
 0,                                /*tp_as_buffer*/
 Py_TPFLAGS_DEFAULT,               /*tp_flags*/
 "Provides a graph, with 2D coordinates attached, so it can be visualised and interacted with in terms of nearest point/intersection/region. Each vertex has a location, radius and density attached, and they are connected together by edges. Uses a winged half-edge structure. Includes the ability to tag locations with text strings, split edges (Without actually splitting them - it can be undone.) and link otherwise disparate parts. Edge splits define subgraphs, which can be extracted to get another LineGraph, getting all tags etc. Links back to the original LineGraph are included. File i/o is also supported, alongside the ability to build a line graph from various sources and smooth it as needed, plus lots of other capabilities.", /* tp_doc */
 0,                                /* tp_traverse */
 0,                                /* tp_clear */
 0,                                /* tp_richcompare */
 0,                                /* tp_weaklistoffset */
 0,                                /* tp_iter */
 0,                                /* tp_iternext */
 LineGraph_methods,                /* tp_methods */
 LineGraph_members,                /* tp_members */
 0,                                /* tp_getset */
 0,                                /* tp_base */
 0,                                /* tp_dict */
 0,                                /* tp_descr_get */
 0,                                /* tp_descr_set */
 0,                                /* tp_dictoffset */
 0,                                /* tp_init */
 0,                                /* tp_alloc */
 LineGraph_new_py,                 /* tp_new */
};



static PyMethodDef line_graph_c_methods[] =
{
 {NULL}
};



#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC initline_graph_c(void)
{
 PyObject * mod = Py_InitModule3("line_graph_c", line_graph_c_methods, "Provides a data structure for representing a 2D graph of lines with lots of attached information.");
 import_array();

 if (PyType_Ready(&LineGraphType) < 0) return;
 
 Py_INCREF(&LineGraphType);
 PyModule_AddObject(mod, "LineGraph", (PyObject*)&LineGraphType);
}
