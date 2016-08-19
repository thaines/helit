// Copyright 2016 Tom SF Haines

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.

// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "line_graph/line_graph_c.h"
#include "composite_c.h"

#define USE_MAXFLOW_C
#include "graph_cuts/maxflow_c.h"



void Composite_new(Composite * this)
{
 this->height = 0;
 this->width = 0;
 this->data = NULL;
 
 this->storage = NULL;
 this->new_pixel = NULL;
 
 this->bg.r = 1.0;
 this->bg.g = 1.0;
 this->bg.b = 1.0;
 this->bg.a = 0.0;
 
 this->next_part = 0;
}

void Composite_dealloc(Composite * this)
{
 free(this->data);
 
 this->height = 0;
 this->width = 0;
 this->data = NULL;
 
 while (this->storage!=NULL)
 {
  PixelBlock * to_die = this->storage;
  this->storage = this->storage->next;
  free(to_die);
 }
 this->new_pixel = NULL;
}


static PyObject * Composite_new_py(PyTypeObject * type, PyObject * args, PyObject * kwds)
{
 // Allocate the object...
  Composite * self = (Composite*)type->tp_alloc(type, 0);

 // On success construct it...
  if (self!=NULL) Composite_new(self);

 // Return the new object...
  return (PyObject*)self;
}

static void Composite_dealloc_py(Composite * self)
{
 Composite_dealloc(self);
 self->ob_type->tp_free((PyObject*)self);
}



// Reset method, that allows you to set the width and height of the image; empties the data structure at the same time...
void Composite_set_size(Composite * this, int width, int height)
{
 // Clean up the current contents...
  Composite_dealloc(this);
  
 // Setup the Pixel list array...
  this->height = height;
  this->width = width;
  
  size_t size = height * width * sizeof(Pixel*);
  this->data = (Pixel**)malloc(size);
  memset(this->data, 0, size);
}


static PyObject * Composite_set_size_py(Composite * self, PyObject * args)
{
 // Extract the parameters...
  int width;
  int height;
  if (!PyArg_ParseTuple(args, "ii", &width, &height)) return NULL;
  
 // Call through to the method...
  Composite_set_size(self, width, height);
  
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}



// Setter and getter for the background colour...
static PyObject * Composite_set_bg_py(Composite * self, PyObject * args)
{
 // Extract the parameters...
  self->bg.a = 1.0;
  if (!PyArg_ParseTuple(args, "fff|f", &self->bg.r, &self->bg.g, &self->bg.b, &self->bg.a)) return NULL;
  
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject * Composite_get_bg_py(Composite * self, PyObject * args)
{
 return Py_BuildValue("(f,f,f,f)", self->bg.r, self->bg.g, self->bg.b, self->bg.a);
}



// Returns a new pixel - handles creating new memory blocks etc as required.
Pixel * Composite_new_pixel(Composite * this)
{
 // Check if the list of unused pixels is null, if so fill it up...
  if (this->new_pixel==NULL)
  {
   PixelBlock * npb = (PixelBlock*)malloc(sizeof(PixelBlock) + this->width * this->height * sizeof(Pixel));
   npb->next = this->storage;
   this->storage = npb;
   
   int i;
   for (i=(this->width*this->height-1); i>=0; i--)
   {
    this->storage->data[i].next = this->new_pixel;
    this->new_pixel = &this->storage->data[i];
   }
  }
 
 // Get the first entry from the list of unused Pixels...
  Pixel * ret = this->new_pixel;
  this->new_pixel = this->new_pixel->next;
  return ret;
}



// Draws a line to the compositing system, recording UV coordinates and weight only...
// s_ for start, e_ for end. x and y are the coordinates, r the radius, br the blending radius, w the weight.
// hg is a 3x3 homography that converts from x,y to uv coordinates.
void Composite_draw_line(Composite * this, int part, float sx, float sy, float sr, float sbr, float sw, float ex, float ey, float er, float ebr, float ew, float * hg)
{
 // First calculate the bounding box in which the line has influence...
  int min_x = (int)floor(((sx-sr-sbr)<(ex-er-ebr)) ? (sx-sr-sbr) : (ex-er-ebr));
  int max_x = (int)ceil(((sx+sr+sbr)>(ex+er+ebr)) ? (sx+sr+sbr) : (ex+er+ebr));
  int min_y = (int)floor(((sy-sr-sbr)<(ex-er-ebr)) ? (sy-sr-sbr) : (ey-er-ebr));
  int max_y = (int)ceil(((sy+sr+sbr)>(ex+er+ebr)) ? (sy+sr+sbr) : (ey+er+ebr));
 
 // Clamp the bounding box to the image size, if its outside return...
  if (min_x<0)
  {
   if (max_x<0) return;
   min_x = 0; 
  }
  
  if (max_x>=this->width)
  {
   if (min_x>=this->width) return;
   max_x = this->width - 1;
  }
  
  if (min_y<0)
  {
   if (max_y<0) return;
   min_y = 0; 
  }
  
  if (max_y>=this->height)
  {
   if (min_y>=this->height) return;
   max_y = this->height - 1;
  }
  
 // Calculate line paramters, so we only do so once...
  float nx = ex - sx;
  float ny = ey - sy;
  float l = sqrt(nx*nx + ny*ny);
  if (l<1e-6) return; // Its not really a line!
  nx /= l;
  ny /= l;
  
 // Iterate the bounding box we just defined...
  int y, x;
  for (y=min_y; y<=max_y; y++)
  {
   for (x=min_x; x<=max_x; x++)
   {
    float ax = x + 0.5;
    float ay = y + 0.5;
     
    // Find its closest point on the line, extended to infinity...
     float t = nx * (ax - sx) + ny * (ay - sy);
    
    // Clamp and find out its closest point on the finite line...
     float ct;
     float ix;
     float iy;
     float ir;
     float ibr;
     float iw;
     
     if (t<0.0)
     {
      ct = 0.0;
      ix = sx;
      iy = sy;
      ir = sr;
      ibr = sbr;
      iw = sw;
     }
     else
     {
      if (t>l)
      {
       ct = l;
       ix = ex;
       iy = ey;
       ir = er;
       ibr = ebr;
       iw = ew;
      }
      else
      {
       ct = t;
       ix = sx + nx*ct;
       iy = sy + ny*ct;
       
       float tt = t/l;
       ir = sr + tt * (er - sr);
       ibr = sbr + tt * (ebr - sbr);
       iw = sw + tt * (ew - sw);
      }
     }
    
    // Find out its distance to the finite line...
     float dx = ax - ix;
     float dy = ay - iy;
     float d = sqrt(dx*dx + dy*dy);
    
    // If the distance is less then the radius we need to draw, otherwise move on to the next pixel...
     if (ir+ibr<=d) continue;
     
    // Down-weight in the biased zone...
     float alpha_mult = 1.0;
     if (d>ir)
     {
      alpha_mult = 1.0 - (d - ir) / ibr;
      iw *= alpha_mult;
     }
    
    // Two scenarios - top entry already belongs to this part, in which case merge, otherwise create a new Pixel entry and drop the details in... 
     Pixel * targ = this->data[y*this->width + x];
     if ((targ==NULL)||(targ->part!=part))
     {
      // Create a new Pixel...
       Pixel * np = Composite_new_pixel(this);
       np->next = targ;
       this->data[y*this->width + x] = np;
       
       np->c.r = 1.0;
       np->c.g = 1.0;
       np->c.b = 1.0;
       np->c.a = alpha_mult;
       
       np->part = part;
       
       np->mass = iw;
       float div = hg[6] * ax + hg[7] * ay + hg[8];
       np->u = (hg[0] * ax + hg[1] * ay + hg[2]) / div;
       np->v = (hg[3] * ax + hg[4] * ay + hg[5]) / div;
       np->w = iw;
     }
     else
     {
      // Update the existing Pixel...
       float div = hg[6] * ax + hg[7] * ay + hg[8];
       float u = (hg[0] * ax + hg[1] * ay + hg[2]) / div;
       float v = (hg[3] * ax + hg[4] * ay + hg[5]) / div;
       
       targ->mass += iw;
       
       targ->c.a += iw * (alpha_mult - targ->c.a) / targ->mass;
       targ->u += iw * (u - targ->u) / targ->mass;
       targ->v += iw * (v - targ->v) / targ->mass;
       targ->w += iw * (iw - targ->w) / targ->mass;
     }
   }
  }
}

// Matrix multiplication, for 3x3 matrices - does a * b = out...
void matrix_mult_33(float * a, float * b, float * out)
{
 int r, c, i;
 for (r=0; r<3; r++)
 {
  for (c=0; c<3; c++)
  {
   int oi = r*3 + c;
   out[oi] = 0.0;
   for (i=0; i<3; i++) out[oi] += a[r*3 + i] * b[i*3 + c];
  }
 }
}

// Draws a LineGraph object, assigning it a part number (returned) so its can be converted from UV coordinates to actual pixel colours with a paint call...
// (bias increases the radius of the lines, but in the uv coordinate system.)
int Composite_draw_line_graph(Composite * this, LineGraph * lg, float bias, float stretch)
{
 // Assign a part number...
  int ret = this->next_part;
  this->next_part += 1;
  
 // Loop and draw each edge in turn...
  int i;
  float hg1[9];
  float hg2[9];
  float hg3[9];
  
  for (i=0; i<lg->edge_count; i++)
  {
   Vertex * s = lg->edge[i].neg.dest;
   Vertex * e = lg->edge[i].pos.dest;
   
   // Safety for the w values - avoids stretching, which never looks good...
    float sw = (s->w>1.0) ? s->w : 1.0;
    float ew = (e->w>1.0) ? e->w : 1.0;
    
   // Calculate homography to represent the UV coordinates...
    // Start by offsetting to start the edge at (0,0)...
     hg1[0] = 1.0; hg1[1] = 0.0; hg1[2] = -s->x;
     hg1[3] = 0.0; hg1[4] = 1.0; hg1[5] = -s->y;
     hg1[6] = 0.0; hg1[7] = 0.0; hg1[8] = 1.0;
    
    // Rotate so the edge is on the y=0 line...
     float nx = e->x - s->x;
     float ny = e->y - s->y;
     float rl = sqrt(nx*nx + ny*ny);
    
     nx /= rl;
     ny /= rl;
    
     hg2[0] = nx;  hg2[1] = ny;  hg2[2] = 0.0;
     hg2[3] = -ny; hg2[4] = nx;  hg2[5] = 0.0;
     hg2[6] = 0.0; hg2[7] = 0.0; hg2[8] = 1.0;
    
     matrix_mult_33(hg2, hg1, hg3);
   
    // Multiply in the required scale, in both dimensions...
     float nu = e->u - s->u;
     float nv = e->v - s->v;
     float tl = sqrt(nu*nu + nv*nv);
    
     nu /= tl;
     nv /= tl;
    
     float r = 0.5 * (s->radius + e->radius);
     float w = 0.5 * (sw + ew);
    
     float sal = tl / rl;
     float sol = w / r;
    
     hg3[0] *= sal; hg3[1] *= sal; hg3[2] *= sal;
     hg3[3] *= sol; hg3[4] *= sol; hg3[5] *= sol;
   
    // Rotate to match the line of the u,v line...
     hg2[0] = nu;  hg2[1] = -nv; hg2[2] = 0.0;
     hg2[3] = nv;  hg2[4] = nu;  hg2[5] = 0.0;
     hg2[6] = 0.0; hg2[7] = 0.0; hg2[8] = 1.0;
    
     matrix_mult_33(hg2, hg3, hg1);
   
    // Transform so we are in the correct offset...
     hg2[0] = 1.0; hg2[1] = 0.0; hg2[2] = s->u;
     hg2[3] = 0.0; hg2[4] = 1.0; hg2[5] = s->v;
     hg2[6] = 0.0; hg2[7] = 0.0; hg2[8] = 1.0;
     
     matrix_mult_33(hg2, hg1, hg3);
     
   // Calculate weight biases, to reduce the weight when the texture is stretched...
    float wb = tl / rl;
    if (wb>1.0) wb = 1.0;
    wb = stretch * wb + (1.0 - stretch);
   
   // Call the actual draw method...
    Composite_draw_line(this, ret, s->x, s->y, s->radius, bias/sw, s->weight * wb, e->x, e->y, e->radius, bias/ew, e->weight * wb, hg3);
  }
 
 // Return the assigned part number...
  return ret;
}


static PyObject * Composite_draw_line_graph_py(Composite * self, PyObject * args)
{
 // Extract the parameters...
  LineGraph * lg;
  float bias = 0.0;
  float stretch = 0.0;
  if (!PyArg_ParseTuple(args, "O|ff", &lg, &bias, &stretch)) return NULL; // Note: Total lack of typechecking - hard to do without significant complexity.
  
 // Call through to the function... 
  int ret = Composite_draw_line_graph(self, lg, bias, stretch);
  
 // Return the part number...
  return Py_BuildValue("i", ret);
}



void Composite_paint_test_pattern(Composite * this, int part)
{
 int y, x;
 for (y=0; y<this->height; y++)
 {
  for (x=0; x<this->width; x++)
  {
   Pixel * targ = this->data[y*this->width + x];
   if ((targ!=NULL)&&(targ->part==part))
   {
    targ->c.r *= 0.5 + 0.3*sin(targ->u) + 0.2*sin(0.1*targ->v);
    targ->c.g *= 0.5;
    targ->c.b *= 0.5 + 0.3*sin(targ->v) + 0.2*sin(0.1*targ->u);
    //targ->c.a *= 1.0;
   }
  }
 }
}


static PyObject * Composite_paint_test_pattern_py(Composite * self, PyObject * args)
{
 // Extract the part #...
  int part = self->next_part-1;
  if (!PyArg_ParseTuple(args, "|i", &part)) return NULL;
  
 // Do the work...
  Composite_paint_test_pattern(self, part);
  
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}



// Textures the composite, coordinates of a pixel are (data + y*y_stride + x*x_stride), with the first byte blue, the second green and the third red. The dim values are to clamp if a coordinate goes outside. No interpolation. If inc_alpha is not 0 the fourth byte is alpha (pre-multiplied), which is then used...
void Composite_paint_texture_nearest(Composite * this, int part, unsigned char * data, int y_dim, int x_dim, int y_stride, int x_stride, int inc_alpha)
{
 int y, x;
 for (y=0; y<this->height; y++)
 {
  for (x=0; x<this->width; x++)
  {
   Pixel * targ = this->data[y*this->width + x];
   if ((targ!=NULL)&&(targ->part==part))
   {
    int sy = floor(targ->v+0.5);
    if (sy<0) sy = 0;
    if (sy>=y_dim) sy = y_dim-1;
    
    int sx = floor(targ->u+0.5);
    if (sx<0) sx = 0;
    if (sx>=x_dim) sx = x_dim-1;
    
    unsigned char * pixel = data + sy * y_stride + sx * x_stride;
    
    if (inc_alpha!=0)
    {
     if (pixel[3]==0)
     {
      targ->c.r = 0.0;
      targ->c.g = 0.0;
      targ->c.b = 0.0;
      targ->c.a = 0.0;
     }
     else
     {
      float alpha = pixel[3] / 255.0;
      targ->c.r *= pixel[2] / (255.0 * alpha);
      targ->c.g *= pixel[1] / (255.0 * alpha);
      targ->c.b *= pixel[0] / (255.0 * alpha);
      targ->c.a *= alpha;
     }
    }
    else
    {
     targ->c.r *= pixel[2] / 255.0;
     targ->c.g *= pixel[1] / 255.0;
     targ->c.b *= pixel[0] / 255.0;
     //targ->c.a *= 1.0;
    }
   }
  }
 }
}


static PyObject * Composite_paint_texture_nearest_py(Composite * self, PyObject * args)
{
 // Extract the part # and numpy array...
  PyArrayObject * image;
  int part = self->next_part-1;
  if (!PyArg_ParseTuple(args, "O!|i", &PyArray_Type, &image, &part)) return NULL;
  
 // Some error checking...
  if (image->nd!=3)
  {
   PyErr_SetString(PyExc_TypeError, "Image numpy array must have 3 dimensions - height, width then colour channels.");
   return NULL;
  }
  
  if (image->descr->kind!='u' || image->descr->elsize!=sizeof(char))
  {
   PyErr_SetString(PyExc_TypeError, "Image must be made of uint8.");
   return NULL;
  }
  
 // Do the work...
  Composite_paint_texture_nearest(self, part, (unsigned char*)(void*)image->data, image->dimensions[0], image->dimensions[1], image->strides[0], image->strides[1], (image->dimensions[2]>3) ? 1 : 0);
  
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}



// Linear version of above...
void Composite_paint_texture_linear(Composite * this, int part, unsigned char * data, int y_dim, int x_dim, int y_stride, int x_stride, int inc_alpha)
{
 int y, x, i;
 for (y=0; y<this->height; y++)
 {
  for (x=0; x<this->width; x++)
  {
   Pixel * targ = this->data[y*this->width + x];
   if ((targ!=NULL)&&(targ->part==part))
   {
    int sy = floor(targ->v);
    float ty = targ->v - sy;
    int ey = sy + 1;
    
    if (sy<0) sy = 0;
    if (sy>=y_dim) sy = y_dim-1;
    if (ey<0) ey = 0;
    if (ey>=y_dim) ey = y_dim-1;
    
    int sx = floor(targ->u);
    float tx = targ->u - sx;
    int ex = sx + 1;
    
    if (sx<0) sx = 0;
    if (sx>=x_dim) sx = x_dim-1;
    if (ex<0) ex = 0;
    if (ex>=x_dim) ex = x_dim-1;
    
    
    unsigned char * pss = data + sy * y_stride + sx * x_stride;
    unsigned char * pse = data + sy * y_stride + ex * x_stride;
    unsigned char * pes = data + ey * y_stride + sx * x_stride;
    unsigned char * pee = data + ey * y_stride + ex * x_stride;
    float pixel[4];
    
    for (i=0; i<((inc_alpha!=0)?4:3); i++)
    {
     pixel[i]  = pss[i] * (1.0-ty) * (1.0-tx);
     pixel[i] += pse[i] * (1.0-ty) * tx;
     pixel[i] += pes[i] * ty       * (1.0-tx);
     pixel[i] += pee[i] * ty       * tx;
     pixel[i] /= 255.0;
    }

    if (inc_alpha!=0)
    {
     if (pixel[3]<1e-3)
     {
      targ->c.r = 0.0;
      targ->c.g = 0.0;
      targ->c.b = 0.0;
      targ->c.a = 0.0;
     }
     else
     {
      targ->c.r *= pixel[2] / pixel[3];
      targ->c.g *= pixel[1] / pixel[3];
      targ->c.b *= pixel[0] / pixel[3];
      targ->c.a *= pixel[3];
     }
    }
    else
    {
     targ->c.r *= pixel[2];
     targ->c.g *= pixel[1];
     targ->c.b *= pixel[0];
     //targ->c.a *= 1.0;
    }
   }
  }
 }
}


static PyObject * Composite_paint_texture_linear_py(Composite * self, PyObject * args)
{
 // Extract the part # and numpy array...
  PyArrayObject * image;
  int part = self->next_part-1;
  if (!PyArg_ParseTuple(args, "O!|i", &PyArray_Type, &image, &part)) return NULL;
  
 // Some error checking...
  if (image->nd!=3)
  {
   PyErr_SetString(PyExc_TypeError, "Image numpy array must have 3 dimensions - height, width then colour channels.");
   return NULL;
  }
  
  if (image->descr->kind!='u' || image->descr->elsize!=sizeof(char))
  {
   PyErr_SetString(PyExc_TypeError, "Image must be made of uint8.");
   return NULL;
  }
  
 // Do the work...
  Composite_paint_texture_linear(self, part, (unsigned char*)(void*)image->data, image->dimensions[0], image->dimensions[1], image->strides[0], image->strides[1], (image->dimensions[2]>3) ? 1 : 0);
  
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}



// Acts like Composite_paint_texture_nearest, except instead of painting it calculates a cost function based on the colourmetric distance between its pixel colours and those already composited. Whilst doing this it deletes the entries, so its as though it was never drawn. Inputs are identical to Composite_paint_texture_nearest. Alpha is used to scale the cost...
float Composite_cost_texture_nearest(Composite * this, int part, unsigned char * data, int y_dim, int x_dim, int y_stride, int x_stride, int inc_alpha)
{
 float ret = 0.0;
 int y, x;
 
 for (y=0; y<this->height; y++)
 {
  for (x=0; x<this->width; x++)
  {
   Pixel * targ = this->data[y*this->width + x];
   if ((targ!=NULL)&&(targ->part==part))
   {
    // Calculate the pixel to use...
     int sy = floor(targ->v+0.5);
     if (sy<0) sy = 0;
     if (sy>=y_dim) sy = y_dim-1;
    
     int sx = floor(targ->u+0.5);
     if (sx<0) sx = 0;
     if (sx>=x_dim) sx = x_dim-1;
    
     unsigned char * pixel = data + sy * y_stride + sx * x_stride;
    
     // Update the cost function, with the distance from the closest value already in position...
      float weight = 0.0;
      float mean[3] = {0.0, 0.0, 0.0};
      Pixel * comp = targ->next;
      
      while (comp!=NULL)
      {
       // Factor in its distance from every pixel thus far...
        if (inc_alpha!=0)
        {
         weight += comp->c.a;

         mean[0] += comp->c.a * (comp->c.r - mean[0]) / weight;
         mean[1] += comp->c.a * (comp->c.g - mean[1]) / weight;
         mean[2] += comp->c.a * (comp->c.b - mean[2]) / weight;
        }
        else
        {
         weight += 1.0;
         
         mean[0] += (comp->c.r - mean[0]) / weight;
         mean[1] += (comp->c.g - mean[1]) / weight;
         mean[2] += (comp->c.b - mean[2]) / weight;
        }
      
       comp = comp->next; 
      }
      
      if ((weight>0.5)&&((inc_alpha==0)||(pixel[3]!=0)))
      {
       float a = (inc_alpha!=0) ? (pixel[3] / 255.0) : 1.0;
       float dr = (pixel[2] / (255.0*a)) - mean[0];
       float dg = (pixel[1] / (255.0*a)) - mean[1];
       float db = (pixel[0] / (255.0*a)) - mean[2];
          
       ret += a * sqrt(dr*dr + dg*dg + db*db);
      }
    
    // Remove the pixel value - we don't want to use it again...
     this->data[y*this->width + x] = targ->next;
     targ->next = this->new_pixel;
     this->new_pixel = targ;
   }
  }
 }
 
 return ret;
}


static PyObject * Composite_cost_texture_nearest_py(Composite * self, PyObject * args)
{
 // Extract the part # and numpy array...
  PyArrayObject * image;
  int part = self->next_part-1;
  if (!PyArg_ParseTuple(args, "O!|i", &PyArray_Type, &image, &part)) return NULL;
  
 // Some error checking...
  if (image->nd!=3)
  {
   PyErr_SetString(PyExc_TypeError, "Image numpy array must have 3 dimensions - height, width then colour channels.");
   return NULL;
  }
  
  if (image->descr->kind!='u' || image->descr->elsize!=sizeof(char))
  {
   PyErr_SetString(PyExc_TypeError, "Image must be made of uint8.");
   return NULL;
  }
  
 // Do the work...
  float cost = Composite_cost_texture_nearest(self, part, (unsigned char*)(void*)image->data, image->dimensions[0], image->dimensions[1], image->strides[0], image->strides[1], (image->dimensions[2]>3) ? 1 : 0);
  
 // Return the cost...
  return Py_BuildValue("f", cost);
}



// Adds alpha multiplied by the given weight to each pixel in the image...
void Composite_inc_weight_alpha(Composite * this, float weight)
{
 int y, x;
 
 for (y=0; y<this->height; y++)
 {
  for (x=0; x<this->width; x++)
  {
   Pixel * targ = this->data[y*this->width + x];
   while (targ!=NULL)
   {
    targ->w += targ->c.a * weight;
    targ = targ->next; 
   }
  }
 }
}


static PyObject * Composite_inc_weight_alpha_py(Composite * self, PyObject * args)
{
 // Extract the parameters...
  float weight = 0.0;
  if (!PyArg_ParseTuple(args, "|f", &weight)) return NULL;
 
 // Call the method...
  Composite_inc_weight_alpha(self, weight);
  
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}



void Composite_draw_pair(Composite * this, int part1, int part2, float weight)
{
 int y, x;
 
 for (y=0; y<this->height; y++)
 {
  for (x=0; x<this->width; x++)
  {
   // Find out if it has either of the parts...
    int has1 = 0;
    int has2 = 0;
    
    Pixel * targ = this->data[y*this->width + x];
    while (targ!=NULL)
    {
     if (targ->part==part1) has1 = 1;
     if (targ->part==part2) has2 = 1;
       
     targ = targ->next; 
    }
    
   // We only have to dance if we have one but not the other...
    if (has1!=has2)
    {
     if (has1==0)
     {
      Pixel * np = Composite_new_pixel(this);
      np->next = targ;
      this->data[y*this->width + x] = np;
       
      np->c.r = 1.0;
      np->c.g = 1.0;
      np->c.b = 1.0;
      np->c.a = 0.0;
       
      np->part = part1;
       
      np->mass = weight;
      np->u = 0.0;
      np->v = 0.0;
      np->w = weight;
     }
      
     if (has2==0)
     {
      Pixel * np = Composite_new_pixel(this);
      np->next = targ;
      this->data[y*this->width + x] = np;
       
      np->c.r = 1.0;
      np->c.g = 1.0;
      np->c.b = 1.0;
      np->c.a = 0.0;
       
      np->part = part2;
       
      np->mass = weight;
      np->u = 0.0;
      np->v = 0.0;
      np->w = weight;
     }
    }
  }
 }
}


static PyObject * Composite_draw_pair_py(Composite * self, PyObject * args)
{
 // Extract the parameters...
  int part1;
  int part2;
  float weight = 1.0;
  if (!PyArg_ParseTuple(args, "ii|f", &part1, &part2, &weight)) return NULL;
 
 // Call the method...
  Composite_draw_pair(self, part1, part2, weight);
  
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}



// Helper structures for below...
typedef struct PixelState PixelState;
typedef struct EdgeState EdgeState;

struct PixelState
{
 PixelState * next;
 int job;
 int pos;
 
 float diff; // Colour difference between pixel selections.
 float weight; // low->w - high->w - allows an edge to be created to bias membership.
 float low_l; // Luminence of low pixel.
 float high_l; // Luminence of high pixel.
 
 int low; // non zero if connected to the source (low segment).
 int high; // The above in opposite land.
};

struct EdgeState
{
 EdgeState * next;
 PixelState * from;
 PixelState * to;
 int pos;
};



int Composite_maxflow_select(Composite * this, MaxFlowAPI * mf, float edge_bias, float smooth_bias)
{
 // MaxFlow object, for fun and games...
  MaxFlow maxflow;
  mf->init(&maxflow, 32, 32); // Will almost certainly grow from these values, but good starting point.
  
 // Create a fun little data structure to allow us to quickly grow regions and find all pixels that we are considering... 
  int job_code = 0;
  int pixels = this->width * this->height;
  PixelState * pstate = (PixelState*)malloc(pixels * sizeof(PixelState));
  EdgeState  * estate =  (EdgeState*)malloc(pixels * 2 * sizeof(EdgeState));
  
  int i;
  for (i=0; i<pixels; i++) pstate[i].job = -1;
 
 // Loop until all cases have been resolved...
  int check_pixel = 0;
  while (1)
  {
   // First find a pixel with overlap...
    while ((check_pixel<pixels) && ((this->data[check_pixel]==NULL) || (this->data[check_pixel]->next==NULL))) check_pixel += 1;
    if (check_pixel==pixels) break;
    
   // Select two segments...
    int low_part = -1;
    int high_part = -1;
     
    Pixel * targ = this->data[check_pixel];
    while (targ)
    {
     high_part = low_part;
     low_part = targ->part;
     targ = targ->next; 
    }
    
   // Grow from the pixel, finding all pixels where overlap between the two selected segments exists...
    pstate[check_pixel].next = NULL;
    pstate[check_pixel].job = job_code;
    PixelState * to_check = pstate + check_pixel;
    
    PixelState * region = NULL;
    int vertex_count = 0;
        
    while (to_check!=NULL)
    {
     // Remove the pixel we are going to play with from the work queue...
      PixelState * tps = to_check;
      to_check = to_check->next;
      
     // Check if its a member - if so add it to the region...
      Pixel * low = NULL;
      Pixel * high = NULL;
      
      targ = this->data[tps - pstate];
      while (targ!=NULL)
      {
       if (targ->part==low_part)  low  = targ;
       if (targ->part==high_part) high = targ;
       targ = targ->next;
      }
      
      if ((low!=NULL)&&(high!=NULL))
      {
       tps->next = region;
       region = tps;
       tps->pos = vertex_count;
       ++vertex_count;
       
       float dr = low->c.r*low->c.a - high->c.r*high->c.a;
       float dg = low->c.g*low->c.a - high->c.g*high->c.a;
       float db = low->c.b*low->c.a - high->c.b*high->c.a;
       float da = low->c.a - high->c.a;
       tps->diff = sqrt(dr*dr + dg*dg + db*db + da*da);
       
       tps->low_l = (low->c.r + low->c.g + low->c.b) / 3.0;
       tps->high_l = (high->c.r + high->c.g + high->c.b) / 3.0;
      }
      else
      {
       tps->pos = -1; 
      }
      
     // Check its neighbours, updating their job value and adding them to the to_check work queue...
      if ((low!=NULL)&&(high!=NULL))
      {
       int y = (tps - pstate) / this->width;
       int x = (tps - pstate) % this->width;
       
       // +ve x...
        if ((x+1)<this->width)
        {
         PixelState * other = tps + 1;
         if (other->job!=job_code)
         {
          other->job = job_code;
          other->next = to_check;
          to_check = other;
         }
        }
       
       // -ve x...
        if (x>0)
        {
         PixelState * other = tps - 1;
         if (other->job!=job_code)
         {
          other->job = job_code;
          other->next = to_check;
          to_check = other;
         }
        }
       
       // +ve y...
        if ((y+1)<this->height)
        {
         PixelState * other = tps + this->width;
         if (other->job!=job_code)
         {
          other->job = job_code;
          other->next = to_check;
          to_check = other;
         }
        }
       
       // -ve y...
        if (y>0)
        {
         PixelState * other = tps - this->width;
         if (other->job!=job_code)
         {
          other->job = job_code;
          other->next = to_check;
          to_check = other;
         }
        }
      }
    }
    
    
   // Collate all the edges - check for a neighbour for each member of the disputed region...
    EdgeState * edges = NULL;
    int edge_count = 0;
    
    PixelState * tps = region;
    while (tps!=NULL)
    {
     int y = (tps - pstate) / this->width;
     int x = (tps - pstate) % this->width;
     
     if (x>0)
     {
      PixelState * other = tps - 1;
      if (other->pos>=0)
      {
       estate[edge_count].next = edges;
       edges = estate + edge_count;
       
       estate[edge_count].from = other;
       estate[edge_count].to = tps;
       estate[edge_count].pos = edge_count;
       
       ++edge_count;
      }
     }
     
     if (y>0)
     {
      PixelState * other = tps - this->width;
      if (other->pos>=0)
      {
       estate[edge_count].next = edges;
       edges = estate + edge_count;
       
       estate[edge_count].from = other;
       estate[edge_count].to = tps;
       estate[edge_count].pos = edge_count;
       
       ++edge_count;
      }
     }
     
     tps = tps->next; 
    }
    
    
   // Determine the source/sink links - occur whenever we have an edge that connects to a undisputed region - for each pixel check the 4-neighbourhood for such fixed region ownership...
    tps = region;
    int special_edges = 0;
    while (tps!=NULL)
    {
     tps->low = 0;
     tps->high = 0;
     
     int y = (tps - pstate) / this->width;
     int x = (tps - pstate) % this->width;
     
     if (x>0)
     {
      PixelState * other = tps - 1;
      if (other->pos<0)
      {
       targ = this->data[other - pstate];
       while (targ!=NULL)
       {
        if (targ->part==low_part) tps->low = 1;
        if (targ->part==high_part) tps->high = 1;
        targ = targ->next; 
       }
      }
     }
     
     if ((x+1)<this->width)
     {
      PixelState * other = tps + 1;
      if (other->pos<0)
      {
       targ = this->data[other - pstate];
       while (targ!=NULL)
       {
        if (targ->part==low_part) tps->low = 1;
        if (targ->part==high_part) tps->high = 1;
        targ = targ->next; 
       }
      }
     }

     if (y>0)
     {
      PixelState * other = tps - this->width;
      if (other->pos<0)
      {
       targ = this->data[other - pstate];
       while (targ!=NULL)
       {
        if (targ->part==low_part) tps->low = 1;
        if (targ->part==high_part) tps->high = 1;
        targ = targ->next; 
       }
      }
     }
     
     if ((y+1)<this->height)
     {
      PixelState * other = tps + this->width;
      if (other->pos<0)
      {
       targ = this->data[other - pstate];
       while (targ!=NULL)
       {
        if (targ->part==low_part) tps->low = 1;
        if (targ->part==high_part) tps->high = 1;
        targ = targ->next; 
       }
      }
     }
     
     if (tps->low!=0) special_edges += 1;
     if (tps->high!=0) special_edges += 1;
     
     tps = tps->next;
    }
    

   // Build a maxflow problem...
    mf->resize(&maxflow, vertex_count + 2, edge_count + special_edges);
    
    mf->set_source(&maxflow, vertex_count);
    mf->set_sink(&maxflow, vertex_count+1);
    
    EdgeState * te = edges;
    while (te!=NULL)
    {
     mf->set_edge(&maxflow, te->pos, te->from->pos, te->to->pos);
     
     float break_cost = 0.5 * (te->from->diff + te->to->diff);
     
     break_cost += smooth_bias * fabs(te->from->low_l - te->to->low_l);
     break_cost += smooth_bias * fabs(te->from->high_l - te->to->high_l);
     
     mf->cap_flow(&maxflow, te->pos, break_cost, break_cost);
     
     te = te->next;
    }
    
    tps = region;
    while (tps!=NULL)
    {
     if (tps->low!=0)
     {
      mf->set_edge(&maxflow, edge_count, tps->pos, vertex_count);
      float break_cost = tps->diff + edge_bias;
      mf->cap_flow(&maxflow, edge_count, break_cost, break_cost);
     
      ++edge_count; 
     }
     
     if (tps->high!=0)
     {
      mf->set_edge(&maxflow, edge_count, tps->pos, vertex_count+1);
      float break_cost = tps->diff + edge_bias;
      mf->cap_flow(&maxflow, edge_count, break_cost, break_cost);
     
      ++edge_count;        
     }
     
     tps = tps->next; 
    }
    
     
   // Solve it...
     mf->solve(&maxflow);
     
     
   // Update the composite with the decision for each pixel...
    tps = region;
    while (tps!=NULL)
    {
     int to_die = (mf->get_side(&maxflow, tps->pos)==1) ? low_part : high_part;
     int loc = tps - pstate;
     
     Pixel * dead;
     if (this->data[loc]->part==to_die)
     {
      dead = this->data[loc];
      this->data[loc] = this->data[loc]->next;
     }
     else
     {
      targ = this->data[loc];
      while (targ->next->part!=to_die) targ = targ->next;
      dead = targ->next;
      targ->next = targ->next->next;
     }
     
     dead = this->new_pixel;
     this->new_pixel = dead;
     
     targ = this->data[loc];
     while (targ!=NULL)
     {
      if ((targ->part==high_part)||(targ->part==low_part))
      {
       targ->part = this->next_part;
      }
      targ = targ->next;
     }
     
     tps = tps->next; 
    }
    
    this->next_part += 1;
   
   // Move to the next job...
    ++job_code;
  }
  
 // Clean up...
  free(estate);
  free(pstate);
  mf->deinit(&maxflow);
  
 return job_code;
}



static PyObject * Composite_maxflow_select_py(Composite * self, PyObject * args)
{
 // Extract the optional parameters...
  float edge_bias = 0.0;
  float smooth_bias = 0.0;
  
  if (!PyArg_ParseTuple(args, "|ff", &edge_bias, &smooth_bias)) return NULL;
  
  if (edge_bias<0.0) edge_bias = 0.0;
  if (smooth_bias<0.0) smooth_bias = 0.0;
 
 // Make sure the maxflow array is loaded...
  if (import_maxflow()!=0) return NULL;
 
 // Simply call through to the method...
  int solved = Composite_maxflow_select(self, maxflow, edge_bias, smooth_bias);
 
 // Return how many mincut problems have been solved...
  return Py_BuildValue("i", solved);
}



int Composite_graphcut_select(Composite * this, MaxFlowAPI * mf, float edge_bias, float smooth_bias, float weight_bias)
{
 // MaxFlow object, for fun and games...
  MaxFlow maxflow;
  mf->init(&maxflow, 32, 32); // Will almost certainly grow from these values, but good starting point.
  
 // Create a fun little data structure to allow us to quickly grow regions and find all pixels that we are considering... 
  int job_code = 0;
  int pixels = this->width * this->height;
  PixelState * pstate = (PixelState*)malloc(pixels * sizeof(PixelState));
  EdgeState  * estate =  (EdgeState*)malloc(pixels * 2 * sizeof(EdgeState));
  
  int i;
  for (i=0; i<pixels; i++) pstate[i].job = -1;
 
 // Loop until all cases have been resolved...
  int check_pixel = 0;
  while (1)
  {
   // First find a pixel with overlap...
    while ((check_pixel<pixels) && ((this->data[check_pixel]==NULL) || (this->data[check_pixel]->next==NULL))) check_pixel += 1;
    if (check_pixel==pixels) break;
    
   // Select two segments...
    int low_part = -1;
    int high_part = -1;
     
    Pixel * targ = this->data[check_pixel];
    while (targ)
    {
     high_part = low_part;
     low_part = targ->part;
     targ = targ->next; 
    }
    
   // Grow from the pixel, finding all pixels where overlap between the two selected segments exists...
    pstate[check_pixel].next = NULL;
    pstate[check_pixel].job = job_code;
    PixelState * to_check = pstate + check_pixel;
    
    PixelState * region = NULL;
    int vertex_count = 0;
        
    while (to_check!=NULL)
    {
     // Remove the pixel we are going to play with from the work queue...
      PixelState * tps = to_check;
      to_check = to_check->next;
      
     // Check if its a member - if so add it to the region...
      Pixel * low = NULL;
      Pixel * high = NULL;
      
      targ = this->data[tps - pstate];
      while (targ!=NULL)
      {
       if (targ->part==low_part)  low  = targ;
       if (targ->part==high_part) high = targ;
       targ = targ->next;
      }
      
      if ((low!=NULL)&&(high!=NULL))
      {
       tps->next = region;
       region = tps;
       tps->pos = vertex_count;
       ++vertex_count;
       
       float dr = low->c.r*low->c.a - high->c.r*high->c.a;
       float dg = low->c.g*low->c.a - high->c.g*high->c.a;
       float db = low->c.b*low->c.a - high->c.b*high->c.a;
       float da = low->c.a - high->c.a;
       tps->diff = sqrt(dr*dr + dg*dg + db*db + da*da);
       
       tps->low_l = (low->c.r + low->c.g + low->c.b) / 3.0;
       tps->high_l = (high->c.r + high->c.g + high->c.b) / 3.0;
       
       tps->weight = weight_bias * (high->w - low->w);
      }
      else
      {
       tps->pos = -1; 
      }
      
     // Check its neighbours, updating their job value and adding them to the to_check work queue...
      if ((low!=NULL)&&(high!=NULL))
      {
       int y = (tps - pstate) / this->width;
       int x = (tps - pstate) % this->width;
       
       // +ve x...
        if ((x+1)<this->width)
        {
         PixelState * other = tps + 1;
         if (other->job!=job_code)
         {
          other->job = job_code;
          other->next = to_check;
          to_check = other;
         }
        }
       
       // -ve x...
        if (x>0)
        {
         PixelState * other = tps - 1;
         if (other->job!=job_code)
         {
          other->job = job_code;
          other->next = to_check;
          to_check = other;
         }
        }
       
       // +ve y...
        if ((y+1)<this->height)
        {
         PixelState * other = tps + this->width;
         if (other->job!=job_code)
         {
          other->job = job_code;
          other->next = to_check;
          to_check = other;
         }
        }
       
       // -ve y...
        if (y>0)
        {
         PixelState * other = tps - this->width;
         if (other->job!=job_code)
         {
          other->job = job_code;
          other->next = to_check;
          to_check = other;
         }
        }
      }
    }
    
    
   // Collate all the edges - check for a neighbour for each member of the disputed region...
    EdgeState * edges = NULL;
    int edge_count = 0;
    
    PixelState * tps = region;
    while (tps!=NULL)
    {
     int y = (tps - pstate) / this->width;
     int x = (tps - pstate) % this->width;
     
     if (x>0)
     {
      PixelState * other = tps - 1;
      if (other->pos>=0)
      {
       estate[edge_count].next = edges;
       edges = estate + edge_count;
       
       estate[edge_count].from = other;
       estate[edge_count].to = tps;
       estate[edge_count].pos = edge_count;
       
       ++edge_count;
      }
     }
     
     if (y>0)
     {
      PixelState * other = tps - this->width;
      if (other->pos>=0)
      {
       estate[edge_count].next = edges;
       edges = estate + edge_count;
       
       estate[edge_count].from = other;
       estate[edge_count].to = tps;
       estate[edge_count].pos = edge_count;
       
       ++edge_count;
      }
     }
     
     tps = tps->next; 
    }
    
    
   // Determine the source/sink links - occur whenever we have an edge that connects to a undisputed region - for each pixel check the 4-neighbourhood for such fixed region ownership and factor in the cost adjustment...
    tps = region;
    while (tps!=NULL)
    {
     int y = (tps - pstate) / this->width;
     int x = (tps - pstate) % this->width;
     
     if (x>0)
     {
      PixelState * other = tps - 1;
      if (other->pos<0)
      {
       targ = this->data[other - pstate];
       while (targ!=NULL)
       {
        if (targ->part==low_part)  tps->weight -= tps->diff + edge_bias;
        if (targ->part==high_part) tps->weight += tps->diff + edge_bias;
        targ = targ->next; 
       }
      }
     }
     
     if ((x+1)<this->width)
     {
      PixelState * other = tps + 1;
      if (other->pos<0)
      {
       targ = this->data[other - pstate];
       while (targ!=NULL)
       {
        if (targ->part==low_part)  tps->weight -= tps->diff + edge_bias;
        if (targ->part==high_part) tps->weight += tps->diff + edge_bias;
        targ = targ->next; 
       }
      }
     }

     if (y>0)
     {
      PixelState * other = tps - this->width;
      if (other->pos<0)
      {
       targ = this->data[other - pstate];
       while (targ!=NULL)
       {
        if (targ->part==low_part)  tps->weight -= tps->diff + edge_bias;
        if (targ->part==high_part) tps->weight += tps->diff + edge_bias;
        targ = targ->next; 
       }
      }
     }
     
     if ((y+1)<this->height)
     {
      PixelState * other = tps + this->width;
      if (other->pos<0)
      {
       targ = this->data[other - pstate];
       while (targ!=NULL)
       {
        if (targ->part==low_part)  tps->weight -= tps->diff + edge_bias;
        if (targ->part==high_part) tps->weight += tps->diff + edge_bias;
        targ = targ->next; 
       }
      }
     }
     
     tps = tps->next;
    }
    

   // Build a maxflow problem...
    mf->resize(&maxflow, vertex_count + 2, vertex_count + edge_count);
    
    mf->set_source(&maxflow, vertex_count);
    mf->set_sink(&maxflow, vertex_count+1);
    
    // Add the edges between pixels...
     EdgeState * te = edges;
     while (te!=NULL)
     {
      mf->set_edge(&maxflow, te->pos, te->from->pos, te->to->pos);
     
      float break_cost = 0.5 * (te->from->diff + te->to->diff);
     
      break_cost += smooth_bias * fabs(te->from->low_l - te->to->low_l);
      break_cost += smooth_bias * fabs(te->from->high_l - te->to->high_l);
     
      mf->cap_flow(&maxflow, te->pos, break_cost, break_cost);
     
      te = te->next;
     }
    
    // Add source and sink edges - all disputed pixels are connected to just one of them...
     tps = region;
     while (tps!=NULL)
     {
      if (tps->weight<0.0)
      {
       // Weight difference is in favour of low - create an edge to break with high...
        mf->set_edge(&maxflow, edge_count, tps->pos, vertex_count);
        mf->cap_flow(&maxflow, edge_count, -tps->weight, -tps->weight);
      }
      else
      {
       // Above in opposite land...
        mf->set_edge(&maxflow, edge_count, tps->pos, vertex_count+1);
        mf->cap_flow(&maxflow, edge_count, tps->weight, tps->weight);
      }
      
      ++edge_count;
      tps = tps->next; 
     }
    
     
   // Solve it...
     mf->solve(&maxflow);
     
     
   // Update the composite with the decision for each pixel...
    tps = region;
    while (tps!=NULL)
    {
     int to_die = (mf->get_side(&maxflow, tps->pos)==1) ? low_part : high_part;
     int loc = tps - pstate;
     
     Pixel * dead;
     if (this->data[loc]->part==to_die)
     {
      dead = this->data[loc];
      this->data[loc] = this->data[loc]->next;
     }
     else
     {
      targ = this->data[loc];
      while (targ->next->part!=to_die) targ = targ->next;
      dead = targ->next;
      targ->next = targ->next->next;
     }
     
     dead = this->new_pixel;
     this->new_pixel = dead;
     
     targ = this->data[loc];
     while (targ!=NULL)
     {
      if ((targ->part==high_part)||(targ->part==low_part))
      {
       targ->part = this->next_part;
      }
      targ = targ->next;
     }
     
     tps = tps->next; 
    }
    
    this->next_part += 1;
   
   // Move to the next job...
    ++job_code;
  }
  
 // Clean up...
  free(estate);
  free(pstate);
  mf->deinit(&maxflow);
  
 return job_code;
}



static PyObject * Composite_graphcut_select_py(Composite * self, PyObject * args)
{
 // Extract the optional parameters...
  float edge_bias = 0.0;
  float smooth_bias = 0.0;
  float weight_bias = 0.0;
  
  if (!PyArg_ParseTuple(args, "|fff", &edge_bias, &smooth_bias, &weight_bias)) return NULL;
  
  if (edge_bias<0.0) edge_bias = 0.0;
  if (smooth_bias<0.0) smooth_bias = 0.0;
  if (weight_bias<0.0) weight_bias = 0.0;
 
 // Make sure the maxflow array is loaded...
  if (import_maxflow()!=0) return NULL;
 
 // Simply call through to the method...
  int solved = Composite_graphcut_select(self, maxflow, edge_bias, smooth_bias, weight_bias);
 
 // Return how many mincut problems have been solved...
  return Py_BuildValue("i", solved);
}



// Given an image, as an unsigned char buffer of size 4*width*height this dumps the pixel values into it, as a,b,g,r, height major. Pixels with no values get assigned the background, of all the rest it takes the most recent colour written to the buffer...
void Composite_render_last(Composite * this, unsigned char * out_image)
{
 unsigned char * out = out_image;
 Pixel ** in = this->data;
 
 int y, x;
 for (y=0; y<this->height; y++)
 {
  for (x=0; x<this->width; x++)
  {
   // Calculate the colour...
    float r, g, b, a;
    
    if (*in==NULL)
    {
     r = this->bg.r;
     g = this->bg.g;
     b = this->bg.b;
     a = this->bg.a;
    }
    else
    {
     r = (*in)->c.r;
     g = (*in)->c.g;
     b = (*in)->c.b;
     a = (*in)->c.a;
    }
   
   // Clamp them...
    if (r<0.0) r = 0.0;
    if (r>1.0) r = 1.0;
    if (g<0.0) g = 0.0;
    if (g>1.0) g = 1.0;
    if (b<0.0) b = 0.0;
    if (b>1.0) b = 1.0;
    if (a<0.0) a = 0.0;
    if (a>1.0) a = 1.0;
    
   // Record it...
    out[0] = (unsigned char)floor(a*b*255.0+0.5);
    out[1] = (unsigned char)floor(a*g*255.0+0.5);
    out[2] = (unsigned char)floor(a*r*255.0+0.5);
    out[3] = (unsigned char)floor(a*255.0+0.5);
    
   // Move to the next position...
    out += 4;
    in += 1;
  }
 }
}


static PyObject * Composite_render_last_py(Composite * self, PyObject * args)
{
 // Create a numpy array to return...
  npy_intp dims[3];
  dims[0] = self->height;
  dims[1] = self->width;
  dims[2] = 4;
  
  PyObject * ret = PyArray_SimpleNew(3, dims, NPY_UINT8);
 
 // Fill it in...
  Composite_render_last(self, (unsigned char*)PyArray_DATA(ret));
 
 // Return it...
  return ret;
}



// Given an image, as an unsigned char buffer of size 4*width*height this dumps the pixel values into it, as a,b,g,r, height major. Pixels with no values get assigned the background, those with 1 value get that value, those with multiple value are averaged using the weights...
void Composite_render_average(Composite * this, unsigned char * out_image)
{
 unsigned char * out = out_image;
 Pixel ** in = this->data;
 
 int y, x;
 for (y=0; y<this->height; y++)
 {
  for (x=0; x<this->width; x++)
  {
   // Calculate the colour...
    float r, g, b, a;
    
    if (*in==NULL)
    {
     r = this->bg.r;
     g = this->bg.g;
     b = this->bg.b;
     a = this->bg.a;
    }
    else
    {
     r = 0.0;
     g = 0.0;
     b = 0.0;
     a = 0.0;
     
     float w = 0.0;
     
     Pixel * targ = *in;
     while (targ!=NULL)
     {
      w += targ->w;
      
      r += targ->w*(targ->c.r - r) / w;
      g += targ->w*(targ->c.g - g) / w;
      b += targ->w*(targ->c.b - b) / w;
      a += targ->w*(targ->c.a - a) / w;
       
      targ = targ->next;
     }
    }
   
   // Clamp them...
    if (r<0.0) r = 0.0;
    if (r>1.0) r = 1.0;
    if (g<0.0) g = 0.0;
    if (g>1.0) g = 1.0;
    if (b<0.0) b = 0.0;
    if (b>1.0) b = 1.0;
    if (a<0.0) a = 0.0;
    if (a>1.0) a = 1.0;
    
   // Record it...
    out[0] = (unsigned char)floor(a*b*255.0+0.5);
    out[1] = (unsigned char)floor(a*g*255.0+0.5);
    out[2] = (unsigned char)floor(a*r*255.0+0.5);
    out[3] = (unsigned char)floor(a*255.0+0.5);
    
   // Move to the next position...
    out += 4;
    in += 1;
  }
 }
}


static PyObject * Composite_render_average_py(Composite * self, PyObject * args)
{
 // Create a numpy array to return...
  npy_intp dims[3];
  dims[0] = self->height;
  dims[1] = self->width;
  dims[2] = 4;
  
  PyObject * ret = PyArray_SimpleNew(3, dims, NPY_UINT8);
 
 // Fill it in...
  Composite_render_average(self, (unsigned char*)PyArray_DATA(ret));
 
 // Return it...
  return ret;
}



static PyMemberDef Composite_members[] =
{
 {"width", T_INT, offsetof(Composite, width), READONLY, "Width of the image."},
 {"height", T_INT, offsetof(Composite, height), READONLY, "Height of the image."},
 {NULL}
};



static PyMethodDef Composite_methods[] =
{
 {"set_size", (PyCFunction)Composite_set_size_py, METH_VARARGS, "Sets the size of the compositing area, and in the process resets the data structure. As the default size after construction is (0,0) you often call this method immediatly afterwards."},
 {"set_bg", (PyCFunction)Composite_set_bg_py, METH_VARARGS, "Allows you to set the background colour, by passing in the parameters r,g, b and a. The last is optional, as it is usually ignored."},
 {"get_bg", (PyCFunction)Composite_get_bg_py, METH_VARARGS, "Returns the background colour, as the tuple (r, g, b, a). The last entry is the alpha value, which is typically ignored."},
 
 {"draw_line_graph", (PyCFunction)Composite_draw_line_graph_py, METH_VARARGS, "Draws a LineGraph object onto the Compositing area, before returning its assigned part number. The part number is then used to call a paint function, which converts the uv coordinates that have been drawn by this into an actual texture (Must be done before further parts are drawn). A second optional parameter is a term that increases the radius of the lines, to make sure everything is copied across without any abrupt edges. A third parameter controls the reweighting based on texture stretch - if a texture is at too low resolution its weight is reduced - 0 means no effect (default), 1 means the full effect."},
 
 {"paint_test_pattern", (PyCFunction)Composite_paint_test_pattern_py, METH_VARARGS, "A paint operation, called immediatly after a paint operation to apply actual colour to the painted pattern, using the UV coordinates. This uses a simple synthetic pattern, constructed using sin curves, for testing things."},
 {"paint_texture_nearest", (PyCFunction)Composite_paint_texture_nearest_py, METH_VARARGS, "Given parameters (numpy array, part number) this uses the numpy array to texture the identified part. Numpy array must be indexed [y,x, c] where c=0 is blue, c=1 is green and c=2 is red, and of type uint8. No interpolation - just nearest pixel."},
 {"paint_texture_linear", (PyCFunction)Composite_paint_texture_linear_py, METH_VARARGS, "Given parameters (numpy array, part number) this uses the numpy array to texture the identified part. Numpy array must be indexed [y,x, c] where c=0 is blue, c=1 is green and c=2 is red, and of type uint8. This uses linear interpolation."},
 {"cost_texture_nearest", (PyCFunction)Composite_cost_texture_nearest_py, METH_VARARGS, "Same as paint_texture_nearest (inc. for parameters) except it deletes the part and returns a cost for painting it, in terms of colour distance from pre-existing painted pixels."},
 
 {"inc_weight_alpha", (PyCFunction)Composite_inc_weight_alpha_py, METH_VARARGS, "Adds to the mixing weight of every pixel in the image a value provided to this method multiplied the alpha of the pixel. Used to bias the output towards pixels that are more opaque"},
 {"draw_pair", (PyCFunction)Composite_draw_pair_py, METH_VARARGS, "Given as input two part numbers this makes sure that every pixel that has one part in it also has the other part in it, by adding a pixel with alpha 0 with the relevent part number as needed. The weight of the extra pixel is by default 1, and can be provided as an optional third parameter."},
 
 {"maxflow_select", (PyCFunction)Composite_maxflow_select_py, METH_VARARGS, "Uses the maxflow (mincut) algorithm to select and process overlapping regions of the image, to select which layer to keep to minimise visual error. Where there are multiple overlaps it processes each in turn as a binary selection problem, until every pixel has at most one sample in. Has two optional parameters (a bias term to increase the cost of breaks on the edge of a contested region, a weight to assign to adjacent colour similarity terms, to avoid cuts when edges overlap.). Returns how many seperate maxflow problems have been solved."},
 {"graphcut_select", (PyCFunction)Composite_graphcut_select_py, METH_VARARGS, "Exactly the same as maxflow_select, except it solves a complete binary graph cut problem for each overlaping region. This means that in principal you could have complex boundaries, but allows it to factor in the weight term provided to the system. Supports an extra third parameter - a multiplier to the weight when factoring it into the cost to be minimised."},
 
 {"render_last", (PyCFunction)Composite_render_last_py, METH_VARARGS, "Renders the image out, returning a numpy array of type uint8, indexed [y, x, c], where red is c=2, green is c=1, blue is c=0 and alpha is c=3. This takes the most recent layer added, and is basically a stupid approach."},
 {"render_average", (PyCFunction)Composite_render_average_py, METH_VARARGS, "Renders the image out, returning a numpy array of type uint8, indexed [y, x, c], where red is c=2, green is c=1, blue is c=0 and alpha is c=3. This combines layers by simply averaging their overlaping colour, weighted by the weight value. Pretty simple basically."},

 {NULL}
};



static PyTypeObject CompositeType =
{
 PyObject_HEAD_INIT(NULL)
 0,                                /*ob_size*/
 "composite_c.Composite",          /*tp_name*/
 sizeof(Composite),                /*tp_basicsize*/
 0,                                /*tp_itemsize*/
 (destructor)Composite_dealloc_py, /*tp_dealloc*/
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
 "Composites pixels from multiple sources together, in terms of parts that are added on one at a time. Designed specifically for combining glyphs together, as represented via LineGraph objects. Provides multiple algorithms to ultimatly combine the parts and obtain a smooth blend.", /* tp_doc */
 0,                                /* tp_traverse */
 0,                                /* tp_clear */
 0,                                /* tp_richcompare */
 0,                                /* tp_weaklistoffset */
 0,                                /* tp_iter */
 0,                                /* tp_iternext */
 Composite_methods,                /* tp_methods */
 Composite_members,                /* tp_members */
 0,                                /* tp_getset */
 0,                                /* tp_base */
 0,                                /* tp_dict */
 0,                                /* tp_descr_get */
 0,                                /* tp_descr_set */
 0,                                /* tp_dictoffset */
 0,                                /* tp_init */
 0,                                /* tp_alloc */
 Composite_new_py,                 /* tp_new */
};



static PyMethodDef composite_c_methods[] =
{
 {NULL}
};



#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC initcomposite_c(void)
{
 PyObject * mod = Py_InitModule3("composite_c", composite_c_methods, "Provides the ability to composite together multiple entites - primarily designed for merging together letters represented by LineGraph objects.");
 
 import_array();
 
 if (PyType_Ready(&CompositeType) < 0) return;
 
 Py_INCREF(&CompositeType);
 PyModule_AddObject(mod, "Composite", (PyObject*)&CompositeType);
}
