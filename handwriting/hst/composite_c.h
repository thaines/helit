#ifndef COMPOSITE_C_H
#define COMPOSITE_C_H

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

#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>



// Pre-decleration...
typedef struct Colour Colour;
typedef struct Pixel Pixel;
typedef struct PixelBlock PixelBlock;
typedef struct Composite Composite;



// Colour struct...
struct Colour
{
 float r;
 float g;
 float b;
 float a; // Alpha, not the pre-multiplied representation.
};



// A pixel. at each location in the image a linked list of these describes the parts that map to it - by combining them we get the output image.
struct Pixel
{
 Pixel * next;
 
 Colour c;
 
 int part; // Part index, used to detect if we merge or prepend during rendering.
 float mass; // For averaging the below values whilst merging.
 
 float u;
 float v;
 float w; // Weight, for interpolating between parts.
};



// Internal structure - used to avoid doing a malloc for every Pixel, by blocking it...
struct PixelBlock
{
 PixelBlock * next;
 Pixel data[0];
};



// Actual structure that represents this data structure in python...
struct Composite
{
 PyObject_HEAD
 
 int height;
 int width;
 
 Pixel ** data; // Height major, singularly linked list at every location.
 
 PixelBlock * storage; // Storage for the actual Pixel objects, to save on a billion small allocations. Blocks are height*width sized.
 Pixel * new_pixel; // Linked list of unused pixels.
 
 Colour bg; // Default background colour.
 
 int next_part; // For assigning part numbers.
};



#endif
