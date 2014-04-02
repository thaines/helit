#include <Python.h>
#include <structmember.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>



#include "gbp_c.h"



// Returns the offset of a half edge, going from source to destination...
static inline float HalfEdge_offset_pmean(HalfEdge * he)
{
 if (he < he->reverse) return  he->pairwise;
                  else return -he->reverse->pairwise;
}

// Returns the precision of the half edge...
static inline float HalfEdge_offset_prec(HalfEdge * he)
{
 if (he > he->reverse) return he->pairwise;
                  else return he->reverse->pairwise;
}

// Allows you to multiply the existing offset with another one - this is equivalent to set in the first instance when its initialised to a zero precision.
static float HalfEdge_offset_mult(HalfEdge * he, float offset, float prec)
{
 if (he < he->reverse)
 {
  he->pairwise += offset * prec;
  he->reverse->pairwise += prec;
 }
 else
 {
  he->pairwise += prec;
  he->reverse->pairwise -= offset * prec; 
 }
}
