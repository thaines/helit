# Copyright (c) 2012, Tom SF Haines
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



from start_cpp import start_cpp



# Some basic matrix operations that come in use...
matrix_code = start_cpp() + """
#ifndef MATRIX_CODE
#define MATRIX_CODE

template <typename T>
inline void MemSwap(T * lhs, T * rhs, int count = 1)
{
 while(count!=0)
 {
  T t = *lhs;
  *lhs = *rhs;
  *rhs = t;

  ++lhs;
  ++rhs;
  --count;
 }
}


// Calculates the determinant - you give it a pointer to the first elment of the array, and its size (It must be square), plus its stride, which would typically be identical to size, which is the default.
template <typename T>
inline T Determinant(T * pos, int size, int stride = -1)
{
 if (stride==-1) stride = size;

 if (size==1) return pos[0];
 else
 {
  if (size==2) return pos[0]*pos[stride+1] - pos[1]*pos[stride];
  else
  {
   T ret = 0.0;
    for (int i=0; i<size; i++)
    {
     if (i!=0) MemSwap(&pos[0], &pos[stride*i], size-1);

     T sub = Determinant(&pos[stride], size-1, stride) * pos[stride*i + size-1];
     if ((i+size)%2) ret += sub;
                else ret -= sub;
    }

    for (int i=1; i<size; i++)
    {
     MemSwap(&pos[(i-1)*stride], &pos[i*stride], size-1);
    }
   return ret;
  }
 }
}


// Inverts a square matrix, will fail on singular and very occasionally on
// non-singular matrices, returns true on success. Uses Gauss-Jordan elimination
// with partial pivoting.
// in is the input matrix, out the output matrix, just be aware that the input matrix is trashed.
// You have to provide its size (Its square, obviously.), and optionally a stride if different from size.
template <typename T>
inline bool Inverse(T * in, T * out, int size, int stride = -1)
{
 if (stride==-1) stride = size;

 for (int r=0; r<size; r++)
 {
  for (int c=0; c<size; c++)
  {
   out[r*stride + c] = (c==r)?1.0:0.0;
  }
 }

 for (int r=0; r<size; r++)
 {
  // Find largest pivot and swap in, fail if best we can get is 0...
   T max = in[r*stride + r];
   int index = r;
   for (int i=r+1; i<size; i++)
   {
    if (fabs(in[i*stride + r])>fabs(max))
    {
     max = in[i*stride + r];
     index = i;
    }
   }
   if (index!=r)
   {
    MemSwap(&in[index*stride], &in[r*stride], size);
    MemSwap(&out[index*stride], &out[r*stride], size);
   }
   if (fabs(max-0.0)<1e-6) return false;

  // Divide through the entire row...
   max = 1.0/max;
   in[r*stride + r] = 1.0;
   for (int i=r+1; i<size; i++) in[r*stride + i] *= max;
   for (int i=0; i<size; i++) out[r*stride + i] *= max;

  // Row subtract to generate 0's in the current column, so it matches an identity matrix...
   for (int i=0; i<size; i++)
   {
    if (i==r) continue;
    T factor = in[i*stride + r];
    in[i*stride + r] = 0.0;

    for (int j=r+1; j<size; j++) in[i*stride + j] -= factor * in[r*stride + j];
    for (int j=0; j<size; j++) out[i*stride + j] -= factor * out[r*stride + j];
   }
 }

 return true;
}

#endif
"""
