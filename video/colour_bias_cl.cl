// Copyright 2012 Tom SF Haines

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.



kernel void from_rgb(const int width, const float chromaScale, const float lumScale, const float noiseFloor, const float16 mat, global const float4 * img_in, global float4 * img_out)
{
 // Get the pixel we are playing with...
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if (x>=width) return;

  const int index = y*width + x;
  const float4 colour = img_in[index];

 // Do the transformation...
  // Rotate the luminance into channel 0...
   float4 rotated;
   rotated.s0 = colour.s0 * mat.s0 + colour.s1 * mat.s1 + colour.s2 * mat.s2;
   rotated.s1 = colour.s0 * mat.s3 + colour.s1 * mat.s4 + colour.s2 * mat.s5;
   rotated.s2 = colour.s0 * mat.s6 + colour.s1 * mat.s7 + colour.s2 * mat.s8;

  // Seperate chromacity...
   float cDiv = max(rotated.s0, noiseFloor);

   rotated.s0 *= lumScale;
   rotated.s12 *= chromaScale / cDiv;

 // Write to the output image...
  img_out[index] = rotated;
}



kernel void to_rgb(const int width, const float chromaScale, const float lumScale, const float noiseFloor, const float16 inv_mat, global const float4 * img_in, global float4 * img_out)
{
 // Get the pixel we are playing with...
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if (x>=width) return;

  const int index = y*width + x;
  const float4 colour = img_in[index];

 // Do the transformation...
  // Unseperate chromacity...
   float4 merged;
   merged.s0 = colour.s0 / lumScale;

   float cDiv = max(merged.s0, noiseFloor);
   merged.s12 = colour.s12 * (cDiv / chromaScale);

  // Rotate back to rgb...
   float4 rotated;
   rotated.s0 = merged.s0 * inv_mat.s0 + merged.s1 * inv_mat.s1 + merged.s2 * inv_mat.s2;
   rotated.s1 = merged.s0 * inv_mat.s3 + merged.s1 * inv_mat.s4 + merged.s2 * inv_mat.s5;
   rotated.s2 = merged.s0 * inv_mat.s6 + merged.s1 * inv_mat.s7 + merged.s2 * inv_mat.s8;

 // Write to the output image...
  img_out[index] = rotated;
}
