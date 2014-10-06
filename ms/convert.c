// Copyright 2014 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include "convert.h"

#include <stdlib.h>
#include <math.h>



// The no operation 'convertor'...
void NoOp_to(const float * before, float * after)
{
 after[0] = before[0]; 
}


const Convert NoOp =
{
 '.',
 "noop",
 "No operation - simply transfers across a single floating point value without modification.",
 1,
 1,
 NoOp_to,
 NoOp_to
};



// The delete 'convertor'...
void Delete_to_int(const float * external, float * internal)
{
 // No-op
}

void Delete_to_ext(const float * internal, float * external)
{
 external[0] = 0.0; // Better than leaving it random!
}


const Convert Delete =
{
 'x',
 "delete",
 "Marks a feature as not to be transferred into the internal system, hence deleting it. Probable use case is if feeding multiple MeanShift objects from a single data matrix, with different subsets of features.",
 1,
 0,
 Delete_to_int,
 Delete_to_ext
};



// The angle convertor...
void Angle_to_int(const float * external, float * internal)
{
 internal[0] = cos(external[0]);
 internal[1] = sin(external[0]);
}

void Angle_to_ext(const float * internal, float * external)
{
 external[0] = atan2(internal[1], internal[0]);
}


const Convert Angle =
{
 'A',
 "angle",
 "Converts an input angle, in radians, into a 2D unit vector that represents the angle - 0 goes to (1, 0). Typically you would then put a directional distribution on this unit vector.",
 1,
 2,
 Angle_to_int,
 Angle_to_ext
};



// The radial convertor...
void Radial_to_int(const float * external, float * internal)
{
 internal[0] = cos(external[0]) * external[1];
 internal[1] = sin(external[0]) * external[1];
}

void Radial_to_ext(const float * internal, float * external)
{
 external[0] = atan2(internal[1], internal[0]);
 external[1] = sqrt(internal[0] * internal[0] + internal[1]*internal[1]);
}


const Convert Radial =
{
 'r',
 "radial",
 "Takes as input 2 values - [angle in radians, radius] which it converts into a (x, y) coordinate, onto which you would put a standard spatial kernel, such as a Gaussian.",
 2,
 2,
 Radial_to_int,
 Radial_to_ext
};



// The spherical angle convertor...
void SphericalAngle_to_int(const float * external, float * internal)
{
 float sin_theta = sin(external[1]);
 
 internal[0] = sin_theta * cos(external[0]);
 internal[1] = sin_theta * sin(external[0]);
 internal[2] = cos(external[1]);
}

void SphericalAngle_to_ext(const float * internal, float * external)
{
 external[0] = atan2(internal[1], internal[0]);
 external[1] = acos(internal[2]);
}


const Convert SphericalAngle =
{
 'S',
 "spherical_angle",
 "Given two values - the angle in the x/y plane (0 is along the x axis) and then the angle off of that plane towards the z axis; all radians. It converts them into an unit 3D vector, which is the representation used by the directional distributions. Can be used with longitude then latitude if you assume that the z axis punctures the poles.",
 2,
 3,
 SphericalAngle_to_int,
 SphericalAngle_to_ext
};



// The spherical coordinates convertor...
void SphericalCoord_to_int(const float * external, float * internal)
{
 float sin_theta = sin(external[1]);
 
 internal[0] = external[2] * sin_theta * cos(external[0]);
 internal[1] = external[2] * sin_theta * sin(external[0]);
 internal[2] = external[2] * cos(external[1]);
}

void SphericalCoord_to_ext(const float * internal, float * external)
{
 external[2] = sqrt(internal[0] * internal[0] + internal[1] * internal[1] + internal[2] * internal[2]);
 
 external[0] = atan2(internal[1], internal[0]);
 external[1] = acos(internal[2] / external[2]);
}


const Convert SphericalCoord =
{
 's',
 "spherical_coord",
 "Given three values - the angle in the x/y plane (0 is along the x axis) and then the angle off of that plane towards the z axis (both radians) then finally a radius. It converts them into a 3D vector, for which a standard spatial distribution can be applied.",
 3,
 3,
 SphericalCoord_to_int,
 SphericalCoord_to_ext
};



// The angle axis convertor...
void AngleAxis_to_int(const float * external, float * internal)
{
 float angle = sqrt(external[0] * external[0] + external[1] * external[1] + external[2] * external[2]);
 float sin_half_angle = sin(0.5*angle);
 
 internal[0] = sin_half_angle * external[0] / angle;
 internal[1] = sin_half_angle * external[1] / angle;
 internal[2] = sin_half_angle * external[2] / angle;
 internal[3] = cos(0.5*angle);
}

void AngleAxis_to_ext(const float * internal, float * external)
{
 float angle = 2.0 * acos(internal[3]);
 
 if (angle>1e-6)
 {
  float mult = angle / sqrt(1.0 - internal[3]*internal[3]);
  external[0] = mult * internal[0];
  external[1] = mult * internal[1];
  external[2] = mult * internal[2];
 }
 else // No rotation - zero it...
 {
  external[0] = 0.0;
  external[1] = 0.0;
  external[2] = 0.0;
 }
}


const Convert AngleAxis =
{
 'V',
 "angle_axis",
 "Converts a rotation provided to the system using the right handed angle axis representation where the length of the vector is the angle in radians into a unit quaternion (versor), which is the right kind of object to put a (assuming path doesn't matter) mirrored directional distribution over.",
 3,
 4,
 AngleAxis_to_int,
 AngleAxis_to_ext
};



// The Euler angle convertor...
void Euler_to_int(const float * external, float * internal)
{
 float cx = cos(0.5 * external[0]);
 float cy = cos(0.5 * external[1]);
 float cz = cos(0.5 * external[2]);
 
 float sx = sin(0.5 * external[0]);
 float sy = sin(0.5 * external[1]);
 float sz = sin(0.5 * external[2]);
  
 internal[0] = sz * sy * cx + cz * cy * sx;
 internal[1] = sz * cy * cx + cz * sy * sx;
 internal[2] = cz * sy * cx - sz * cy * sx;
 internal[3] = cz * cy * cx - sz * sy * sx;
}

void Euler_to_ext(const float * internal, float * external)
{
 external[0] = atan2(2*internal[0]*internal[3] - 2*internal[1]*internal[2], 1 - 2*internal[0]*internal[0] - 2*internal[2]*internal[2]);
 external[1] = asin(2*internal[0]*internal[1] + 2*internal[2]*internal[3]);
 external[2] = atan2(2*internal[1]*internal[3] - 2*internal[0]*internal[2], 1 - 2*internal[1]*internal[1] - 2*internal[2]*internal[2]);
}


const Convert Euler =
{
 'E',
 "euler",
 "Converts a 3-vector of euler angles (radians) with the storage order being x-y-z, whilst the rotations are applied in the reverse of that, as in z-y-x. The conversion is into a unit quaternion (versor) which represents the rotation, for which a (typically mirrored) directional kernel makes sense.",
 3,
 4,
 Euler_to_int,
 Euler_to_ext
};



// The list of known convertors...
const Convert * ListConvert[] =
{
 &NoOp,
 &Delete,
 &Angle,
 &Radial,
 &SphericalAngle,
 &SphericalCoord,
 &AngleAxis,
 &Euler,
 NULL
};
