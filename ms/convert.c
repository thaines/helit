// Copyright 2014 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include "convert.h"

#include <stdlib.h>
#include <math.h>



// The angle convertor...
void Angle_to(const float * before, float * after)
{
 after[0] = cos(before[0]);
 after[1] = sin(before[0]);
}

void Angle_from(const float * before, float * after)
{
 after[0] = atan2(before[1], before[0]);
}


const Convert Angle =
{
 'A',
 "angle",
 "Converts an input angle, in radians, into a 2D unit vector that represents the angle - 0 goes to (1, 0). Typically you would then put a directional distribution on this unit vector.",
 1,
 2,
 Angle_to,
 Angle_from
};



// The radial convertor...
void Radial_to(const float * before, float * after)
{
 after[0] = cos(before[0]) * before[1];
 after[1] = sin(before[0]) * before[1];
}

void Radial_from(const float * before, float * after)
{
 after[0] = atan2(before[1], before[0]);
 after[1] = sqrt(before[0] * before[0] + before[1]*before[1]);
}


const Convert Radial =
{
 'r',
 "radial",
 "Takes as input 2 values - [angle in radians, radius] which it converts into a (x, y) coordinate, onto which you would put a standard spatial kernel, such as a Gaussian.",
 2,
 2,
 Radial_to,
 Radial_from
};



// The spherical angle convertor...
void SphericalAngle_to(const float * before, float * after)
{
 float sin_theta = sin(before[1]);
 
 after[0] = sin_theta * cos(before[0]);
 after[1] = sin_theta * sin(before[0]);
 after[2] = cos(before[1]);
}

void SphericalAngle_from(const float * before, float * after)
{
 after[0] = atan2(before[1], before[0]);
 after[1] = acos(before[2]);
}


const Convert SphericalAngle =
{
 'S',
 "spherical_angle",
 "Given two values - the angle in the x/y plane (0 is along the x axis) and then the angle off of that plane towards the z axis; all radians. It converts them into an unit 3D vector, which is the representation used by the directional distributions. Can be used with longitude then latitude if you assume that the z axis punctures the poles.",
 2,
 3,
 SphericalAngle_to,
 SphericalAngle_from
};



// The spherical coordinates convertor...
void SphericalCoord_to(const float * before, float * after)
{
 float sin_theta = sin(before[1]);
 
 after[0] = before[2] * sin_theta * cos(before[0]);
 after[1] = before[2] * sin_theta * sin(before[0]);
 after[2] = before[2] * cos(before[1]);
}

void SphericalCoord_from(const float * before, float * after)
{
 after[2] = sqrt(before[0] * before[0] + before[1] * before[1] + before[2] * before[2]);
 
 after[0] = atan2(before[1], before[0]);
 after[1] = acos(before[2] / after[2]);
}


const Convert SphericalCoord =
{
 's',
 "spherical_coord",
 "Given three values - the angle in the x/y plane (0 is along the x axis) and then the angle off of that plane towards the z axis (both radians) then finally a radius. It converts them into a 3D vector, for which a standard spatial distribution can be applied.",
 3,
 3,
 SphericalCoord_to,
 SphericalCoord_from
};



// The angle axis convertor...
void AngleAxis_to(const float * before, float * after)
{
 float angle = sqrt(before[0] * before[0] + before[1] * before[1] + before[2] * before[2]);
 float sin_half_angle = sin(0.5*angle);
 
 after[0] = sin_half_angle * before[0] / angle;
 after[1] = sin_half_angle * before[1] / angle;
 after[2] = sin_half_angle * before[2] / angle;
 after[3] = cos(0.5*angle);
}

void AngleAxis_from(const float * before, float * after)
{
 float angle = 2.0 * acos(before[3]);
 
 if (angle>1e-6)
 {
  float mult = angle / sqrt(1.0 - before[3]*before[3]);
  after[0] = mult * before[0];
  after[1] = mult * before[1];
  after[2] = mult * before[2];
 }
 else // No rotation - zero it...
 {
  after[0] = 0.0;
  after[1] = 0.0;
  after[2] = 0.0;
 }
}


const Convert AngleAxis =
{
 'V',
 "angle_axis",
 "Converts a rotation provided to the system using the right handed angle axis representation where the length of the vector is the angle in radians into a unit quaternion (versor), which is the right kind of object to put a (assuming path doesn't matter) mirrored directional distribution over.",
 3,
 4,
 AngleAxis_to,
 AngleAxis_from
};



// The Euler angle convertor...
void Euler_to(const float * before, float * after)
{
 float cx = cos(0.5 * before[0]);
 float cy = cos(0.5 * before[1]);
 float cz = cos(0.5 * before[2]);
 
 float sx = sin(0.5 * before[0]);
 float sy = sin(0.5 * before[1]);
 float sz = sin(0.5 * before[2]);
  
 after[0] = sz * sy * cx + cz * cy * sx;
 after[1] = sz * cy * cx + cz * sy * sx;
 after[2] = cz * sy * cx - sz * cy * sx;
 after[3] = cz * cy * cx - sz * sy * sx;
}

void Euler_from(const float * before, float * after)
{
 after[0] = atan2(2*before[0]*before[3] - 2*before[1]*before[2], 1 - 2*before[0]*before[0] - 2*before[2]*before[2]);
 after[1] = asin(2*before[0]*before[1] + 2*before[2]*before[3]);
 after[2] = atan2(2*before[1]*before[3] - 2*before[0]*before[2], 1 - 2*before[1]*before[1] - 2*before[2]*before[2]);
}


const Convert Euler =
{
 'E',
 "euler",
 "Converts a 3-vector of euler angles (radians) with the storage order being x-y-z, whilst the rotations are applied in the reverse of that, as in z-y-x. The conversion is into a unit quaternion (versor) which represents the rotation, for which a (typically mirrored) directional kernel makes sense.",
 3,
 4,
 Euler_to,
 Euler_from
};



// The list of known convertors...
const Convert * ListConvert[] =
{
 &Angle,
 &Radial,
 &SphericalAngle,
 &SphericalCoord,
 &AngleAxis,
 &Euler,
 NULL
};
