#ifndef BESSEL_H
#define BESSEL_H

// Copyright 2013 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



// Calculates the modified bessel function of the first kind. The order is given as twice the order, so it supports only whole and half orders. Iterative in nature, accuracy indicates how accurate it should be. Limit is a cap on how many iterations. Highest input this can take is 84 (And you shouldn't really get that close to the limit), to produce a value of about 1e35, any larger and it will return a finite but wrong answer.
float ModBesselFirst(int orderX2, float x, float accuracy, int limit);



#endif
