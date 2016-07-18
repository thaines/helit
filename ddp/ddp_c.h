#ifndef DDP_C_H
#define DDP_C_H

// Copyright 2016 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include <Python.h>
#include <structmember.h>



// Decleration of a type to represent the costs between two adjacent labels...
typedef void * PairCost;


// Allows you to create a PairCost object - it gets a single python object (any type) to define its behaviour. On error it should return NULL...
typedef PairCost (*PairCostNew)(PyObject * data);

// Opposite of the above...
typedef void (*PairCostDelete)(PairCost this);

// Given the indices of the labels and the associated PairCost object this has to return the cost (negative log liklihood, potentially not normalised) of that combination. Its allowed to return inf if it wants. Generally not used as the below can do clever optimisations - exists for completeness. first is the label for the random variable that appears first in the chain, second as you would expect...
typedef float (*PairCostCost)(PairCost this, int first, int second);

// Given an array of total costs thus far for a random variable this outputs the total cost on reaching the next label, factoring in the cost of taking the shortest transition. It also outputs the relevent backwards index. Whilst equivalent to calling the PairCostCost function this has the ability to do optimisations, which for many cost schemes can make it orders of magnitude faster...
typedef void (*PairCostCosts)(PairCost this, int first_count, float * first_total, int second_count, float * second_out, int * second_back);

// The exact oposite of the above, for doing a backwards pass if desired...
typedef void (*PairCostCostsRev)(PairCost this, int first_count, float * first_out, int * first_forward, int second_count, float * second_total);


// V-table for a given PairCost type...
typedef struct PairCostType PairCostType;

struct PairCostType
{
 const char *     name;
 const char *     description;
 PairCostNew      new;
 PairCostDelete   delete;
 PairCostCost     cost;
 PairCostCosts    costs;
 PairCostCostsRev costs_rev;
};



// Interface to PairCosts that assume that the first entry in the object is a pointer to its v-table...
void DeletePC(PairCost this);
float CostPC(PairCost this, int first, int second);
void CostsPC(PairCost this, int first_count, float * first_total, int second_count, float * second_out, int * second_back);
void CostsRevPC(PairCost this, int first_count, float * first_out, int * first_forward, int second_count, float * second_total);


// Declerations of various types of PairCostType...
extern const PairCostType DifferentType;
extern const PairCostType LinearType;
extern const PairCostType OrderedType;
extern const PairCostType FullType;

extern const PairCostType * ListPairCostType[];



// The actual structure...
typedef struct DDP DDP;

struct DDP
{
 PyObject_HEAD
 
 int variables; // Number of random variables in the chain.
 int * count; // Number of states associated with each random variable.
 
 int * offset; // Offset into below to get the costs associated with the given.
 float * cost; // Unary cost of selecting each random variable state.
 
 PairCost * pair_cost; // Indexed by the first of the pair random variable index, so one shorter than variables. NULL is interpreted as a reset - no information passed.
 
 int state; // 0 = not run, 1 = run, no backpass, 2 = run, with backpass.
 
 float * total; // Total cost thus far for the given random variable state.
 int * back; // Index in the previous random variable to get the state that gave the minimum cost. Kept for all variables, even if the destination is out of range.
 int * forward; // Opposite to the above - optionally calculated.
};



#endif
