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



// Impliments the prefered counter-based pseudo random number generator from the paper 'Parallel Random Numbers: As easy as 1,2, 3'. This is the Philox model with a counter consisting of 4 values and a key consisting of 2 values, each 32 bits, with 10 rounds. Input is 4 32 bit unsigned integers as a counter and 2 32 bit unsigned integers as keys. The idea is that each key gives you a new sequence, which you step through using the counter. Note that it is designed with this indexing structure such that you can assign the details arbitarilly, and as long as you don't request the same counter/key combo twice you will get 'random' data. Typical use with a GPU is to have the counter increasing for each needed random input, whilst the key is set differently for each location the kernel is run, using get_global_id. You can just as easilly swap these roles around however. The original paper indicates that this algorithm passes a large battery of statistical tests for randomness, but this was implimented from scratch, without reference to the original authors code, so there is a risk of bugs which could cause significant bias in the results, so use at your own risk. Hasn't caused me any issues however, and I did run a (small) set of tests on it to verify it was returning a uniform distribution (p-values from binomial test done on each bucket of a histogram with 10 buckets, for multiple indexing strategies and sample counts.).
uint4 philox(const uint4 counter, const uint2 key)
{
 const uint2 mult = (uint2)(0xCD9E8D57, 0xD2511F53);

 // Iterate and do each round in turn, updating the counter before we finally return it (Indexing from 1 is conveniant for the Weyl sequence.)...
 uint4 ret = counter;
 for (int rnd=1;rnd<=10;rnd++)
 {
  // Calculate key for this step, by applying the Weyl sequence on it...
   uint2 keyWeyl = key;
   keyWeyl *= rnd;

  // Apply the s-blocks, also swap the r values between the s-blocks...
   uint4 next;
   next.s02 = ret.s13 * mult;
   next.s31 = mul_hi(ret.s13, mult) ^ keyWeyl ^ ret.s02;

  // Prepare for the next step...
   ret = next;
 }

 return ret;
}

// Wrapper for the above - bit inefficient as it only uses part of the 'random' data...
float uniform(const uint4 counter, const uint2 key)
{
 uint4 ret = philox(counter, key);
 return ((float)ret.s0) / ((float)0xffffffff);
}



// This is used to reset the mix data structure - hardly complicated...
kernel void reset(const int width, const int comp_count, global float8 * mix)
{
 // Fetch the part that is our responsibility...
  const int c = get_global_id(0);
  const int x = get_global_id(1);
  const int y = get_global_id(2);

  if (x>=width) return;

 // Calculate the correct offset...
  const int mixPos = (y*width + x)*comp_count + c;

 // Zero it out...
  mix[mixPos] = 0.0;
}



// Prior update - used when the prior is changed, to keep the model in-line with the change...
kernel void prior_update_mix(const int width, const int comp_count, const float4 deltaMu, const float4 deltaSigma2, global float8 * mix)
{
 // Get the mixture component that we are working on...
  const int c = get_global_id(0);
  const int x = get_global_id(1);
  const int y = get_global_id(2);

  if (x>=width) return;

  const int mixPos = (y*width + x)*comp_count + c;
  const float8 comp = mix[mixPos];

 // Do the update if safe...
  if (comp.s0>1e-2)
  {
   comp.s123 -= deltaMu.s012;
   comp.s456 -= deltaSigma2.s012 / comp.s0;

   // Write back...
    mix[mixPos] = comp;
  }
}


// Applys a multiplicative constant to the mixture components...
kernel void light_update_mix(const int width, const int comp_count, const float4 prior_mu, const float4 mult, global float8 * mix)
{
 // Get the  mixture component that we are working on...
  const int c = get_global_id(0);
  const int x = get_global_id(1);
  const int y = get_global_id(2);

  if (x>=width) return;

  const int mixPos = (y*width + x)*comp_count + c;
  const float4 comp = mix[mixPos].s0123;

  comp.s123 += prior_mu.s012;
  comp.s123 *= mult.s012;
  comp.s123 = fmin(comp.s123, 1.0);
  comp.s123 -= prior_mu.s012;

 // Write back...
  mix[mixPos].s123 = comp.s123;
}



// This is run for each mixture component in the system; it calculates the probability of the current pixel value being drawn from the component, storing it in the 8th value of the component vector, ready for the next step...
kernel void comp_prob(const int width, const int comp_count, const float prior_count, const float4 prior_mu, const float4 prior_sigma2, global const float4 * image, global float8 * mix)
{
 // Get the pixel and mixture component that we are working on...
  const int c = get_global_id(0);
  const int x = get_global_id(1);
  const int y = get_global_id(2);

  if (x>=width) return;

   const int base = y*width + x;
   const float4 pixel = image[base];
   const int mixPos = base*comp_count + c;
   const float8 comp = mix[mixPos];

  // Calculate the parameters for the t-distributions...
   float4 mean;
   float4 var;

   const float n = prior_count + comp.s0;
   const float nMult = (n + 1.0f) / (n * n);
   mean.s012 = prior_mu.s012 + comp.s123;
   var.s012 = nMult * (prior_sigma2.s012 + comp.s0 * comp.s456);

  // Calculate the shared parts of the student-t distribution - the normalising constant basically...
   const float term = 0.5f * (n + 1.0f);
   const float norm_cube = 0.06349363593424101f;

  // Evaluate the student-t distribution for each of the colour channels...
   const float4 delta = pixel - mean;

   float4 core;
   core.s012 = (delta.s012*delta.s012) / (n*var.s012);
   core.s012 += 1.0f;

   const float eval = norm_cube * rsqrt(var.s0*var.s1*var.s2);
   const float evalCore = core.s0 * core.s1 * core.s2;

   const float result = eval * pow(evalCore, -term);

  // Store the result...
   mix[mixPos].s7 = result;
}


// Same as comp_prob, but only uses the luminence channel...
kernel void comp_prob_lum(const int width, const int comp_count, const float prior_count, const float4 prior_mu, const float4 prior_sigma2, global const float4 * image, global float8 * mix)
{
 // Get the pixel and mixture component that we are working on...
  const int c = get_global_id(0);
  const int x = get_global_id(1);
  const int y = get_global_id(2);

  if (x>=width) return;

   const int base = y*width + x;
   const float4 pixel = image[base];
   const int mixPos = base*comp_count + c;
   const float8 comp = mix[mixPos];

  // Calculate the parameters for the t-distributions...
   const float n = prior_count + comp.s0;
   const float nMult = (n + 1.0f) / (n * n);
   
   float mean = prior_mu.s0 + comp.s1;
   float var = nMult * (prior_sigma2.s0 + comp.s0 * comp.s4);

  // Calculate the shared parts of the student-t distribution - the normalising constant basically...
   const float term = 0.5f * (n + 1.0f);
   const float norm = 0.39894228040143276;

  // Evaluate the student-t distribution for each of the colour channels...
   const float delta = pixel.s0 - mean;

   const float core = ((delta*delta) / (n*var)) + 1.0;
   const float eval = norm * rsqrt(var);
   const float result = eval * pow(core, -term);

  // Store the result...
   mix[mixPos].s7 = result;
}



// This is run for each pixel; it calculates the probability of the current pixel value being drawn from a new component, storing it in the 4th colour channel, ready for the next step...
kernel void new_comp_prob(const int width, const float prior_count, const float4 prior_mu, const float4 prior_sigma2, global float4 * image)
{
 // Get the pixel and mixture component that we are working on...
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if (x>=width) return;

  const int pixelInd = y*width + x;
  const float4 pixel = image[pixelInd];


 // Calculate the parameters for the t-distributions...
  float4 mean;
  float4 var;

  float n = prior_count;
  const float nMult = (n + 1.0f) / (n * n);
  mean.s012 = prior_mu.s012;
  var.s012 = nMult * prior_sigma2.s012;

 // Calculate the shared parts of the student-t distribution - the normalising constant basically...
  const float term = 0.5f * (n + 1.0f);
  const float norm_cube = 0.06349363593424101f;

 // Evaluate the student-t distribution for each of the colour channels...
  const float4 delta = pixel - mean;

  float4 core;
  core.s012 = (delta.s012*delta.s012) / (n*var.s012);
  core.s012 += 1.0f;

  const float eval = norm_cube * rsqrt(var.s0*var.s1*var.s2);
  const float evalCore = core.s0 * core.s1 * core.s2;

  const float result = eval * pow(evalCore, -term);

 // Store the result...
  image[pixelInd].s3 = result;
}


// Same as new_comp_prob, but only uses the first component...
kernel void new_comp_prob_lum(const int width, const float prior_count, const float4 prior_mu, const float4 prior_sigma2, global float4 * image)
{
 // Get the pixel and mixture component that we are working on...
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if (x>=width) return;

  const int pixelInd = y*width + x;
  const float4 pixel = image[pixelInd];


 // Calculate the parameters for the t-distributions...
  float n = prior_count;
  const float nMult = (n + 1.0f) / (n * n);
  
  float mean = prior_mu.s0;
  float var = nMult * prior_sigma2.s0;

 // Calculate the shared parts of the student-t distribution - the normalising constant basically...
  const float term = 0.5f * (n + 1.0f);
  const float norm = 0.39894228040143276;

 // Evaluate the student-t distribution for each of the colour channels...
  const float delta = pixel.s0 - mean;

  const float core = ((delta*delta) / (n*var)) + 1.0;
  const float eval = norm * rsqrt(var);
  const float result = eval * pow(core, -term);

 // Store the result...
  image[pixelInd].s3 = result;
}



// This processes a pixel, using the results of the *comp_prob kernels. This consists of doing two things - calculating the probability of the current pixel value belonging to the background, which is stored in the prob array, and updating the model with the new pixel value...
kernel void update_pixel(const int frame, const int width, const int height, const int comp_count, const float prior_count, const float4 prior_mu, const float4 prior_sigma2, const float concentration, const float cap, const float weight, const float minWeight, global const float4 * image, global float8 * mix, global float * pixel_prob, float var_mult)
{
 // Get the pixel that we are working on...
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if (x>=width) return;

  const int base = mad24(y, width, x);
  const float4 pixel = image[base];
  const int mixBase = base*comp_count;


 // Calculate the probability of the image pixel given the background model...
  float probSum = concentration * pixel.s3;
  float countSum = concentration;

  int lowIndex = 0;
  float lowValue = 1e9f;
  float highValue = concentration;

  for (int c=0;c<comp_count;c++)
  {
   const float8 comp = mix[mixBase+c];
   
   if (comp.s0>1e-2)
   {
    probSum += comp.s0 * comp.s7;
    countSum += comp.s0;
   }

   if (comp.s0<lowValue)
   {
    lowIndex = c;
    lowValue = comp.s0;
   }
   highValue = fmax(highValue, comp.s0);
  }

  float prob = probSum / countSum;

 // Apply bayes rule to get the probability of the pixel belonging to the background; store it...
  prob /= prob + (highValue / cap); // (highValue/cap) represents P(data|foreground), and is assuming a uniform distribution over the unit-sized colour space. The value is faded in, which acheives the confidence boost during startup.

  pixel_prob[base] = prob;


 // Now we need to update the model - either update an existing component or 'create' a new component. After that the confidence cap needs to be applied...
  // Clamp the probability, which is now used as a weight for the below update...
   prob *= weight;
   prob = fmax(prob, minWeight);

  // Fetch a psuedo-random number...
   float r = probSum * uniform((uint4)(x,y,frame,102349), (uint2)(6546524,378946));

  // Check for usage of an existing component...
   int home = 0; // Index of updated component.
   for (;home<comp_count;++home)
   {
    float8 comp = mix[mixBase+home];
    if (comp.s0>1e-2)
    {
     r -= comp.s0 * comp.s7;
     if (r<0.0f) break;
    }
   }

  // Either update an existing component or create a new one (Actually, we always update, to save on branching, we just might be updating a zeroed out entry!)...
   // Local copy - either new or an old one...
    float8 comp;
    if (home<comp_count) comp = mix[mixBase+home];
    else
    {
     comp = 0.0;
     home = lowIndex;
    }

   // Get the current meaning, so its not an offset from the prior...
    const float trueCount = prior_count + comp.s0;
    const float4 trueMu = prior_mu + comp.s1233;
    const float4 trueSigma2 = prior_sigma2 + comp.s0*comp.s4566;
   
   // Calculate the half way values, to adjacent pixels...
    const float4 pixelUp = 0.5 * (image[mad24(max(0,y-1), width, x)] + pixel);
    const float4 pixelDown = 0.5 * (image[mad24(min(height-1,y+1), width, x)] + pixel);
    const float4 pixelLeft = 0.5 * (image[mad24(y, width, max(0,x-1))] + pixel);
    const float4 pixelRight = 0.5 * (image[mad24(y, width, min(width-1,x+1))] + pixel);

   // Calculate the standard deviation, using the half way values to define 4 squares (We use pixel as the mean, even though it is not the mean of the 4 pixel quarters surrounding it.)...
    float4 var = 0.0;
    var += (pixel * pixel) / 6.0;
    
    var += (pixelUp * pixelUp) / 6.0;
    var += (pixelDown * pixelDown) / 6.0;
    var += (pixelLeft * pixelLeft) / 6.0;
    var += (pixelRight * pixelRight) / 6.0;
    
    var += (pixelUp * pixelLeft) / 8.0;
    var += (pixelUp * pixelRight) / 8.0;
    var += (pixelDown * pixelLeft) / 8.0;
    var += (pixelDown * pixelRight) / 8.0;
    
    var -= (pixel * pixelUp) / 12.0;
    var -= (pixel * pixelDown) / 12.0;
    var -= (pixel * pixelLeft) / 12.0;
    var -= (pixel * pixelRight) / 12.0;
    
    const float4 mean = (pixelUp + pixelDown + pixelLeft + pixelRight) / 4.0;
    var -= (mean * mean);
    var *= var_mult;
   
   // Do the update...
    const float4 diff = pixel - trueMu;

    comp.s123 = (trueCount*trueMu.s012 + prob*pixel.s012) / (trueCount+prob);
    comp.s456 = trueSigma2.s012 + (prob*var.s012) + (trueCount*prob)*diff.s012*diff.s012 / (trueCount+prob);

   // Correct back so its an offset from the prior again...
    comp.s123 -= prior_mu.s012;
    comp.s456 = (comp.s456 - prior_sigma2.s012) / (comp.s0 + prob);

   // Also update the weight...
    comp.s0 += prob;


  // Write back...
   mix[mixBase+home] = comp;

  // Apply the confidence cap as necessary...
   float mult = fmin(cap / comp.s0, 1.0);
   for (int c=0;c<comp_count;c++) mix[mixBase+c].s0 *= mult;
}



// Used to calculate the background plate, as the modes of the background model...
kernel void extract_mode(const int width, const int comp_count, const float4 prior_mu, global float4 * image, global const float8 * mix)
{
 // Get the details of the pixel we are calculating for...
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if (x>=width) return;

  const int mixBase = (y*width + x)*comp_count;

 // Work out its mode...
  float4 pixel;
  float best = -1.0f;

  for (int c=0;c<comp_count;c++)
  {
   const float8 comp = mix[mixBase+c];

   if (comp.s0>best)
   {
    best = comp.s0;
    pixel.s012 = comp.s123;
   }
  }

 // Factor in the effect of the prior...
  pixel.s012 += prior_mu.s012;

 // Write out...
  image[y*width + x] = pixel;
}



// Extracts an estimate of the mode count for each pixel, so it can be visualised - outputs as a greyscale map (As colour), with black for 0 and white for the component_cap...
kernel void extract_component_count(const int width, const int comp_count, const float com_count_mass, const float cap, global float4 * image, global const float8 * mix)
{
 // Get the details of the pixel we are calculating for...
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if (x>=width) return;

  const int mixBase = (y*width + x) * comp_count;

 // Find the maximum...
  float threshold = 0.0;
  int c;
 
  for (c=0;c<comp_count;c++)
  {
   const float8 comp = mix[mixBase+c];
   threshold = max(threshold, comp.s0);
  }
  
 // Factor in the percentage...
  threshold *= com_count_mass;
  
  float count = 0.0;
  for (c=0;c<comp_count;c++)
  {
   const float8 comp = mix[mixBase+c];
   if (comp.s0>threshold) count += 1.0;
  }

 // Write out...
  float val = count / comp_count;
  image[y*width + x].s0 = val;
  image[y*width + x].s1 = val;
  image[y*width + x].s2 = val;
}



// This set of functions calculates the BP model (bgCost and changeCost) given the input image and probability map - bgCost is a straight calculation, but changeCost is first filled with colourmetric distances, using 4 methods to handle boundary conditions, plus a 5th to fill in said boundary, followed by a per-pixel function that converts them to costs, handling the interactions...
// (priorOffset is the offset applied to the background cost by the prior.)
kernel void setup_model_bgCost(const int width, const float prior_offset, const float cert_limit, global const float * pixel_prob, global float * bgCost)
{
 // Get the indexing for the pixel being processed...
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if (x>=width) return;

  const int index = y*width + x;

 // Do the bgCost - this is a simple transformation of the pixel_prob input...
  float p = fmax(cert_limit, pixel_prob[index]);
  float omp = fmax(cert_limit, 1.0f-pixel_prob[index]);

  bgCost[index] = prior_offset + log(p/omp);
}


kernel void setup_model_dist_boundary(const int width, const int d, global const float4 * image, global float * changeCost)
{
 // Get the indexing for the pixel being processed...
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if (x>=width) return;

  const int index = y*width + x;

 // Set the relevent entry to a suitably distant value...
  changeCost[index*4 + d] = 1e10f;
}


kernel void setup_model_dist_pos_x(const int width, global const float4 * image, global float * changeCost)
{
 // Get the indexing for the pixel being processed...
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if (x>=width) return;

  const int index = y*width + x;
  const int other = index + 1;

 // Get the data, zero out the last componet so we can use the built in distance function...
  float4 a = image[index];
  float4 b = image[other];
  a.s3 = 0.0f;
  b.s3 = 0.0f;

 // Calculate the distance...
  float d = distance(a, b);

 // Store it...
  changeCost[index*4 + 0] = d;
}


kernel void setup_model_dist_pos_y(const int width, global const float4 * image, global float * changeCost)
{
 // Get the indexing for the pixel being processed...
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if (x>=width) return;

  const int index = y*width + x;
  const int other = index + width;

 // Get the data, zero out the last componet so we can use the built in distance function...
  float4 a = image[index];
  float4 b = image[other];
  a.s3 = 0.0f;
  b.s3 = 0.0f;

 // Calculate the distance...
  float d = distance(a, b);

 // Store it...
  changeCost[index*4 + 1] = d;
}


kernel void setup_model_dist_neg_x(const int width, global const float4 * image, global float * changeCost)
{
 // Get the indexing for the pixel being processed...
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if (x>=width) return;

  const int index = y*width + x;
  const int other = index - 1;

 // Get the data, zero out the last componet so we can use the built in distance function...
  float4 a = image[index];
  float4 b = image[other];
  a.s3 = 0.0f;
  b.s3 = 0.0f;

 // Calculate the distance...
  float d = distance(a, b);

 // Store it...
  changeCost[index*4 + 2] = d;
}


kernel void setup_model_dist_neg_y(const int width, global const float4 * image, global float * changeCost)
{
 // Get the indexing for the pixel being processed...
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if (x>=width) return;

  const int index = y*width + x;
  const int other = index - width;

 // Get the data, zero out the last componet so we can use the built in distance function...
  float4 a = image[index];
  float4 b = image[other];
  a.s3 = 0.0f;
  b.s3 = 0.0f;

 // Calculate the distance...
  float d = distance(a, b);

 // Store it...
  changeCost[index*4 + 3] = d;
}



// Note: The target_dist is the distance required to satisfy the min_same_prob requirement.
kernel void setup_model_changeCost(const int width, const float half_life, const float target_dist, const float change_limit, const float change_mult, global float4 * changeCost)
{
 // Get the indexing for the pixel being processed...
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if (x>=width) return;

  const int index = y*width + x;

 // Fetch the distances, find the minimum...
  float4 dist = changeCost[index];
  dist = fmin(dist, half_life); // Probably not needed.
  float minDist = fmin(fmin(dist.s0, dist.s1), fmin(dist.s2, dist.s3));

 // Calculate and apply the adjustment required to make sure that at least one of the distances are at least target_dist...
  if (minDist>target_dist) dist *= target_dist / minDist;

 // Clamp all the distances at the half life, i.e. the distance of indifference...
  dist = fmin(dist, half_life);

 // Finally, convert the distances to costs...
  dist = 1.0f - (half_life / (half_life + dist));
  dist = fmax(dist, change_limit);
  dist = change_mult * log((1.0f-dist) / dist);

 // Store the costs ready for use...
  changeCost[index] = dist;
}



// Function for zeroing out the message passing structure, for initialisation...
kernel void reset_in(const int width, global float4 * in)
{
 // Get the indexes of the two pixels that are involved...
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if (x>=width) return;

  const int index = y*width + x;

 // Zero it out...
  in[index] = 0.0;
}



// These 4 functions send messages in the four directions as required for BP - kept seperate such that boundary conditions can be handled at the higher level, and it seperates things up even more for greater parallism, despite some repeated calculation. Naming scheme is send_* where * is the direction it sends the message..,
kernel void send_pos_x(const int width, global const float * bgCost, global const float4 * changeCost, global float4 * in, int iter)
{
 // Get the indexes of the two pixels that are involved...
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  
  //if (((x+y+iter)%2)==1) return;
  if ((x+1)>=width) return;

  const int from = y*width + x;
  const int to = from + 1;

 // Send the message...
  // The cost of assigning background to the 'from' pixel, ignoring the 'to' pixels evidence...
   const float bgOffset = bgCost[from] + in[from].s1 + in[from].s2 + in[from].s3;

  // The cost of these two pixels having different labels...
   const float cCost = changeCost[from].s0;

  // The final message, as an offset...
   in[to].s2 = fmin(cCost, bgOffset) - fmin(0.0f, cCost + bgOffset);
}

kernel void send_pos_y(const int width, global const float * bgCost, global const float4 * changeCost, global float4 * in, int iter)
{
 // Get the indexes of the two pixels that are involved...
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  //if (((x+y+iter)%2)==1) return;
  if (x>=width) return;

  const int from = y*width + x;
  const int to = from + width;

 // Send the message...
  // The cost of assigning background to the 'from' pixel, ignoring the 'to' pixels evidence...
   const float bgOffset = bgCost[from] + in[from].s0 + in[from].s2 + in[from].s3;

  // The cost of these two pixels having different labels...
   const float cCost = changeCost[from].s1;

  // The final message, as an offset...
   in[to].s3 = fmin(cCost, bgOffset) - fmin(0.0f, cCost + bgOffset);
}

kernel void send_neg_x(const int width, global const float * bgCost, global const float4 * changeCost, global float4 * in, int iter)
{
 // Get the indexes of the two pixels that are involved...
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  //if (((x+y+iter)%2)==1) return;
  if (x>=width) return;

  const int from = y*width + x;
  const int to = from - 1;

 // Send the message...
  // The cost of assigning background to the 'from' pixel, ignoring the 'to' pixels evidence...
   const float bgOffset = bgCost[from] + in[from].s0 + in[from].s1 + in[from].s3;

  // The cost of these two pixels having different labels...
   const float cCost = changeCost[from].s2;

  // The final message, as an offset...
   in[to].s0 = fmin(cCost, bgOffset) - fmin(0.0f, cCost + bgOffset);
}

kernel void send_neg_y(const int width, global const float * bgCost, global const float4 * changeCost, global float4 * in, int iter)
{
 // Get the indexes of the two pixels that are involved...
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  //if (((x+y+iter)%2)==1) return;
  if (x>=width) return;

  const int from = y*width + x;
  const int to = from - width;

 // Send the message...
  // The cost of assigning background to the 'from' pixel, ignoring the 'to' pixels evidence...
   const float bgOffset = bgCost[from] + in[from].s0 + in[from].s1 + in[from].s2;

  // The cost of these two pixels having different labels...
   const float cCost = changeCost[from].s3;

  // The final message, as an offset...
   in[to].s1 = fmin(cCost, bgOffset) - fmin(0.0f, cCost + bgOffset);
}



// Function for calculating the final state after all the BP iterations have been run...
kernel void calc_mask(const int width, global const float * bgCost, global const float4 * in, global char * mask)
{
 // Get the index of the pixel to calculate for...
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if (x>=width) return;

  const int index = y*width + x;

 // Calculate the sum of the messages and background cost, which are all offsets...
  float4 msg = in[index];
  float val = bgCost[index] + msg.s0 + msg.s1 + msg.s2 + msg.s3;

 // Due to use of offsets all that matters is which side of zero the final cost is - store it...
  char m;
  if (val<0.0) m = 1;
          else m = 0;
  mask[index] = m;
}



// This is used to downsample the BP model, so it can be solved at a lower resolution first to accelerate convergance...
// Note: that the coordinates are in the lower resolution model.
// Scalling is down by lossing pixels, i.e. divide the resolution by two, rounding down, so that edge pixels have to start from scratch - not ideal, but it means boundary conditions are easy.
// high = high resolution source, low = low resolution destination.
kernel void downsample_model(const int highWidth, global const float * highBgCost, global const float4 * highChangeCost, const int lowWidth, global float * lowBgCost, global float4 * lowChangeCost)
{
 // Get the index of the pixel to calculate for...
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if (x>=lowWidth) return;

  const int lowIndex = y*lowWidth + x;

 // And now the indices of the source pixels (y then x)...
  const int source00 = y*2*highWidth + x*2;
  const int source01 = source00 + 1;
  const int source10 = source00 + highWidth;
  const int source11 = source10 + 1;

 // The outputs...
  float bgCost;
  float4 changeCost;

 // Sum in the revevant cost terms...
  bgCost = highBgCost[source00] + highBgCost[source01] + highBgCost[source10] + highBgCost[source11];

  changeCost.s0 = highChangeCost[source01].s0 + highChangeCost[source11].s0;
  changeCost.s1 = highChangeCost[source10].s1 + highChangeCost[source11].s1;
  changeCost.s2 = highChangeCost[source00].s2 + highChangeCost[source10].s2;
  changeCost.s3 = highChangeCost[source00].s3 + highChangeCost[source01].s3;

 // Write back...
  lowBgCost[lowIndex] = bgCost;
  lowChangeCost[lowIndex] = changeCost;
}



// Moves the mesages from a half-resolution model to a full resolution model, as part of a hierachical solver...
// Note: The coordinates are for the high resolution model.
kernel void upsample_messages(const int lowWidth, global const float4 * lowIn, const int highWidth, global float4 * highIn)
{
 // Get the indices of the two pixels involved...
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if (x>=highWidth) return;

  const int lowIndex = (y/2)*lowWidth + (x/2); // From
  const int highIndex = y*highWidth + x; // To

 // Transfer across...
  highIn[highIndex] = 0.25f * lowIn[lowIndex];
}
