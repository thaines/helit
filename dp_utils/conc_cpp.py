# Copyright (c) 2011, Tom SF Haines
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



from utils.start_cpp import start_cpp



conc_code = start_cpp() + """

// This funky little function is used to resample the concentration parameter of a Dirichlet process, using the previous parameter - allows this parameter to be Gibbs sampled. Also works for any level of a HDP, due to the limited interactions.
// Parameters are:
// pcp - previous concentration parameter.
// n - number of samples taken from the Dirichlet process
// k - number of discretly different samples, i.e. table count in the Chinese restaurant process.
// prior_alpha - alpha value of the Gamma prior on the concentration parameter.
// prior_beta - beta value of the Gamma prior on the concentration parameter.
double sample_dirichlet_proc_conc(double pcp, double n, double k, double prior_alpha = 1.01, double prior_beta = 0.01)
{
 if ((n<(1.0-1e-6))||(k<(2.0-1e-6)))
 {
  return pcp; // Doesn't work in this case, so just repeat.
 }
 
 double nn = sample_beta(pcp+1.0, n);
 double log_nn = log(nn);
 
 double f_alpha = prior_alpha + k;
 double f_beta = prior_beta - log_nn;

 double pi_n_mod = (f_alpha - 1.0) / (n * f_beta);
 double r = sample_uniform();
 double r_mod = r / (1.0 - r);
 if (r_mod>=pi_n_mod) f_alpha -= 1.0;

 double ret = sample_gamma(f_alpha, f_beta);
 if (ret<1e-3) ret = 1e-3;
 return ret;
}


// Class to represent the concentration parameter associated with a DP - consists of the prior and the previous/current value...
struct Conc
{
 float alpha; // Parameter for Gamma prior.
 float beta;  // "
 float conc; // Previously sampled concentration value - needed for next sample, and for output/use.

 // Resamples the concentration value, assuming only a single DP is using it. n = number of samples from DP, k = number of unique samples, i.e. respectivly RefTotal() and Size() for a ListRef.
 void ResampleConc(int n, int k)
 {
  conc = sample_dirichlet_proc_conc(conc, n, k, alpha, beta);
  if (conc<1e-3) conc = 1e-3;
 }
};


// This class is the generalisation of the above for when multiple Dirichlet processes share a single concentration parameter - again allows a new concentration parameter to be drawn given the previous one and a Gamma prior, but takes multiple pairs of sample count/discrete sample counts, hence the class interface to allow it to accumilate the relevant information.
class SampleConcDP
{
 public:
   SampleConcDP():f_alpha(1.0),f_beta(1.0),prev_conc(1.0) {}
  ~SampleConcDP() {}

  // Sets the prior and resets the entire class....
   void SetPrior(double alpha, double beta)
   {
    f_alpha = alpha;
    f_beta = beta;
   }

  // Set the previous concetration parameter - must be called before any DP stats are added...
   void SetPrevConc(double prev)
   {
    prev_conc = prev;
   }

  // Call once for each DP that is using the concentration parameter...
  // (n is the number of samples drawn, k the number of discretly different samples.)
   void AddDP(double n, double k)
   {
    if (k>1.0)
    {
     double s = 0.0;
     if (sample_uniform()>(1.0/(1.0+n/prev_conc))) s = 1.0;

     double w = sample_beta(prev_conc+1.0,n);

     f_alpha += k - s;
     f_beta -= log(w);
    }
   }

  // Once all DP have been added call this to draw a new concentration value...
   double Sample()
   {
    double ret = sample_gamma(f_alpha, f_beta);
    if (ret<1e-3) ret = 1e-3;
    return ret;
   }

 private:
  double f_alpha;
  double f_beta;
  double prev_conc;
};

"""
