# Copyright (c) 2011, Tom SF Haines
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import unittest
import random
import math

from scipy.special import gammaln, psi, polygamma
from scipy import weave

from utils.start_cpp import start_cpp



# Provides various gamma-related functions...
gamma_code = start_cpp() + """
#ifndef GAMMA_CODE
#define GAMMA_CODE

#include <cmath>

// Returns the natural logarithm of the Gamma function...
// (Uses Lanczos's approximation.)
double lnGamma(double z)
{
 static const double coeff[9] = {0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7};

 if (z<0.5)
 {
  // Use reflection formula, as approximation doesn't work down here...
   return log(M_PI) - log(sin(M_PI*z)) - lnGamma(1.0-z);
 }
 else
 {
  double x = coeff[0];
  for (int i=1;i<9;i++) x += coeff[i]/(z+i-1);

  double t = z + 6.5;

  return log(sqrt(2.0*M_PI)) + (z-0.5)*log(t) - t + log(x);
 }
}



// Calculates the Digamma function, i.e. the derivative of the log of the Gamma function - uses a partial expansion of an infinite series to 4 terms that is good for high values, and an identity to express lower values in terms of higher values...
double digamma(double z)
{
 static const double highVal = 13.0; // A bit of fiddling shows that the last term with this is of the order 1e-10, so we can expect at least 9 digits of accuracy past the decimal point.

 double ret = 0.0;
 while (z<highVal)
 {
  ret -= 1.0/z;
  z += 1.0;
 }

 double iz1 = 1.0/z;
 double iz2 = iz1*iz1;
 double iz4 = iz2*iz2;
 double iz6 = iz4*iz2;
 
 ret += log(z) - iz1/2.0 - iz2/12.0 + iz4/120.0 - iz6/252.0;
 return ret;
}

// Calculates the trigamma function - uses a partial expansion of an infinite series that is accurate for large values, and then uses an identity to express lower values in terms of higher values - same approach as for the digamma function basically...
double trigamma(double z)
{
 static const double highVal = 8.0;

 double ret = 0.0;
 while (z<highVal)
 {
  ret += 1.0/(z*z);
  z += 1.0;
 }

 z -= 1.0;
 double iz1 = 1.0/z;
 double iz2 = iz1*iz1;
 double iz3 = iz1*iz2;
 double iz5 = iz3*iz2;
 double iz7 = iz5*iz2;
 double iz9 = iz7*iz2;

 ret += iz1 - 0.5*iz2 + iz3/6.0 - iz5/30.0 + iz7/42.0 - iz9/30.0;
 return ret;
}

#endif
"""



def lnGamma(z):
  """Pointless as scipy, a library this is dependent on, defines this, but useful for testing. Returns the logorithm of the gamma function"""
  code = start_cpp(gamma_code) + """
  return_val = lnGamma(z);
  """
  return weave.inline(code, ['z'], support_code=gamma_code)

def digamma(z):
  """Pointless as scipy, a library this is dependent on, defines this, but useful for testing. Returns an evaluation of the digamma function"""
  code = start_cpp(gamma_code) + """
  return_val = digamma(z);
  """
  return weave.inline(code, ['z'], support_code=gamma_code)

def trigamma(z):
  """Pointless as scipy, a library this is dependent on, defines this, but useful for testing. Returns an evaluation of the trigamma function"""
  code = start_cpp(gamma_code) + """
  return_val = trigamma(z);
  """
  return weave.inline(code, ['z'], support_code=gamma_code)


class TestFuncs(unittest.TestCase):
  """Test code for the assorted gamma-related functions."""
  def test_compile(self):
    code = start_cpp(gamma_code) + """
    """
    weave.inline(code, support_code=gamma_code)

  def test_error_lngamma(self):
    for _ in xrange(1000):
      z = random.uniform(0.01, 100.0)
      own = lnGamma(z)
      good = gammaln(z)
      assert(math.fabs(own-good)<1e-12)

  def test_error_digamma(self):
    for _ in xrange(1000):
      z = random.uniform(0.01, 100.0)
      own = digamma(z)
      good = psi(z)
      assert(math.fabs(own-good)<1e-9)

  def test_error_trigamma(self):
    for _ in xrange(1000):
      z = random.uniform(0.01, 100.0)
      own = trigamma(z)
      good = polygamma(1,z)
      assert(math.fabs(own-good)<1e-9)


# If this file is run do the unit tests...
if __name__ == '__main__':
  unittest.main()

