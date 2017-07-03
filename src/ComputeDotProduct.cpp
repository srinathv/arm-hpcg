
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeDotProduct.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
#include <mpi.h>
#include "mytimer.hpp"
#endif
#ifndef HPCG_NO_OPENMP
#include "omp.h"
#endif
#include <cassert>

#ifdef HPCG_USE_OPTIMIZED_BLAS
#include "armpl.h"
#endif

#include "ComputeDotProduct.hpp"
#include "ComputeDotProduct_ref.hpp"

/*!
  Routine to compute the dot product of two vectors.

  This routine calls the reference dot-product implementation by default, but
  can be replaced by a custom routine that is optimized and better suited for
  the target system.

  @param[in]  n the number of vector elements (on this processor)
  @param[in]  x, y the input vectors
  @param[out] result a pointer to scalar value, on exit will contain the result.
  @param[out] time_allreduce the time it took to perform the communication between processes
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeDotProduct_ref
*/
int ComputeDotProduct(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce, bool & isOptimized) {
  assert(x.localLength>=n); // Test vector lengths
  assert(y.localLength>=n);

  double local_result = 0.0;
  double * xv = x.values;
  double * yv = y.values;
  /*
   * Try to unroll to 8, 4 or 2, if not possible, don't do anything
   */
  if ( n % 8 == 0 ) {
	  double local_result1, local_result2, local_result3, local_result4;
	  double local_result5, local_result6, local_result7, local_result8;
	  local_result1 = local_result2 = local_result3 = local_result4 = 0.0;
	  local_result5 = local_result6 = local_result7 = local_result8 = 0.0;
    if (yv==xv) {
#ifndef HPCG_NO_OPENMP
    #pragma omp parallel for reduction (+:local_result1,local_result2,local_result3,local_result4,local_result5,local_result6,local_result7,local_result8)
#endif
      for (local_int_t i=0; i<n; i+=8) {
      	local_result1 += xv[i  ]*xv[i  ];
      	local_result2 += xv[i+1]*xv[i+1];
      	local_result3 += xv[i+2]*xv[i+2];
      	local_result4 += xv[i+3]*xv[i+3];
      	local_result5 += xv[i+4]*xv[i+4];
      	local_result6 += xv[i+5]*xv[i+5];
      	local_result7 += xv[i+6]*xv[i+6];
      	local_result8 += xv[i+7]*xv[i+7];
      }
    } else {
#ifndef HPCG_NO_OPENMP
    #pragma omp parallel for reduction (+:local_result1,local_result2,local_result3,local_result4,local_result5,local_result6,local_result7,local_result8)
#endif
      for (local_int_t i=0; i<n; i+=8) {
      	local_result1 += xv[i  ]*yv[i  ];
      	local_result2 += xv[i+1]*yv[i+1];
      	local_result3 += xv[i+2]*yv[i+2];
      	local_result4 += xv[i+3]*yv[i+3];
      	local_result5 += xv[i+4]*yv[i+4];
      	local_result6 += xv[i+5]*yv[i+5];
      	local_result7 += xv[i+6]*yv[i+6];
      	local_result8 += xv[i+7]*yv[i+7];
      }
    }
	local_result = local_result1 + local_result2 + local_result3 + local_result4 + local_result5 + local_result6 + local_result7 + local_result8;
  } else if ( n % 4  == 0 ) {
	  double local_result1, local_result2, local_result3, local_result4;
	  local_result1 = local_result2 = local_result3 = local_result4 = 0.0;
    if (yv==xv) {
#ifndef HPCG_NO_OPENMP
    #pragma omp parallel for reduction (+:local_result1,local_result2,local_result3,local_result4)
#endif
      for (local_int_t i=0; i<n; i+=4) {
      	local_result1 += xv[i  ]*xv[i  ];
      	local_result2 += xv[i+1]*xv[i+1];
      	local_result3 += xv[i+2]*xv[i+2];
      	local_result4 += xv[i+3]*xv[i+3];
      }
    } else {
#ifndef HPCG_NO_OPENMP
    #pragma omp parallel for reduction (+:local_result1,local_result2,local_result3,local_result4)
#endif
      for (local_int_t i=0; i<n; i+=4) {
      	local_result1 += xv[i  ]*yv[i  ];
      	local_result2 += xv[i+1]*yv[i+1];
      	local_result3 += xv[i+2]*yv[i+2];
      	local_result4 += xv[i+3]*yv[i+3];
      }
    }
	local_result = local_result1 + local_result2 + local_result3 + local_result4;
  } else if ( n % 2 == 0) {
	  double local_result1, local_result2;
	  local_result1 = local_result2 = 0.0;
    if (yv==xv) {
#ifndef HPCG_NO_OPENMP
    #pragma omp parallel for reduction (+:local_result1,local_result2)
#endif
      for (local_int_t i=0; i<n; i+=2) {
      	local_result1 += xv[i  ]*xv[i  ];
      	local_result2 += xv[i+1]*xv[i+1];
      }
    } else {
#ifndef HPCG_NO_OPENMP
    #pragma omp parallel for reduction (+:local_result1,local_result2)
#endif
      for (local_int_t i=0; i<n; i+=2) {
      	local_result1 += xv[i  ]*yv[i  ];
      	local_result2 += xv[i+1]*yv[i+1];
      }
    }
	local_result = local_result1 + local_result2;
  } else {
    if (yv==xv) {
#ifndef HPCG_NO_OPENMP
    #pragma omp parallel for reduction (+:local_result)
#endif
      for (local_int_t i=0; i<n; i++) {
      	local_result += xv[i]*xv[i];
      }
    } else {
#ifndef HPCG_NO_OPENMP
    #pragma omp parallel for reduction (+:local_result)
#endif
      for (local_int_t i=0; i<n; i++) {
      	local_result += xv[i]*yv[i];
      }
    }
  }

#ifndef HPCG_NO_MPI
  // Use MPI's reduce function to collect all partial sums
  double t0 = mytimer();
  double global_result = 0.0;
  MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM,
      MPI_COMM_WORLD);
  result = global_result;
  time_allreduce += mytimer() - t0;
#else
  time_allreduce += 0.0;
  result = local_result;
#endif

  return 0;
}
