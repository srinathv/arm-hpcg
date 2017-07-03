
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
 @file ComputeWAXPBY.cpp

 HPCG routine
 */

#include "ComputeWAXPBY.hpp"
#include "ComputeWAXPBY_ref.hpp"

/*!
  Routine to compute the update of a vector with the sum of two
  scaled vectors where: w = alpha*x + beta*y

  This routine calls the reference WAXPBY implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in] n the number of vector elements (on this processor)
  @param[in] alpha, beta the scalars applied to x and y respectively.
  @param[in] x, y the input vectors
  @param[out] w the output vector
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeWAXPBY_ref
*/
int ComputeWAXPBY(const local_int_t n, const double alpha, const Vector & x,
    const double beta, const Vector & y, Vector & w, bool & isOptimized) {

  assert(x.localLength>=n); // Test vector lengths
  assert(y.localLength>=n);

  const double * const xv = x.values;
  const double * const yv = y.values;
  double * const wv = w.values;
  /*
   * NOTE: Added a new if case where alpha and beta equal to 1. In that case
   * no multiplications are actually needed.
   *
   * Try to unroll to 8, 4 or 2, if not possible, don't do anything
   */
  if ( n % 8 == 0) {
  	if (alpha==1.0 and beta==1.0) {
#ifndef HPCG_NO_OPENMP
		#pragma omp parallel for
#endif
		for (local_int_t i = 0; i<n; i+=8) {
			wv[i  ] = xv[i  ] + yv[i  ];
			wv[i+1] = xv[i+1] + yv[i+1];
			wv[i+2] = xv[i+2] + yv[i+2];
			wv[i+3] = xv[i+3] + yv[i+3];
			wv[i+4] = xv[i+4] + yv[i+4];
			wv[i+5] = xv[i+5] + yv[i+5];
			wv[i+6] = xv[i+6] + yv[i+6];
			wv[i+7] = xv[i+7] + yv[i+7];
		}
	  } else if (alpha==1.0) {
#ifndef HPCG_NO_OPENMP
		#pragma omp parallel for
#endif
	    for (local_int_t i=0; i<n; i+=8) {
			wv[i  ] = xv[i  ] + beta * yv[i  ];
			wv[i+1] = xv[i+1] + beta * yv[i+1];
			wv[i+2] = xv[i+2] + beta * yv[i+2];
			wv[i+3] = xv[i+3] + beta * yv[i+3];
			wv[i+4] = xv[i+4] + beta * yv[i+4];
			wv[i+5] = xv[i+5] + beta * yv[i+5];
			wv[i+6] = xv[i+6] + beta * yv[i+6];
			wv[i+7] = xv[i+7] + beta * yv[i+7];
		}
 	 } else if (beta==1.0) {
#ifndef HPCG_NO_OPENMP
  	  #pragma omp parallel for
#endif
	    for (local_int_t i=0; i<n; i+=8) {
			wv[i  ] = alpha * xv[i  ] + yv[i  ];
			wv[i+1] = alpha * xv[i+1] + yv[i+1];
			wv[i+2] = alpha * xv[i+2] + yv[i+2];
			wv[i+3] = alpha * xv[i+3] + yv[i+3];
			wv[i+4] = alpha * xv[i+4] + yv[i+4];
			wv[i+5] = alpha * xv[i+5] + yv[i+5];
			wv[i+6] = alpha * xv[i+6] + yv[i+6];
			wv[i+7] = alpha * xv[i+7] + yv[i+7];
		}
 	 } else  {
#ifndef HPCG_NO_OPENMP
 	   #pragma omp parallel for
#endif
		 for (local_int_t i=0; i<n; i+=8) {
			wv[i  ] = alpha * xv[i  ] + beta * yv[i  ];
			wv[i+1] = alpha * xv[i+1] + beta * yv[i+1];
			wv[i+2] = alpha * xv[i+2] + beta * yv[i+2];
			wv[i+3] = alpha * xv[i+3] + beta * yv[i+3];
			wv[i+4] = alpha * xv[i+4] + beta * yv[i+4];
			wv[i+5] = alpha * xv[i+5] + beta * yv[i+5];
			wv[i+6] = alpha * xv[i+6] + beta * yv[i+6];
			wv[i+7] = alpha * xv[i+7] + beta * yv[i+7];
		 }
 	 }
  } else if ( n % 4 == 0) {
  	if (alpha==1.0 and beta==1.0) {
#ifndef HPCG_NO_OPENMP
		#pragma omp parallel for
#endif
		for (local_int_t i = 0; i<n; i+=4) {
			wv[i  ] = xv[i  ] + yv[i  ];
			wv[i+1] = xv[i+1] + yv[i+1];
			wv[i+2] = xv[i+2] + yv[i+2];
			wv[i+3] = xv[i+3] + yv[i+3];
		}
	  } else if (alpha==1.0) {
#ifndef HPCG_NO_OPENMP
		#pragma omp parallel for
#endif
	    for (local_int_t i=0; i<n; i+=4) {
			wv[i  ] = xv[i  ] + beta * yv[i  ];
			wv[i+1] = xv[i+1] + beta * yv[i+1];
			wv[i+2] = xv[i+2] + beta * yv[i+2];
			wv[i+3] = xv[i+3] + beta * yv[i+3];
		}
 	 } else if (beta==1.0) {
#ifndef HPCG_NO_OPENMP
  	  #pragma omp parallel for
#endif
	    for (local_int_t i=0; i<n; i+=4) {
			wv[i  ] = alpha * xv[i  ] + yv[i  ];
			wv[i+1] = alpha * xv[i+1] + yv[i+1];
			wv[i+2] = alpha * xv[i+2] + yv[i+2];
			wv[i+3] = alpha * xv[i+3] + yv[i+3];
		}
 	 } else  {
#ifndef HPCG_NO_OPENMP
 	   #pragma omp parallel for
#endif
		 for (local_int_t i=0; i<n; i+=4) {
			wv[i  ] = alpha * xv[i  ] + beta * yv[i  ];
			wv[i+1] = alpha * xv[i+1] + beta * yv[i+1];
			wv[i+2] = alpha * xv[i+2] + beta * yv[i+2];
			wv[i+3] = alpha * xv[i+3] + beta * yv[i+3];
		 }
 	 }
  } else if ( n % 2 == 0 ) {
 	/*
 	 * Unroll these loops
 	 */
  	if (alpha==1.0 and beta==1.0) {
#ifndef HPCG_NO_OPENMP
		#pragma omp parallel for
#endif
		for (local_int_t i = 0; i<n; i+=2) {
			wv[i  ] = xv[i  ] + yv[i  ];
			wv[i+1] = xv[i+1] + yv[i+1];
		}
	  } else if (alpha==1.0) {
#ifndef HPCG_NO_OPENMP
		#pragma omp parallel for
#endif
	    for (local_int_t i=0; i<n; i+=2) {
			wv[i  ] = xv[i  ] + beta * yv[i  ];
			wv[i+1] = xv[i+1] + beta * yv[i+1];
		}
 	 } else if (beta==1.0) {
#ifndef HPCG_NO_OPENMP
  	  #pragma omp parallel for
#endif
	    for (local_int_t i=0; i<n; i+=2) {
			wv[i  ] = alpha * xv[i  ] + yv[i  ];
			wv[i+1] = alpha * xv[i+1] + yv[i+1];
		}
 	 } else  {
#ifndef HPCG_NO_OPENMP
 	   #pragma omp parallel for
#endif
		 for (local_int_t i=0; i<n; i+=2) {
			wv[i  ] = alpha * xv[i  ] + beta * yv[i  ];
			wv[i+1] = alpha * xv[i+1] + beta * yv[i+1];
		 }
 	 }
  } else {
  	if (alpha==1.0 and beta==1.0) {
#ifndef HPCG_NO_OPENMP
		#pragma omp parallel for
#endif
		for (local_int_t i = 0; i<n; i++) {
			wv[i  ] = xv[i  ] + yv[i  ];
		}
	  } else if (alpha==1.0) {
#ifndef HPCG_NO_OPENMP
		#pragma omp parallel for
#endif
	    for (local_int_t i=0; i<n; i++) {
			wv[i  ] = xv[i  ] + beta * yv[i  ];
		}
 	 } else if (beta==1.0) {
#ifndef HPCG_NO_OPENMP
  	  #pragma omp parallel for
#endif
	    for (local_int_t i=0; i<n; i++) {
			wv[i  ] = alpha * xv[i  ] + yv[i  ];
		}
 	 } else  {
#ifndef HPCG_NO_OPENMP
 	   #pragma omp parallel for
#endif
		 for (local_int_t i=0; i<n; i++) {
			wv[i  ] = alpha * xv[i  ] + beta * yv[i  ];
		 }
 	 }
  }

  return 0;
}
