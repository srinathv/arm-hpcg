
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
 @file ComputeSPMV.cpp

 HPCG routine
 */

#include "ComputeSPMV.hpp"
#include "ComputeSPMV_ref.hpp"
#include "ExchangeHalo.hpp"
#include "hpcg.hpp"

__attribute__((optimize("unroll-loops")))
inline void SPMV_unroll(double *sum, const double *val, const double *xv, const local_int_t *ind, const int cur_nnz) {
	for ( int i = 0; i<cur_nnz; ++i ) {
		*sum += val[i] * xv[ind[i]];
	}
}


/*!
  Routine to compute sparse matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This routine calls the reference SpMV implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV_ref
*/
int ComputeSPMV( const SparseMatrix & A, Vector & x, Vector & y) {
  assert(x.localLength>=A.localNumberOfColumns); // Test vector lengths
  assert(y.localLength>=A.localNumberOfRows);

#ifndef HPCG_NO_MPI
    ExchangeHalo(A,x);
#endif
  const double * const xv = x.values;
  double * const yv = y.values;
  const local_int_t nrow = A.localNumberOfRows;

  /*
   * We can perform 2 unroll assuming nx % 2 = ny % 2 = nz % 2 = 0,
   * which seems quite reasonable
   */
#ifndef HPCG_NO_OPENMP
  #pragma omp parallel for
#endif
  for (local_int_t i=0; i < nrow; i+=2)  {
    double sum1 = 0.0;
	double sum2 = 0.0;
    const int cur_nnz1 = A.nonzerosInRow[i  ];
	const int cur_nnz2 = A.nonzerosInRow[i+1];
	const double * const val1 = A.matrixValues[i  ];
	const double * const val2 = A.matrixValues[i+1];
	const local_int_t * const ind1 = A.mtxIndL[i  ];
	const local_int_t * const ind2 = A.mtxIndL[i+1];

	/*
	 * Unroll only these loops. For that, we will specify an inline
	 * function for them with an attribute
	 */
	SPMV_unroll(&sum1, val1, xv, ind1, cur_nnz1);
	SPMV_unroll(&sum2, val2, xv, ind2, cur_nnz2);

    yv[i] = sum1;
	yv[i+1] = sum2;
  }
  return 0;
}
