
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
  for (local_int_t i=0; i< nrow; i+=2)  {
    double sum1 = 0.0;
	double sum2 = 0.0;
    const int cur_nnz1 = A.nonzerosInRow[i  ];
	const int cur_nnz2 = A.nonzerosInRow[i+1];
	const double * const val1 = A.matrixValues[i  ];
	const double * const val2 = A.matrixValues[i+1];
	const local_int_t * const ind1 = A.mtxIndL[i  ];
	const local_int_t * const ind2 = A.mtxIndL[i+1];
	/*
	 * We now that nnz <= 27, and actually
	 * the amount of different values it can have is not that big...
	 * Experimentation told me that the only possible numbers are
	 * 8, 12, 18 and 27
	 *
	 * FIXME: Don't hardcode numbers, do the unrolling dynamically
	 */
	// First eight values can be done safely
	sum1 += val1[0] * xv[ind1[0]];
	sum1 += val1[1] * xv[ind1[1]];
	sum1 += val1[2] * xv[ind1[2]];
	sum1 += val1[3] * xv[ind1[3]];
	sum1 += val1[4] * xv[ind1[4]];
	sum1 += val1[5] * xv[ind1[5]];
	sum1 += val1[6] * xv[ind1[6]];
	sum1 += val1[7] * xv[ind1[7]];
	if (cur_nnz1 >= 12) {
		sum1 += val1[ 8] * xv[ind1[ 8]];
		sum1 += val1[ 9] * xv[ind1[ 9]];
		sum1 += val1[10] * xv[ind1[10]];
		sum1 += val1[11] * xv[ind1[11]];
	}
	if (cur_nnz1 >= 18) {
		sum1 += val1[12] * xv[ind1[12]];
		sum1 += val1[13] * xv[ind1[13]];
		sum1 += val1[14] * xv[ind1[14]];
		sum1 += val1[15] * xv[ind1[15]];
		sum1 += val1[16] * xv[ind1[16]];
		sum1 += val1[17] * xv[ind1[17]];
	}
	if (cur_nnz1 == 27) {
		sum1 += val1[18] * xv[ind1[18]];
		sum1 += val1[19] * xv[ind1[19]];
		sum1 += val1[20] * xv[ind1[20]];
		sum1 += val1[21] * xv[ind1[21]];
		sum1 += val1[22] * xv[ind1[22]];
		sum1 += val1[23] * xv[ind1[23]];
		sum1 += val1[24] * xv[ind1[24]];
		sum1 += val1[25] * xv[ind1[25]];
		sum1 += val1[26] * xv[ind1[26]];
	}

	sum2 += val2[0] * xv[ind2[0]];
	sum2 += val2[1] * xv[ind2[1]];
	sum2 += val2[2] * xv[ind2[2]];
	sum2 += val2[3] * xv[ind2[3]];
	sum2 += val2[4] * xv[ind2[4]];
	sum2 += val2[5] * xv[ind2[5]];
	sum2 += val2[6] * xv[ind2[6]];
	sum2 += val2[7] * xv[ind2[7]];
	if (cur_nnz2 >= 12) {
		sum2 += val2[ 8] * xv[ind2[ 8]];
		sum2 += val2[ 9] * xv[ind2[ 9]];
		sum2 += val2[10] * xv[ind2[10]];
		sum2 += val2[11] * xv[ind2[11]];
	}
	if (cur_nnz2 >= 18) {
		sum2 += val2[12] * xv[ind2[12]];
		sum2 += val2[13] * xv[ind2[13]];
		sum2 += val2[14] * xv[ind2[14]];
		sum2 += val2[15] * xv[ind2[15]];
		sum2 += val2[16] * xv[ind2[16]];
		sum2 += val2[17] * xv[ind2[17]];
	}
	if (cur_nnz2 == 27) {
		sum2 += val2[18] * xv[ind2[18]];
		sum2 += val2[19] * xv[ind2[19]];
		sum2 += val2[20] * xv[ind2[20]];
		sum2 += val2[21] * xv[ind2[21]];
		sum2 += val2[22] * xv[ind2[22]];
		sum2 += val2[23] * xv[ind2[23]];
		sum2 += val2[24] * xv[ind2[24]];
		sum2 += val2[25] * xv[ind2[25]];
		sum2 += val2[26] * xv[ind2[26]];
	}
    yv[i] = sum1;
	yv[i+1] = sum2;
  }
  return 0;
}
