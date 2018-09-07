
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
 @file ComputeSYMGS.cpp

 HPCG routine
 */

#include "ComputeSYMGS.hpp"
#include "ComputeSYMGS_ref.hpp"
#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif
#include <cassert>
#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

/*!
  Routine to compute one step of symmetric Gauss-Seidel:

  Assumption about the structure of matrix A:
  - Each row 'i' of the matrix has nonzero diagonal value whose address is matrixDiagonal[i]
  - Entries in row 'i' are ordered such that:
       - lower triangular terms are stored before the diagonal element.
       - upper triangular terms are stored after the diagonal element.
       - No other assumptions are made about entry ordering.

  Symmetric Gauss-Seidel notes:
  - We use the input vector x as the RHS and start with an initial guess for y of all zeros.
  - We perform one forward sweep.  Since y is initially zero we can ignore the upper triangular terms of A.
  - We then perform one back sweep.
       - For simplicity we include the diagonal contribution in the for-j loop, then correct the sum after

  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On entry, x should contain relevant values, on exit x contains the result of one symmetric GS sweep with r as the RHS.

  @return returns 0 upon success and non-zero otherwise

  @warning Early versions of this kernel (Version 1.1 and earlier) had the r and x arguments in reverse order, and out of sync with other kernels.

  @see ComputeSYMGS_ref
*/
int ComputeSYMGS( const SparseMatrix & A, const Vector & r, Vector & x) {
	assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

#ifndef HPCG_NO_MPI
	ExchangeHalo(A,x);
#endif

	const local_int_t nrow = A.localNumberOfRows;
	double ** matrixDiagonal = A.matrixDiagonal;  // An array of pointers to the diagonal entries A.matrixValues
	const double * const rv = r.values;
	double * const xv = x.values;

	// Check some premises
	// 1. number of threads is a divisor of nz
	assert(A.geom->nz % A.geom->numThreads  == 0);
	// 2. Number of colors to be used is a divisor of nz
	assert(A.geom->nz % (A.geom->nz / A.geom->numThreads) == 0);

	const local_int_t numberOfColors = A.geom->nz / A.geom->numThreads;
	const local_int_t slicesPerColor = A.geom->nz / numberOfColors;
	const local_int_t sliceSize = A.geom->nx * A.geom->ny;

	for ( local_int_t color = 0; color < numberOfColors; color++ ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t k = 0; k < slicesPerColor; k++ ) {
#ifndef HPCG_NO_OPENMP
			local_int_t firstRow = sliceSize * numberOfColors * omp_get_thread_num() + color * sliceSize;
#else
			local_int_t firstRow = color * sliceSize;
#endif
			local_int_t lastRow = firstRow + sliceSize;

			// Forward sweep
			for ( local_int_t i = firstRow; i < lastRow; i++ ) {
				const double *const currentValues = A.matrixValues[i];
				const local_int_t *const currentColIndices = A.mtxIndL[i];
				const int currentNumberOfNonzeros = A.nonzerosInRow[i];
				const double currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value

				double sum = rv[i]; //RHS value
				for ( local_int_t j = 0; j < currentNumberOfNonzeros; j++ ) {
					sum -= currentValues[j] * xv[currentColIndices[j]];
				}
				sum += xv[i] * currentDiagonal; // Remove diagonal contribution from previous loop
				xv[i] = sum / currentDiagonal;
			}
		}
	}
	for ( local_int_t color = numberOfColors - 1; color >= 0; color-- ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t k = slicesPerColor - 1; k >= 0; k-- ) {
#ifndef HPCG_NO_OPENMP
			local_int_t firstRow = sliceSize * numberOfColors * omp_get_thread_num() + color * sliceSize;
#else
			local_int_t firstRow = color * sliceSize;
#endif
			local_int_t lastRow = firstRow + sliceSize;

			// Back sweep
			for ( local_int_t i = lastRow - 1; i >= firstRow; i-- ) {
				const double *const currentValues = A.matrixValues[i];
				const local_int_t *const currentColIndices = A.mtxIndL[i];
				const int currentNumberOfNonzeros = A.nonzerosInRow[i];
				const double currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value

				double sum = rv[i]; // RHS value
				for ( local_int_t j = 0; j < currentNumberOfNonzeros; j++ ) {
					sum -= currentValues[j] * xv[currentColIndices[j]];
				}
				sum += xv[i] * currentDiagonal; // Remove diagonal contribution from previous loop
				xv[i] = sum / currentDiagonal;
			}
		}
	}

	return 0;

}
