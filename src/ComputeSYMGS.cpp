
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

/*
 * Copyright (c) 2018, Barcelona Supercomputing Center
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met: 
 * redistributions of source code must retain the above copyright notice, this 
 * list of conditions and the following disclaimer; redistributions in binary form
 * must reproduce the above copyright notice, this list of conditions and the 
 * following disclaimer in the documentation and/or other materials provided with 
 * the distribution; neither the name of the copyright holder nor the names of its 
 * contributors may be used to endorse or promote products derived from this 
 * software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*!
  @file ComputeSYMGS.cpp

  HPCG routine
  */

#include "ComputeSYMGS.hpp"
#include "ComputeSYMGS_ref.hpp"

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

	// Make sure x contain space for halo values
	assert( x.localLength == A.localNumberOfColumns);

#ifndef HPCG_NO_MPI
	ExchangeHalo(A, x);
#endif

	const local_int_t nrow = A.localNumberOfRows;
	double **matrixDiagonal = A.matrixDiagonal;
	const double *const rv = r.values;
	double *const xv = x.values;

#if HPCG_USE_COLOR_REORDERING
	// Grab the coloring information
	std::vector<local_int_t> computationOrder(A.optimizationData[0]);
	std::vector<local_int_t> colorIndices(A.optimizationData[1]);

	// for each color, do the fwd sweep
	for ( int ic = 0; ic < colorIndices.size(); ic++ ) {
		int firstInd = colorIndices[ic];
		int lastInd = (ic != colorIndices.size()-1) ? colorIndices[ic+1] : nrow;
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t i = firstInd; i < lastInd; i++ ) {
			const double *const currentValues = A.matrixValues[computationOrder[i]];
			const local_int_t *const currentColIndices = A.mtxIndL[computationOrder[i]];
			const int currentNumberOfNonzeros = A.nonzerosInRow[computationOrder[i]];
			const double currentDiagonal = matrixDiagonal[computationOrder[i]][0];

			double sum = rv[computationOrder[i]];
			for ( int j = 0; j < currentNumberOfNonzeros; j++ ) {
				local_int_t curCol = currentColIndices[j];
				sum -= currentValues[j] * xv[curCol];
			}
			sum += xv[computationOrder[i]] * currentDiagonal;
			xv[computationOrder[i]] = sum / currentDiagonal;
		}
	}

	// Now the back sweep
	for ( int ic = colorIndices.size() - 1; ic >= 0; ic-- ) {
		int firstInd = colorIndices[ic];
		int lastInd = (ic != colorIndices.size()-1) ? colorIndices[ic+1] : nrow;

#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t i = lastInd - 1; i >= firstInd; i-- ) {
			const double *const currentValues = A.matrixValues[computationOrder[i]];
			const local_int_t *const currentColIndices = A.mtxIndL[computationOrder[i]];
			const int currentNumberOfNonzeros = A.nonzerosInRow[computationOrder[i]];
			const double currentDiagonal = matrixDiagonal[computationOrder[i]][0];

			double sum = rv[computationOrder[i]];
			for ( int j = currentNumberOfNonzeros - 1; j >= 0; j-- ) {
				local_int_t curCol = currentColIndices[j];
				sum -= currentValues[j] * xv[curCol];
			}
			sum += xv[computationOrder[i]] * currentDiagonal;
			xv[computationOrder[i]] = sum / currentDiagonal;
		}
	}
#elif defined(HPCG_USE_SLICE_COLOR_REORDERING)
	// Grab the optimization data
	std::vector<std::vector<local_int_t> > computationOrder(A.optimizationData);

	/*
	 * NCOLORS = amount of colors and size of computationOrder
	 * LAYERS_PER_BLOCK * nx * ny = blockSize
	 */
	local_int_t blockSize = A.geom->nx * A.geom->ny * LAYERS_PER_BLOCK;

	// forward sweep
	for ( local_int_t color = 0; color < NCOLORS; color++ ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t k = 0; k < computationOrder[color].size(); k++ ) {
			local_int_t firstRow = computationOrder[color][k] * blockSize;
			local_int_t lastRow = firstRow + blockSize;
			for ( local_int_t i = firstRow; i < lastRow; i++ ) {
				const double *const currentValues = A.matrixValues[i];
				const local_int_t *const currentColIndices = A.mtxIndL[i];
				const int currentNumberOfNonzeros = A.nonzerosInRow[i];
				const double currentDiagonal = matrixDiagonal[i][0];
				double sum = rv[i];
				for ( int j = 0; j < currentNumberOfNonzeros; j++ ) {
					local_int_t curCol = currentColIndices[j];
					sum -= currentValues[j] * xv[curCol];
				}

				sum += xv[i] * currentDiagonal;
				xv[i] = sum / currentDiagonal;
			}
		}
	}

	// back sweep
	for ( local_int_t color = NCOLORS - 1; color >= 0; color-- ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t k = computationOrder[color].size() - 1; k >= 0; k-- ) {
			local_int_t firstRow = computationOrder[color][k] * blockSize;
			local_int_t lastRow = firstRow + blockSize;
			for ( local_int_t i = lastRow - 1; i >= firstRow; i-- ) {
				const double *const currentValues = A.matrixValues[i];
				const local_int_t *const currentColIndices = A.mtxIndL[i];
				const int currentNumberOfNonzeros = A.nonzerosInRow[i];
				const double currentDiagonal = matrixDiagonal[i][0];
				double sum = rv[i];

				for ( int j = currentNumberOfNonzeros - 1; j >= 0; j-- ) {
					local_int_t curCol = currentColIndices[j];
					sum -= currentValues[j] * xv[curCol];
				}

				sum += xv[i] * currentDiagonal;
				xv[i] = sum / currentDiagonal;
			}
		}
	}
#else
	// forward sweep
	for ( local_int_t i = 0; i < nrow; i++ ) {
		const double *const currentValues = A.matrixValues[i];
		const local_int_t *const currentColIndices = A.mtxIndL[i];
		const int currentNumberOfNonzeros = A.nonzerosInRow[i];
		const double currentDiagonal = matrixDiagonal[i][0];

		double sum = rv[i];
		for ( int j = 0; j < currentNumberOfNonzeros; j++ ) {
			local_int_t curCol = currentColIndices[j];
			sum -= currentValues[j] * xv[curCol];
		}
		sum += xv[i] * currentDiagonal;
		xv[i] = sum / currentDiagonal;
	}

	// back sweep
	for ( local_int_t i = nrow - 1; i >= 0; i-- ) {
		const double *const currentValues = A.matrixValues[i];
		const local_int_t *const currentColIndices = A.mtxIndL[i];
		const int currentNumberOfNonzeros = A.nonzerosInRow[i];
		const double currentDiagonal = matrixDiagonal[i][0];

		double sum = rv[i];
		for ( int j = currentNumberOfNonzeros - 1; j >= 0; j-- ) {
			local_int_t curCol = currentColIndices[j];
			sum -= currentValues[j] * xv[curCol];
		}

		sum += xv[i] * currentDiagonal;
		xv[i] = sum / currentDiagonal;
	}
#endif

	return 0;
}
