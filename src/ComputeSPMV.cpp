
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
#include <cassert>
#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

int ComputeSPMV( const SparseMatrix & A, Vector & x, Vector & y) {

	assert(x.localLength >= A.localNumberOfColumns);
	assert(y.localLength >= A.localNumberOfRows);

#ifndef HPCG_NO_MPI
	ExchangeHalo(A,x);
#endif
	const double * const xv = x.values;
	double * const yv = y.values;
	const local_int_t nrow = A.localNumberOfRows;

#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
	for ( local_int_t i = 0; i < nrow; i++ ) {
		double sum = 0.0;
		for ( local_int_t j = 0; j < A.nonzerosInRow[i]; j++ ) {
			local_int_t curCol = A.mtxIndL[i][j];
			sum += A.matrixValues[i][j] * xv[curCol];
		}
		yv[i] = sum;
	}

	return 0;
}
