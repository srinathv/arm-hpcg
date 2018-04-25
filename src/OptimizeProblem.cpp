/*!
 @file OptimizeProblem.cpp

 HPCG routine
 */

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

#include "OptimizeProblem.hpp"
/*!
  Optimizes the data structures used for CG iteration to increase the
  performance of the benchmark version of the preconditioned CG algorithm.

  @param[inout] A      The known system matrix, also contains the MG hierarchy in attributes Ac and mgData.
  @param[inout] data   The data structure with all necessary CG vectors preallocated
  @param[inout] b      The known right hand side vector
  @param[inout] x      The solution vector to be computed in future CG iteration
  @param[inout] xexact The exact solution vector

  @return returns 0 upon success and non-zero otherwise

  @see GenerateGeometry
  @see GenerateProblem
*/
int OptimizeProblem(SparseMatrix & A, CGData & data, Vector & b, Vector & x, Vector & xexact) {

#ifdef HPCG_USE_COLOR_REORDERING
	const local_int_t nrow = A.localNumberOfRows;
	std::vector<local_int_t> colors(nrow, nrow);
	int totalColors = 1;
	colors[0] = 0;

	/*
	 * This loop assigns a color to each node
	 */
	for ( local_int_t i = 1; i < nrow; i++ ) {
		if ( colors[i] == nrow ) {
			std::vector<int> assigned(totalColors, 0);
			int currentlyAssigned = 0;
			const local_int_t * const currentColIndices = A.mtxIndL[i];
			const int currentNumberOfNonzeros = A.nonzerosInRow[i];

			/*
			 * Check colors assigned to eigh neighbor
			 */
			for ( int j = 0; j < currentNumberOfNonzeros; j++ ) {
				local_int_t curCol = currentColIndices[j];
				if ( curCol < i ) {
					if ( assigned[colors[curCol]] == 0 ) {
						currentlyAssigned += 1;
					}
					assigned[colors[curCol]] = 1;
				}
			}

			if ( currentlyAssigned < totalColors ) {
				// Look for the smallest color available and assign it
				for ( int j = 0; j < totalColors; j++ ) {
					if ( assigned[j] == 0 ) {
						colors[i] = j;
						break;
					}
				}
			} else {
				// All the available colors are present in neighbors,
				// therefore, create a new color and assigned
				if ( colors[i] == nrow ) {
					colors[i] = totalColors;
					totalColors += 1;
				}
			}
		}
	}

	/*
	 * Till here, colors[i] contains the color of row i
	 */
	std::vector<local_int_t> counters(totalColors, 0);
	for ( local_int_t i = 0; i < nrow; i++ ) {
		counters[colors[i]]++;
	}

	/*
	 * Till here counteres[i] contains number of rows assigned to color i
	 */

	std::vector<local_int_t> colorIndices(totalColors, 0);
	local_int_t old = 0, old0;
	for ( int i = 0; i < totalColors; i++ ) {
		old0 = counters[i];
		counters[i] = counters[i-1] + old;
		colorIndices[i] = counters[i];
		old = old0;
	}
	counters[0] = 0;
	colorIndices[0] = 0;

	// computationOrder will contain the order on which the ros should be
	// computed. i.e., rows ordered by color. This is what will be used during
	// the SYMGS
	std::vector<local_int_t> computationOrder(nrow);
	local_int_t j = 0;
	for ( int ic = 0; ic < colorIndices.size(); ic++ ) {
		for ( local_int_t i = 0; i < nrow; i++ ) {
			if ( colors[i] == ic ) {
				computationOrder[j] = i;
				j++;
			}
		}
	}

	/*
	 * Till here:
	 *     colorIndices[i] contains the first row with color i after reordering
	 */

	/*
	 * Store the information to be used on the different kernels
	 */
	A.optimizationData[0] = std::vector<local_int_t>(computationOrder);
	A.optimizationData[1] = std::vector<local_int_t>(colorIndices);

	// Colorize the coarser levels
	if ( A.mgData != 0 ) {
		OptimizeProblem((SparseMatrix &) *A.Ac, data, b, x, xexact);
	}
#elif defined(HPCG_USE_SLICE_COLOR_REORDERING)
    /*
     * Apply block multicoloring reodering to the matrix in order to
     * parallelize the symmetric Gauss-Seidel.
     *
     * The idea is to have blocks of size nx*ny*LAYERS_PER_BLOCK (i.e., slices
     * of the grid)
     * At the end, we end up having nz/LAYERS_PER_BLOCK blocks
     * Therefore, nz%LAYERS_PER_BLOCK must be 0 (it is asserted in the code)
     *
     * The total amount of colors is equal to NCOLORS, so
     * (nz/LAYERS_PER_BLOCK) % NCOLORS must be 0 to have all colors with the
     * same amount of blocks (also asserted in the code)
     */
    const local_int_t nrow = A.localNumberOfRows;

    const local_int_t blockSize = A.geom->nx * A.geom->ny * LAYERS_PER_BLOCK;
    const local_int_t nBlocks = A.geom->nz / LAYERS_PER_BLOCK;

    assert(A.geom->nz % LAYERS_PER_BLOCK == 0);
    assert(nBlocks % NCOLORS == 0);

    // Initialize a vector containing the color of block N
    // -1 means no color is assigned
    std::vector<local_int_t> colors(nBlocks, -1);

    // Simplest way to assign colors is to assign consecutive colors to
    // consecutive blocks. When color == NCOLORS, start from color 0 again
    local_int_t colorToAssign = 0;
    for ( local_int_t i = 0; i < nBlocks; i++ ) {
        colors[i] = colorToAssign;
        colorToAssign++;
        if ( colorToAssign == NCOLORS ) {
            colorToAssign = 0;
        }
    }

    /*
     * Till here we have colored each block, color of each block is at colors
     *
     * Now we have to create a structure of data to know the order on which
     * computations will be performed
     */
    std::vector<std::vector<local_int_t> > computationOrder(NCOLORS);
    for ( local_int_t i = 0; i < NCOLORS; i++ ) {
        for ( local_int_t j = 0; j < nBlocks; j++ ) {
            if ( colors[j] == i ) {
                computationOrder[i].push_back(j);
            }
        }
    }

    /*
     * Store the information to be used on the kernels
     */
    A.optimizationData = std::vector<std::vector<local_int_t> >(computationOrder);

    // If there are coarser levels, colorize them also
    if ( A.mgData != 0 ) {
        OptimizeProblem( (SparseMatrix &) *A.Ac, data, b, x, xexact);
    }
#endif

	return 0;
}

// Helper function (see OptimizeProblem.hpp for details)
double OptimizeProblemMemoryUse(const SparseMatrix & A) {

  return 0.0;

}
