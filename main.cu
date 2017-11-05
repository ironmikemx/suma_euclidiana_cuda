#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/gather.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <stdio.h>
#include "data.cuh"
using namespace thrust::placeholders;


/*

void load(thrust::device_vector<int> & data, thrust::device_vector<int> & vec, const int u, const int m) {
  thrust::copy(data.begin(), data.begin() + (u*m), vec.begin());
  return;
}

*/

/********/
/* MAIN */
/********/
int main(int argc, char** argv)
{


std::cout << argv[1] << "\n";
std::cout << argv[2] << "\n";


	/**************************/
	/* SETTING UP THE PROBLEM */
	/**************************/

       // int usuario_id =80;  
	//const int N_movies_orig		= 3;			// --- Number of vector elements
	//const int N_users_orig	= 5;			// --- Number of vectors for each matrix

        const int N_users_orig = atoi(argv[1]);
        const int N_movies_orig = atoi(argv[2]);

        int usuario_id = atoi(argv[3]);

	// --- Matrix allocation and initialization

//	thrust::device_vector<int> d_matrixA(N_users_orig * N_movies_orig);
//	thrust::device_vector<int> d_matrixB(N_users_orig * N_movies_orig);

        thrust::device_vector<int> d_matrixA(N_users_orig * N_movies_orig);
        load(d_matrixA, N_users_orig,N_movies_orig);

	printf("\n\nmatrixA\n");
	for(int i = 0; i < N_users_orig; i++) {
		std::cout << " [ ";
		for(int j = 0; j < N_movies_orig; j++)
			std::cout << d_matrixA[i * N_movies_orig + j]%100 << " ";
		std::cout << "]\n";
	}


        
       


	return 0; 
}
