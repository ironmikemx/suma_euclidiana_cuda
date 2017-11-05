#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/gather.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <stdio.h>
using namespace thrust::placeholders;




void load(thrust::device_vector<int> & data, thrust::device_vector<int> & vec, const int u, const int m) {
  thrust::copy(data.begin(), data.begin() + (u*m), vec.begin());
  return;
}


/********/
/* MAIN */
/********/
int main(int argc, char** argv)
{


std::cout << argv[1] << "\n";
std::cout << argv[2] << "\n";

thrust::device_vector<int> data(80);

/* user 0*/     data[0] = 5;    data[1] = 3;    data[2] = 4;    data[3] = 3;    data[4] = 3;    data[5] = 5;    data[6] = 4;    data[7] = 1;
/* user 1*/     data[8] = 4;    data[9] = 0;    data[10] = 0;   data[11] = 0;   data[12] = 0;   data[13] = 0;   data[14] = 0;   data[15] = 0;
/* user 2*/     data[16] = 0;   data[17] = 0;   data[18] = 0;   data[19] = 0;   data[20] = 0;   data[21] = 0;   data[22] = 0;   data[23] = 0;
/* user 3*/     data[24] = 0;   data[25] = 0;   data[26] = 0;   data[27] = 0;   data[28] = 0;   data[29] = 0;   data[30] = 0;   data[31] = 0;
/* user 4*/     data[32] = 4;   data[33] = 3;   data[34] = 0;   data[35] = 0;   data[36] = 0;   data[37] = 0;   data[38] = 0;   data[39] = 0;
/* user 5*/     data[40] = 4;   data[41] = 0;   data[42] = 0;   data[43] = 0;   data[44] = 0;   data[45] = 0;   data[46] = 2;   data[47] = 4;
/* user 6*/     data[48] = 0;   data[49] = 0;   data[50] = 0;   data[51] = 5;   data[52] = 0;   data[53] = 0;   data[54] = 5;   data[55] = 5;
/* user 7*/     data[56] = 0;   data[57] = 0;   data[58] = 0;   data[59] = 0;   data[60] = 0;   data[61] = 0;   data[62] = 3;   data[63] = 0;
/* user 8*/     data[64] = 0;   data[65] = 0;   data[66] = 0;   data[67] = 0;   data[68] = 0;   data[69] = 5;   data[70] = 4;   data[71] = 0;
/* user 9*/     data[72] = 4;   data[73] = 0;   data[74] = 0;   data[75] = 4;   data[76] = 0;   data[77] = 0;   data[78] = 4;   data[79] = 0;


	/**************************/
	/* SETTING UP THE PROBLEM */
	/**************************/

        int usuario_id =80;  
	//const int N_movies_orig		= 3;			// --- Number of vector elements
        const int N_movies_orig = atoi(argv[1]);        
        std::cout << "movies:" << N_movies_orig << "\n";
	const int N_users_orig	= 5;			// --- Number of vectors for each matrix


	// --- Matrix allocation and initialization

//	thrust::device_vector<int> d_matrixA(N_users_orig * N_movies_orig);
//	thrust::device_vector<int> d_matrixB(N_users_orig * N_movies_orig);

        thrust::device_vector<int> d_matrixA(N_users_orig * N_movies_orig);
        load(data, d_matrixA, N_users_orig,N_movies_orig);

	printf("\n\nmatrixA\n");
	for(int i = 0; i < N_users_orig; i++) {
		std::cout << " [ ";
		for(int j = 0; j < N_movies_orig; j++)
			std::cout << d_matrixA[i * N_movies_orig + j]%100 << " ";
		std::cout << "]\n";
	}


        
       


	return 0; 
}
