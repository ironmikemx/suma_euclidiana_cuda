#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/gather.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <stdio.h>

using namespace thrust::placeholders;

/****************************************************/
/* POWER DIFFERENCE FUNCTOR FOR EUCLIDEAN DISTANCES */
/****************************************************/
struct PowerDifference {
	__host__ __device__ float operator()(const float& a, const float& b) const { 
        if ( fmodf(a,100) == 0.0f || fmodf(b,100)== 0.0f) {
          return 0.0f;
        } else {
          return pow(fmodf(a,100) - fmodf(b,100), 2); 
        }
   }
};


struct countIfNoZeros {
        __host__ __device__ float operator()(const float& a, const float& b) const {
        if ( fmodf(a,100) > 0.0f && fmodf(b,100) > 0.0f) {
          return 1.0f;
        } else {
          return 0.0f;
        }
   }
};

struct copy_if_value {
        __host__ __device__ float operator()(const float& a, const float& b) const {
        if ( fmodf(a,100) > 0.0f) {
          return b;
        } else {
          return 9999999999.0f;
        }
   }
};


// note: functor inherits from unary_function
struct not_99 : public thrust::unary_function<float,float>
{
  __host__ __device__
  bool operator()(float x) const
  {
    return x < 9999999999.0f;
  }
};



/*******************/
/* EXPAND OPERATOR */
/*******************/
template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator expand(InputIterator1 first1,
                      InputIterator1 last1,
                      InputIterator2 first2,
                      OutputIterator output)
{
	typedef typename thrust::iterator_difference<InputIterator1>::type difference_type;
  
	difference_type input_size  = thrust::distance(first1, last1);
	difference_type output_size = thrust::reduce(first1, last1);

	// scan the counts to obtain output offsets for each input element
	thrust::device_vector<difference_type> output_offsets(input_size, 0);
	thrust::exclusive_scan(first1, last1, output_offsets.begin()); 

	// scatter the nonzero counts into their corresponding output positions
	thrust::device_vector<difference_type> output_indices(output_size, 0);
	thrust::scatter_if(thrust::counting_iterator<difference_type>(0), thrust::counting_iterator<difference_type>(input_size),
					   output_offsets.begin(), first1, output_indices.begin());

	// compute max-scan over the output indices, filling in the holes
	thrust::inclusive_scan(output_indices.begin(), output_indices.end(), output_indices.begin(), thrust::maximum<difference_type>());

	// gather input values according to index array (output = first2[output_indices])
	OutputIterator output_end = output; thrust::advance(output_end, output_size);
	thrust::gather(output_indices.begin(), output_indices.end(), first2, output);

	// return output + output_size
	thrust::advance(output, output_size);
  
	return output;
}

/********/
/* MAIN */
/********/
int main()
{
	/**************************/
	/* SETTING UP THE PROBLEM */
	/**************************/
  
	const int N_movies_orig		= 20;			// --- Number of vector elements
	const int N_users_orig	= 3;			// --- Number of vectors for each matrix

	// --- Random uniform integer distribution between 0 and 100
	thrust::default_random_engine rng;
	thrust::uniform_int_distribution<int> dist(0, 20);

	// --- Matrix allocation and initialization

	thrust::device_vector<float> d_matrixA(N_users_orig * N_movies_orig);
	thrust::device_vector<float> d_matrixB(N_users_orig * N_movies_orig);

        d_matrixA[0]  =  1000000;
        d_matrixA[1]  =  2000101;
        d_matrixA[2]  =  3000202;
        d_matrixA[3]  =  4000303;
        d_matrixA[4]  =  5000405;
        d_matrixA[5]  =  6000500;
        d_matrixA[6]  =  7000600;
        d_matrixA[7]  =  8000700;
        d_matrixA[8]  =  9000800;
        d_matrixA[9]  =  10000900;
        d_matrixA[10]  =  11001000;
        d_matrixA[11]  =  12001100;
        d_matrixA[12]  =  13001200;
        d_matrixA[13]  =  14001300;
        d_matrixA[14]  =  15001400;
        d_matrixA[15]  =  16001500;
        d_matrixA[16]  =  17001600;
        d_matrixA[17]  =  18001700;
        d_matrixA[18]  =  19001800;
        d_matrixA[19]  =  20001900;
        d_matrixA[20]  =  21000000;
        d_matrixA[21]  =  22000102;
        d_matrixA[22]  =  23000202;
        d_matrixA[23]  =  24000303;
        d_matrixA[24]  =  25000400;
        d_matrixA[25]  =  26000505;
        d_matrixA[26]  =  27000600;
        d_matrixA[27]  =  28000700;
        d_matrixA[28]  =  29000800;
        d_matrixA[29]  =  30000900;
        d_matrixA[30]  =  31001001;
        d_matrixA[31]  =  32001100;
        d_matrixA[32]  =  33001200;
        d_matrixA[33]  =  34001300;
        d_matrixA[34]  =  35001400;
        d_matrixA[35]  =  36001500;
        d_matrixA[36]  =  37001600;
        d_matrixA[37]  =  38001700;
        d_matrixA[38]  =  39001800;
        d_matrixA[39]  =  40001900;
        d_matrixA[40]  =  41000005;
        d_matrixA[41]  =  42000105;
        d_matrixA[42]  =  43000205;
        d_matrixA[43]  =  44000301;
        d_matrixA[44]  =  45000400;
        d_matrixA[45]  =  46000500;
        d_matrixA[46]  =  47000600;
        d_matrixA[47]  =  48000700;
        d_matrixA[48]  =  49000800;
        d_matrixA[49]  =  50000900;
        d_matrixA[50]  =  51001000;
        d_matrixA[51]  =  52001100;
        d_matrixA[52]  =  53001200;
        d_matrixA[53]  =  54001300;
        d_matrixA[54]  =  55001400;
        d_matrixA[55]  =  56001500;
        d_matrixA[56]  =  57001600;
        d_matrixA[57]  =  58001700;
        d_matrixA[58]  =  59001800;
        d_matrixA[59]  =  60001900;


        d_matrixB[0]  =  1000000;
        d_matrixB[1]  =  2000101;
        d_matrixB[2]  =  3000202;
        d_matrixB[3]  =  4000303;
        d_matrixB[4]  =  5000405;
        d_matrixB[5]  =  6000500;
        d_matrixB[6]  =  7000600;
        d_matrixB[7]  =  8000700;
        d_matrixB[8]  =  9000800;
        d_matrixB[9]  =  10000900;
        d_matrixB[10]  =  11001000;
        d_matrixB[11]  =  12001100;
        d_matrixB[12]  =  13001200;
        d_matrixB[13]  =  14001300;
        d_matrixB[14]  =  15001400;
        d_matrixB[15]  =  16001500;
        d_matrixB[16]  =  17001600;
        d_matrixB[17]  =  18001700;
        d_matrixB[18]  =  19001800;
        d_matrixB[19]  =  20001900;
        d_matrixB[20]  =  21000000;
        d_matrixB[21]  =  22000101;
        d_matrixB[22]  =  23000202;
        d_matrixB[23]  =  24000303;
        d_matrixB[24]  =  25000405;
        d_matrixB[25]  =  26000500;
        d_matrixB[26]  =  27000600;
        d_matrixB[27]  =  28000700;
        d_matrixB[28]  =  29000800;
        d_matrixB[29]  =  30000900;
        d_matrixB[30]  =  31001000;
        d_matrixB[31]  =  32001100;
        d_matrixB[32]  =  33001200;
        d_matrixB[33]  =  34001300;
        d_matrixB[34]  =  35001400;
        d_matrixB[35]  =  36001500;
        d_matrixB[36]  =  37001600;
        d_matrixB[37]  =  38001700;
        d_matrixB[38]  =  39001800;
        d_matrixB[39]  =  40001900;
        d_matrixB[40]  =  41000000;
        d_matrixB[41]  =  42000101;
        d_matrixB[42]  =  43000202;
        d_matrixB[43]  =  44000303;
        d_matrixB[44]  =  45000405;
        d_matrixB[45]  =  46000500;
        d_matrixB[46]  =  47000600;
        d_matrixB[47]  =  48000700;
        d_matrixB[48]  =  49000800;
        d_matrixB[49]  =  50000900;
        d_matrixB[50]  =  51001000;
        d_matrixB[51]  =  52001100;
        d_matrixB[52]  =  53001200;
        d_matrixB[53]  =  54001300;
        d_matrixB[54]  =  55001400;
        d_matrixB[55]  =  56001500;
        d_matrixB[56]  =  57001600;
        d_matrixB[57]  =  58001700;
        d_matrixB[58]  =  59001800;
        d_matrixB[59]  =  60001900;




        



	printf("\n\nFirst matrixA\n");
	for(int i = 0; i < N_users_orig; i++) {
		std::cout << " [ ";
		for(int j = 0; j < N_movies_orig; j++)
			std::cout << fmodf(d_matrixA[i * N_movies_orig + j],10000) << " ";
		std::cout << "]\n";
	}

	printf("\n\nSecond matrixB\n");
	for(int i = 0; i < N_users_orig; i++) {
		std::cout << " [ ";
		for(int j = 0; j < N_movies_orig; j++)
			std::cout << fmodf(d_matrixB[i * N_movies_orig + j],10000) << " ";
		std::cout << "]\n";
	}


        
      // Reducing matrixes

      thrust::device_vector<float> temp1(N_movies_orig*N_users_orig);
      thrust::device_vector<float> temp2(N_movies_orig*N_users_orig);
      thrust::device_vector<float> temp1sorted(N_movies_orig*N_users_orig);
      thrust::device_vector<float> temp2sorted(N_movies_orig*N_users_orig);
      thrust::transform(d_matrixB.begin(), d_matrixB.end(), d_matrixA.begin(), temp1.begin(), copy_if_value());
      thrust::transform(d_matrixB.begin(), d_matrixB.end(), d_matrixB.begin(), temp2.begin(), copy_if_value());


      thrust::sort_by_key(temp1.begin(), temp1.end(), temp1sorted.begin());
      thrust::sort_by_key(temp2.begin(), temp2.end(), temp2sorted.begin());

      int N = thrust::count_if( temp1.begin(), temp1.end(), not_99());
      int N_users = N_users_orig;
      int N_movies = N / N_users;


      std::cout << "MOVIES ="<< N << "]\n";

      thrust::device_vector<float> d_matrix1(N);
      thrust::device_vector<float> d_matrix2(N);
      
      thrust::copy_if(temp1.begin(), temp1.end(), d_matrix1.begin(), not_99());
      thrust::copy_if(temp2.begin(), temp2.end(), d_matrix2.begin(), not_99());


printf("\n\ntemp1 matrix\n");
        for(int i = 0; i < N_users_orig; i++) {
                std::cout << " [ ";
                for(int j = 0; j < N_movies; j++)
                        std::cout << temp1[i * N_movies_orig + j] << " ";
                std::cout << "]\n";
        }




 printf("\n\nFirst matrix\n");
        for(int i = 0; i < N_users; i++) {
                std::cout << " [ ";
                for(int j = 0; j < N_movies; j++)
                        std::cout << d_matrix1[i * N_movies + j] << " ";
                std::cout << "]\n";
        }

        printf("\n\nSecond matrix\n");
        for(int i = 0; i < N_users; i++) {
                std::cout << " [ ";
                for(int j = 0; j < N_movies; j++)
                        std::cout << d_matrix2[i * N_movies + j] << " ";
                std::cout << "]\n";
        }




	thrust::device_vector<int> d_sequence(N_users);
	thrust::device_vector<int> d_indices(N_users * N_movies);
	thrust::device_vector<int> d_counts(N_users, N_movies);
	thrust::sequence(d_sequence.begin(), d_sequence.begin() + N_users);
	expand(d_counts.begin(), d_counts.end(), d_sequence.begin(), d_indices.begin());

	printf("\n\nIndex matrix\n");
	for(int i = 0; i < N_users; i++) {
		std::cout << " [ ";
		for(int j = 0; j < N_movies; j++)
			std::cout << d_indices[i * N_movies + j] << " ";
		std::cout << "]\n";
	}

        thrust::device_vector<float> d_devnull(N_users);

	thrust::device_vector<float> d_squared_differences(N_users * N_movies);

	thrust::transform(d_matrix1.begin(), d_matrix1.end(), d_matrix2.begin(), d_squared_differences.begin(), PowerDifference());

	thrust::device_vector<float> d_norms(N_users);
	thrust::reduce_by_key(d_indices.begin(), d_indices.end(), d_squared_differences.begin(), d_devnull.begin(), d_norms.begin());
	
        thrust::device_vector<float> d_cuenta(N_users * N_movies);
        thrust::transform(d_matrix1.begin(), d_matrix1.end(), d_matrix2.begin(), d_cuenta.begin(), countIfNoZeros());

        thrust::device_vector<float> d_dividendo(N_users);
        thrust::reduce_by_key(d_indices.begin(), d_indices.end(), d_cuenta.begin(), d_devnull.begin(), d_dividendo.begin());



       thrust::device_vector<float> d_distancias_euclidianas(N_users);
       thrust::transform(d_norms.begin(), d_norms.end(), d_dividendo.begin(), d_distancias_euclidianas.begin(), thrust::divides<float>());

       printf("\n\nDistancia Euclidiana \n");
        for(int i = 0; i < N_users; i++) {
                //      std::cout << (d_norms[i]/d_dividendo[i]) << " ";
                std::cout << d_norms[i] << "/" << d_dividendo[i] << "=" << d_distancias_euclidianas[i] << " \n";
        }


       thrust::device_vector<int> user_index(N_users);
       thrust::sequence(user_index.begin(), user_index.end(), 0, 1);
       

       thrust::sort_by_key(user_index.begin(), user_index.end(), d_distancias_euclidianas.begin());

       std::cout << "La menor distancias es :" << d_distancias_euclidianas[1] << " del usuario " << user_index[1]<< " \n";
       


	return 0; 
}
