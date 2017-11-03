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
        if ( a == 0.0f || b == 0.0f) {
          return 0.0f;
        } else {
          return pow(a - b, 2); 
        }
   }
};


struct countIfNoZeros {
        __host__ __device__ float operator()(const float& a, const float& b) const {
        if ( a > 0.0f && b > 0.0f) {
          return 1.0f;
        } else {
          return 0.0f;
        }
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
  
	const int N		= 20;			// --- Number of vector elements
	const int Nvec	= 3;			// --- Number of vectors for each matrix

	// --- Random uniform integer distribution between 0 and 100
	thrust::default_random_engine rng;
	thrust::uniform_int_distribution<int> dist(0, 20);

	// --- Matrix allocation and initialization

	thrust::device_vector<float> d_matrix1(Nvec * N);
	thrust::device_vector<float> d_matrix2(Nvec * N);

        d_matrix1[0]  =  0;
        d_matrix1[1]  =  1;
        d_matrix1[2]  =  2;
        d_matrix1[3]  =  3;
        d_matrix1[4]  =  5;
        d_matrix1[5]  =  0;
        d_matrix1[6]  =  0;
        d_matrix1[7]  =  0;
        d_matrix1[8]  =  0;
        d_matrix1[9]  =  0;
        d_matrix1[10]  =  0;
        d_matrix1[11]  =  0;
        d_matrix1[12]  =  0;
        d_matrix1[13]  =  0;
        d_matrix1[14]  =  0;
        d_matrix1[15]  =  0;
        d_matrix1[16]  =  0;
        d_matrix1[17]  =  0;
        d_matrix1[18]  =  0;
        d_matrix1[19]  =  0;
        d_matrix1[20]  =  0;
        d_matrix1[21]  =  2;
        d_matrix1[22]  =  2;
        d_matrix1[23]  =  3;
        d_matrix1[24]  =  0;
        d_matrix1[25]  =  5;
        d_matrix1[26]  =  0;
        d_matrix1[27]  =  0;
        d_matrix1[28]  =  0;
        d_matrix1[29]  =  0;
        d_matrix1[30]  =  0;
        d_matrix1[31]  =  1;
        d_matrix1[32]  =  0;
        d_matrix1[33]  =  0;
        d_matrix1[34]  =  0;
        d_matrix1[35]  =  0;
        d_matrix1[36]  =  0;
        d_matrix1[37]  =  0;
        d_matrix1[38]  =  0;
        d_matrix1[39]  =  0;
        d_matrix1[40]  =  5;
        d_matrix1[41]  =  5;
        d_matrix1[42]  =  5;
        d_matrix1[43]  =  1;
        d_matrix1[44]  =  0;
        d_matrix1[45]  =  0;
        d_matrix1[46]  =  0;
        d_matrix1[47]  =  0;
        d_matrix1[48]  =  0;
        d_matrix1[49]  =  0;
        d_matrix1[50]  =  0;
        d_matrix1[51]  =  0;
        d_matrix1[52]  =  0;
        d_matrix1[53]  =  0;
        d_matrix1[54]  =  0;
        d_matrix1[55]  =  0;
        d_matrix1[56]  =  0;
        d_matrix1[57]  =  0;
        d_matrix1[58]  =  0;
        d_matrix1[59]  =  0;


        d_matrix2[0]  =  0;
        d_matrix2[1]  =  1;
        d_matrix2[2]  =  2;
        d_matrix2[3]  =  3;
        d_matrix2[4]  =  5;
        d_matrix2[5]  =  0;
        d_matrix2[6]  =  0;
        d_matrix2[7]  =  0;
        d_matrix2[8]  =  0;
        d_matrix2[9]  =  0;
        d_matrix2[10]  =  0;
        d_matrix2[11]  =  0;
        d_matrix2[12]  =  0;
        d_matrix2[13]  =  0;
        d_matrix2[14]  =  0;
        d_matrix2[15]  =  0;
        d_matrix2[16]  =  0;
        d_matrix2[17]  =  0;
        d_matrix2[18]  =  0;
        d_matrix2[19]  =  0;
        d_matrix2[20]  =  0;
        d_matrix2[21]  =  1;
        d_matrix2[22]  =  2;
        d_matrix2[23]  =  3;
        d_matrix2[24]  =  5;
        d_matrix2[25]  =  0;
        d_matrix2[26]  =  0;
        d_matrix2[27]  =  0;
        d_matrix2[28]  =  0;
        d_matrix2[29]  =  0;
        d_matrix2[30]  =  0;
        d_matrix2[31]  =  0;
        d_matrix2[32]  =  0;
        d_matrix2[33]  =  0;
        d_matrix2[34]  =  0;
        d_matrix2[35]  =  0;
        d_matrix2[36]  =  0;
        d_matrix2[37]  =  0;
        d_matrix2[38]  =  0;
        d_matrix2[39]  =  0;
        d_matrix2[40]  =  0;
        d_matrix2[41]  =  1;
        d_matrix2[42]  =  2;
        d_matrix2[43]  =  3;
        d_matrix2[44]  =  5;
        d_matrix2[45]  =  0;
        d_matrix2[46]  =  0;
        d_matrix2[47]  =  0;
        d_matrix2[48]  =  0;
        d_matrix2[49]  =  0;
        d_matrix2[50]  =  0;
        d_matrix2[51]  =  0;
        d_matrix2[52]  =  0;
        d_matrix2[53]  =  0;
        d_matrix2[54]  =  0;
        d_matrix2[55]  =  0;
        d_matrix2[56]  =  0;
        d_matrix2[57]  =  0;
        d_matrix2[58]  =  0;
        d_matrix2[59]  =  0;



        



	printf("\n\nFirst matrix\n");
	for(int i = 0; i < Nvec; i++) {
		std::cout << " [ ";
		for(int j = 0; j < N; j++)
			std::cout << d_matrix1[i * N + j] << " ";
		std::cout << "]\n";
	}

	printf("\n\nSecond matrix\n");
	for(int i = 0; i < Nvec; i++) {
		std::cout << " [ ";
		for(int j = 0; j < N; j++)
			std::cout << d_matrix2[i * N + j] << " ";
		std::cout << "]\n";
	}


        
       



	/****************************************************************************/
	/* CALCULATING THE EUCLIDEAN DISTANCES BETWEEN THE ROWS OF THE TWO MATRICES */
	/****************************************************************************/
	// --- Creating the indices for the reduction by key
	thrust::device_vector<int> d_sequence(Nvec);
	thrust::device_vector<int> d_indices(Nvec * N);
	thrust::device_vector<int> d_counts(Nvec, N);
	thrust::sequence(d_sequence.begin(), d_sequence.begin() + Nvec);
	expand(d_counts.begin(), d_counts.end(), d_sequence.begin(), d_indices.begin());

	printf("\n\nIndex matrix\n");
	for(int i = 0; i < Nvec; i++) {
		std::cout << " [ ";
		for(int j = 0; j < N; j++)
			std::cout << d_indices[i * N + j] << " ";
		std::cout << "]\n";
	}

        thrust::device_vector<float> d_devnull(Nvec);

	thrust::device_vector<float> d_squared_differences(Nvec * N);

	thrust::transform(d_matrix1.begin(), d_matrix1.end(), d_matrix2.begin(), d_squared_differences.begin(), PowerDifference());

	thrust::device_vector<float> d_norms(Nvec);
	thrust::reduce_by_key(d_indices.begin(), d_indices.end(), d_squared_differences.begin(), d_devnull.begin(), d_norms.begin());
	
        thrust::device_vector<float> d_cuenta(Nvec * N);
        thrust::transform(d_matrix1.begin(), d_matrix1.end(), d_matrix2.begin(), d_cuenta.begin(), countIfNoZeros());

        thrust::device_vector<float> d_dividendo(Nvec);
        thrust::reduce_by_key(d_indices.begin(), d_indices.end(), d_cuenta.begin(), d_devnull.begin(), d_dividendo.begin());



       thrust::device_vector<float> d_distancias_euclidianas(Nvec);
       thrust::transform(d_norms.begin(), d_norms.end(), d_dividendo.begin(), d_distancias_euclidianas.begin(), thrust::divides<float>());

       printf("\n\nDistancia Euclidiana \n");
        for(int i = 0; i < Nvec; i++) {
                //      std::cout << (d_norms[i]/d_dividendo[i]) << " ";
                std::cout << d_norms[i] << "/" << d_dividendo[i] << "=" << d_distancias_euclidianas[i] << " \n";
        }


       thrust::device_vector<int> user_index(Nvec);
       thrust::sequence(user_index.begin(), user_index.end(), 0, 1);
       

       thrust::sort_by_key(user_index.begin(), user_index.end(), d_distancias_euclidianas.begin());

       std::cout << "La menor distancias es :" << d_distancias_euclidianas[1] << " del usuario " << user_index[1]<< " \n";
       
       


	return 0; 
}
