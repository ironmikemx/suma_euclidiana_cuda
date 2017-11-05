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

struct PowerDifference {
        __host__ __device__ float operator()(const int& a, const int& b) const {
        if ( a%100 == 0 || b%100 == 0) {
          return 0;
        } else {
          return powf(a%100 - b%100, 2);
        }
   }
};

struct division_especial {
        __host__ __device__ float operator()(const int& a, const int& b) const {
       return (a+(0.00001f-b*0.000001f))/b;
   }
};


struct countIfNoZeros {
        __host__ __device__ int operator()(const int& a, const int& b) const {
        if ( a%100 > 0 && b%100 > 0) {
          return 1;
        } else {
          return 0;
        }
   }
};

struct mark {
        __host__ __device__ int operator()(const int& a, const int& b) const {
        if ( b%100 > 0) {
          return a;
        } else {
          return 999999999;
        }
   }
};

// note: functor inherits from unary_function
struct not_99 : public thrust::unary_function<int,int>
{
  __host__ __device__
  bool operator()(int x) const
  {
    return x < 999999999;
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

        int user_id = atoi(argv[3]);

	// --- Matrix allocation and initialization

//	thrust::device_vector<int> d_matrixA(N_users_orig * N_movies_orig);
//	thrust::device_vector<int> d_matrixB(N_users_orig * N_movies_orig);

        thrust::device_vector<int> d_matrixA(N_users_orig * N_movies_orig);
        load(d_matrixA, N_users_orig,N_movies_orig);

        thrust::device_vector<int> d_matrixB(N_users_orig * N_movies_orig);
        load(d_matrixB, N_users_orig,N_movies_orig,user_id);

	printf("\n\nmatrixA\n");
	for(int i = 0; i < N_users_orig; i++) {
		std::cout << " [ ";
		for(int j = 0; j < N_movies_orig; j++)
			std::cout << d_matrixA[i * N_movies_orig + j]%100 << " ";
		std::cout << "]\n";
	}


 
      // Reducing matrixes

      thrust::device_vector<int> temp1(N_movies_orig*N_users_orig);
      thrust::device_vector<int> temp2(N_movies_orig*N_users_orig);
      //thrust::device_vector<int> temp1sorted(N_movies_orig*N_users_orig);
      //thrust::device_vector<int> temp2sorted(N_movies_orig*N_users_orig);
      thrust::transform(d_matrixA.begin(), d_matrixA.end(), d_matrixB.begin(), temp1.begin(), mark());
      thrust::transform(d_matrixB.begin(), d_matrixB.end(), d_matrixB.begin(), temp2.begin(), mark());


      int N = thrust::count_if( temp1.begin(), temp1.end(), not_99());
      int N_users = N_users_orig;
      int N_movies = N / N_users;


      std::cout << "MOVIES ="<< N << "]\n";

      thrust::device_vector<int> d_matrix1(N);
      thrust::device_vector<int> d_matrix2(N);

      thrust::copy_if(temp1.begin(), temp1.end(), d_matrix1.begin(), not_99());
      thrust::copy_if(temp2.begin(), temp2.end(), d_matrix2.begin(), not_99());


        printf("\n\ntemp1\n");
        for(int i = 0; i < N_users_orig; i++) {
                std::cout << " [ ";
                for(int j = 0; j < N_movies_orig; j++)
                        std::cout << temp1[i * N_movies_orig + j]%100 << " ";
                std::cout << "]\n";
        }

        printf("\n\ntemp2\n");
        for(int i = 0; i < N_users_orig; i++) {
                std::cout << " [ ";
                for(int j = 0; j < N_movies_orig; j++)
                        std::cout << temp2[i * N_movies_orig + j]%100 << " ";
                std::cout << "]\n";
        }



printf("\n\nd_matrix1\n");
        for(int i = 0; i < N_users; i++) {
                std::cout << " [ ";
                for(int j = 0; j < N_movies; j++)
                        std::cout << d_matrix1[i * N_movies + j]%100 << " ";
                std::cout << "]\n";
        }

        printf("\n\nd_matrix2\n");
        for(int i = 0; i < N_users; i++) {
                std::cout << " [ ";
                for(int j = 0; j < N_movies; j++)
                        std::cout << d_matrix2[i * N_movies + j]%100 << " ";
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

        thrust::device_vector<int> d_cuenta(N_users * N_movies);
        thrust::transform(d_matrix1.begin(), d_matrix1.end(), d_matrix2.begin(), d_cuenta.begin(), countIfNoZeros());

        thrust::device_vector<float> d_dividendo(N_users);
        thrust::reduce_by_key(d_indices.begin(), d_indices.end(), d_cuenta.begin(), d_devnull.begin(), d_dividendo.begin());



       thrust::device_vector<float> d_distancias_euclidianas(N_users);
       //thrust::transform(d_norms.begin(), d_norms.end(), d_dividendo.begin(), d_distancias_euclidianas.begin(), thrust::divides<float>());

       thrust::transform(d_norms.begin(), d_norms.end(), d_dividendo.begin(), d_distancias_euclidianas.begin(), division_especial());

       printf("\n\nDistancia Euclidiana \n");
        for(int i = 0; i < N_users; i++) {
                //      std::cout << (d_norms[i]/d_dividendo[i]) << " ";
                std::cout << "usuario: " << i << " " << d_norms[i] << "/" << d_dividendo[i] << "=" << d_distancias_euclidianas[i] << " \n";
        }


       thrust::device_vector<int> user_index(N_users);
       thrust::sequence(user_index.begin(), user_index.end(), 0, 1);

       thrust::sort_by_key(d_distancias_euclidianas.begin(), d_distancias_euclidianas.end(), user_index.begin());


       printf("\n\nDistancias Ordenadas \n");
       for(int i = 0; i < N_users; i++) {
         std::cout << "usuario: " << user_index[i] <<  "=" << d_distancias_euclidianas[i] << " \n";
       }


       int answer = 0;
       if (user_id == user_index[answer]) {
         answer++;
       }
       std::cout << "La menor distancias es :" << d_distancias_euclidianas[answer] << " del usuario " << user_index[answer]<< " \n";

       
       


	return 0; 
}
