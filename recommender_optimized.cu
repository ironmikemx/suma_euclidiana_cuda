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

#define MASK 99
#define INF 999

/*
 * Operator:  power_difference 
 * --------------------
 * computes the square power difference of two numbers using:
 *    pow(a - b, 2) only if both numbers are different to 0.
 *    This is a special power_difference as we only want to
 *    compute when both elements in our vector have a value.
 *    Meaning only when two users have seen the movie for
 *    our recommender.
 *
 *  a: first number to compute the power difference
 *  b: second number to compute the power difference
 *
 *  returns: 0 when one of the input values is 0
 *           power difference of the two values in any other case
 */
struct power_difference {
    __host__ __device__ float operator()(const int& a, const int& b) const {
        if ( a % 100 == 0 || b % 100 == 0) {
            return 0;
        } else {
            return powf(a % 100 - b % 100, 2);
        }
    }
};

/*
 * Operator:  weight_division
 * --------------------
 * computes the division of two numbers a and b using the fomula:
 *    (a + (0.00001 - 0.000001 * b)) / b
 *    This is a special divison used to rank matches. With these
 *    even if two division would be the same, we favor a lowe value
 *    by the weight of the dividend
 *    Example
 *       Normal divison         Weighted division
 *     a    b    result        a    b    result
 *    ---  ---  --------      ---  ---  --------
 *     1    2      0.5         1    2   0.5000040
 *     2    4      0.5         2    4   0.5000015
 *
 *    This way if the quotients are sorted, we favor values with larger
 *    number of matches (b).
 *
 *  a: dividend
 *  b: divided
 *
 *  returns: Weighted quotient of two numbers
 */
struct weight_division {
    __host__ __device__ float operator()(const float& a, const float& b) const {
        if(b == 0) {
            return INF;
        } else {
            return (a + (0.00001f - b * 0.000001f)) / b;
        }
    }
};


/*
 * Operator:  one_if_not_zeros
 * --------------------
 * this operator return 1 when both inputs are different to 0
 *
 *  a: number
 *  b: number
 *
 *  returns: 1 when the two input values are different to 0
 *           0 otherwise
 */
struct one_if_not_zeros {
    __host__ __device__ int operator()(const int& a, const int& b) const {
        if ( a > 0 && b > 0) {
            return 1;
        } else {
            return 0;
        }
    }
};

/*
 * Operator:  mask_if_zero
 * --------------------
 * this operator returns the first value received if the second
 *    number is not zero. Otherwise return 999999999
 *
 *  a: first number
 *  b: second number
 *
 *  returns: a when b > 0
 *           MASK otherwise
 */
struct mask_if_zero {
    __host__ __device__ int operator()(const int& a, const int& b) const {
        if ( b > 0) {
            return a;
        } else {
            return MASK;
        }
    }
};


/*
 * Operator:  is_less_than_mask
 * --------------------
 * Unary operator. Returns if the input value is less than mask
 *
 *  n: number
 *
 *  returns: TRUE when n < MASK
 *           FALSE otherwise
 */
struct is_less_than_mask : public thrust::unary_function<int, int> {
    __host__ __device__ bool operator()(int n) const {
        return n < MASK;
    }
};


/*
 * Operator:  not_in_common
 * --------------------
 * this operator returns the rating of a movie if the client
 *    has not seen it yet. Otherwise 0
 *
 *  a: user movie rating
 *  b: client movie rating
 *
 *  returns: a when b is 0
 *           0 otherwise
 */
struct not_in_common {
    __host__ __device__ int operator()(const int& a, const int& b) const {
        if ( b == 0) {
            return a;
        } else {
            return 0;
        }
    }
};


/*
 * Iterator:  make_matrix_index
 * --------------------
 * creates an iterator that is a one dimension representation of a two
 *    dimentional matrix. Where all rows have the same value. 
 *    Example: In a 4 x 3 the content will be:
 *    (1, 1, 1, 1
 *     2, 2, 2, 2
 *     3, 3, 3, 3)
 *
 *  first1: Beginning of fist range
 *  last1: End of first range
 *  fist2: Beginning of the second range
 *  output: where to store the output
 *
 *  returns: An iterator with an indexed row matrix 
 */
template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
   OutputIterator make_matrix_index(InputIterator1 first1, InputIterator1 last1,
                      InputIterator2 first2, OutputIterator output) {

    typedef typename thrust::iterator_difference<InputIterator1>::type difference_type;

    difference_type input_size = thrust::distance(first1, last1);
    difference_type output_size = thrust::reduce(first1, last1);

    // scan the counts to obtain output offsets for each input element
    thrust::device_vector<difference_type> output_offsets(input_size, 0);
    thrust::exclusive_scan(first1, last1, output_offsets.begin());
    // scatter the nonzero counts into their corresponding output positions
    thrust::device_vector<difference_type> output_indices(output_size, 0);
    thrust::scatter_if(thrust::counting_iterator<difference_type>(0), 
        thrust::counting_iterator<difference_type>(input_size), output_offsets.begin(), 
        first1, output_indices.begin());

    // compute max-scan over the output indices, filling in the holes
    thrust::inclusive_scan(output_indices.begin(), output_indices.end(), output_indices.begin(), 
        thrust::maximum<difference_type>());

    // gather input values according to index array (output = first2[output_indices])
    OutputIterator output_end = output;
    thrust::advance(output_end, output_size);
    thrust::gather(output_indices.begin(), output_indices.end(), first2, output);

    // return output + output_size
    thrust::advance(output, output_size);

    return output;
}

/*
 * Function:  print_matrix
 * --------------------
 * print a vector as a formated 2D matrix
 *
 *  matrix: vector of size x * y 
 *  x: Number of rows
 *  y: Numbe of columns
 *  label: Label to display above the matrix
 *
 */
template <class T>
void print_matrix (thrust::device_vector<T>& matrix, const int x, const int y, const char* label) {
    std::cout << "\n\n  " << label << "\n";
    std::cout << "  ----------------------\n";
    for(int i = 0; i < x; i++) {
        std::cout << "   u[" << i << "] ";
        for(int j = 0; j < y; j++) {
            std::cout << matrix[i * y + j] << " ";
	}
        std::cout << "\n";
    }
    std::cout << "\n";
}

/*
 * Function:  main 
 * --------------------
 * compute which user has the lowest euclidean distance for Client
 *
 *  N_users: Number of users to select from our initial data. Max 943
 *  N_movies: Number movies to select from our initial data. Max 1682
 *  Client: An user_id we want to find a closes match
 *
 */
int main(int argc, char** argv) {

   /*
    * Read the input parameters
    * --------------------
    */
    const int amount_of_users_in_dataset = atoi(argv[1]); //Users in our initial dataset
    const int amount_of_movies_in_dataset = atoi(argv[2]); //Movies in our initial dataset
    int client_id = atoi(argv[3]); //user_id of the person we want to find similar users for
    int verbose = atoi(argv[4]); //verbose 0 - Print only results, verbose 1 - print steps
    const int dataset_size = amount_of_users_in_dataset * amount_of_movies_in_dataset;

   /*
    * Dataset generation
    * --------------------
    * generate our initial dataset as a subset of MovieLense 100K dataset stored in data.cu 
    */
    thrust::device_vector<int> user_ratings_dataset(dataset_size);
    thrust::device_vector<int> client_ratings_dataset(dataset_size);
    thrust::device_vector<int> movie_ratings_dataset(dataset_size);

    load(user_ratings_dataset, amount_of_users_in_dataset, amount_of_movies_in_dataset);
    load(client_ratings_dataset, amount_of_users_in_dataset, amount_of_movies_in_dataset, client_id);
    thrust::copy(user_ratings_dataset.begin(), user_ratings_dataset.end(), movie_ratings_dataset.begin());

    // Show original ratings dataset
    if(verbose) {
        print_matrix (user_ratings_dataset, amount_of_users_in_dataset, 
           amount_of_movies_in_dataset, "user_ratings_dataset");
        print_matrix (client_ratings_dataset, amount_of_users_in_dataset,
           amount_of_movies_in_dataset, "client_ratings_dataset");
    }


    thrust::device_vector<int> masked_user_ratings(dataset_size);
    thrust::device_vector<int> masked_client_ratings(dataset_size);
    thrust::transform(user_ratings_dataset.begin(), user_ratings_dataset.end(),
           client_ratings_dataset.begin(), masked_user_ratings.begin(), mask_if_zero());
    thrust::transform(client_ratings_dataset.begin(), client_ratings_dataset.end(),
           client_ratings_dataset.begin(), masked_client_ratings.begin(), mask_if_zero());



     // Show masked ratings dataset
       if(verbose) {
           print_matrix(masked_user_ratings, amount_of_users_in_dataset,
              amount_of_movies_in_dataset, "masked_user_ratings_dataset");
           print_matrix(masked_client_ratings, amount_of_users_in_dataset,
              amount_of_movies_in_dataset, "masked_client_ratings_dataset");
       }


       int reduced_dataset_size = thrust::count_if(masked_user_ratings.begin(),
           masked_user_ratings.end(), is_less_than_mask());
       int N_users = amount_of_users_in_dataset;
       int N_movies = reduced_dataset_size / amount_of_users_in_dataset;



       thrust::device_vector<int> user_ratings_working_dataset(reduced_dataset_size);
       thrust::device_vector<int> client_ratings_working_dataset(reduced_dataset_size);

       thrust::copy_if(masked_user_ratings.begin(), masked_user_ratings.end(), 
           user_ratings_working_dataset.begin(), is_less_than_mask());
       thrust::copy_if(masked_client_ratings.begin(), masked_client_ratings.end(), 
           client_ratings_working_dataset.begin(), is_less_than_mask());

       // Show reduced ratings dataset
       if(verbose) {
           print_matrix(user_ratings_working_dataset, N_users, N_movies, 
               "reduced_user_ratings_dataset");
           print_matrix(client_ratings_working_dataset, N_users, N_movies,
               "reduced_client_ratings_dataset");
       }
    

   /*
    * Create index matrix for reduction
    * --------------------
    * create a vector that will help us to reduce by rows in next step
    * E.g. In a 3 x 2
    * (1 1 1
    *  2 2 2)
    */
    thrust::device_vector<int> seq(N_users);
    thrust::device_vector<int> reduce_by_key_index(N_users * N_movies);
    thrust::device_vector<int> reps(N_users, N_movies);
    thrust::sequence(seq.begin(), seq.begin() + N_users);
    make_matrix_index(reps.begin(), reps.end(), seq.begin(), reduce_by_key_index.begin());


   /*
    * Compute Euclidean distance
    * --------------------
    */
    thrust::device_vector<float> dev_null(N_users);
    thrust::device_vector<float> squared_differences(N_users * N_movies);
    thrust::device_vector<float> squared_differences_sum(N_users);
    thrust::device_vector<int> common_movies(N_users * N_movies);
    thrust::device_vector<float> common_movies_count(N_users);
    thrust::device_vector<float> euclidean_distance(N_users);

    thrust::transform(user_ratings_working_dataset.begin(), user_ratings_working_dataset.end(), 
       client_ratings_working_dataset.begin(), squared_differences.begin(), power_difference());
    // Show squared differences dataset
    if(verbose) {
        print_matrix (squared_differences, N_users, N_movies, "squared_differences");
    }

    thrust::reduce_by_key(reduce_by_key_index.begin(), reduce_by_key_index.end(), squared_differences.begin(), 
       dev_null.begin(), squared_differences_sum.begin());
    thrust::transform(user_ratings_working_dataset.begin(), user_ratings_working_dataset.end(), 
        client_ratings_working_dataset.begin(), common_movies.begin(), one_if_not_zeros());
    if(verbose) {
        print_matrix (user_ratings_working_dataset, N_users, N_movies, "users");
        print_matrix (client_ratings_working_dataset, N_users, N_movies, "client");
        print_matrix (common_movies, N_users, N_movies, "common_movies");
    }
    thrust::reduce_by_key(reduce_by_key_index.begin(), reduce_by_key_index.end(), common_movies.begin(), 
        dev_null.begin(), common_movies_count.begin());
    thrust::transform(squared_differences_sum.begin(), squared_differences_sum.end(), common_movies_count.begin(), 
        euclidean_distance.begin(), weight_division());

    // Show Euclidean distance
    if(verbose) {
        std::cout << "\n\n  euclidean_distances \n";
        std::cout << "  ----------------------\n"; 
        for(int i = 0; i < N_users; i++) {
            std::cout << "   u[" << i << "] " << squared_differences_sum[i] << " / " << common_movies_count[i] << "=" 
                << euclidean_distance[i] << " \n";
        }
    }


   /*
    * Find lowest distance in data set
    * --------------------
    */
    thrust::device_vector<int> user_index(N_users);
    thrust::sequence(user_index.begin(), user_index.end(), 0, 1);
    thrust::sort_by_key(euclidean_distance.begin(), euclidean_distance.end(), 
        user_index.begin());
    // Show Euclidean distance
    if(verbose) {
        std::cout << "\n\n  sorted euclidean_distances \n";
        std::cout << "  ----------------------\n";
        for(int i = 0; i < N_users; i++) {
            std::cout << "   u[" << user_index[i] << "] " << euclidean_distance[i]
                << " \n";
        }
    }
    int answer = 0;
    if (client_id == user_index[answer]) {
        answer++;
    }
    std::cout << "Lowest Euclidean Distance: " << euclidean_distance[answer] 
        << " from user: " << user_index[answer]<< " \n";

   /*
    * Recommend a movie
    * --------------------
    */
    int offset=user_index[answer] * amount_of_movies_in_dataset;
    int client_offset = client_id * amount_of_movies_in_dataset;
    thrust::device_vector<char> possible_movies(amount_of_movies_in_dataset);

    thrust::transform(movie_ratings_dataset.begin() + offset, movie_ratings_dataset.begin()
        + offset + amount_of_movies_in_dataset, movie_ratings_dataset.begin() + client_offset, possible_movies.begin(), not_in_common());

    // Show movies in common
    if(verbose) {
        std::cout << "\n\n  possible_movie_ratings \n";
        std::cout << "  ----------------------\n";
        std::cout << "   u["  << user_index[answer] << "] \n";
        std::cout << "   movie    rating \n";
        for(int i = 0; i < amount_of_movies_in_dataset ; i++) {
           int movie = (int)possible_movies[i];
           std::cout << "     " << i<< "         " << movie << "\n";
        }
        std::cout << "\n";
    }

    thrust::device_vector<int> movie_index(amount_of_movies_in_dataset);
    thrust::sequence(movie_index.begin(), movie_index.end(), 0, 1);
    thrust::sort_by_key(possible_movies.begin(), possible_movies.end(),
        movie_index.begin());
    std::cout << "Recommended Movies: ";
    std::cout << movie_index[amount_of_movies_in_dataset-1] << ", ";
    std::cout << movie_index[amount_of_movies_in_dataset-2] << ", ";
    std::cout << movie_index[amount_of_movies_in_dataset-3] << ", ";
    std::cout << " \n";
    return 0;
}
