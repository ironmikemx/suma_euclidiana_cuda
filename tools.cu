#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/gather.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <stdio.h>
#include "tools.cuh"

using namespace thrust::placeholders;

#define MASK 99
#define INF 9999;


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
        return (a + (0.00001f - b * 0.000001f)) / b;
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
