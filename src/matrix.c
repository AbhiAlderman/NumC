#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid. Note that the matrix is in row-major order.
 */
double get(matrix *mat, int row, int col) {
    double return_val = 0.0;
    int step_size = mat->cols;
    return_val = *(mat->data + (row * step_size + col));
    return return_val;
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid. Note that the matrix is in row-major order.
 */
void set(matrix *mat, int row, int col, double val) {
    int step_size = mat->cols;
    *(mat->data + (row * step_size + col)) = val;
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    // Task 1.2 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    // 3. Allocate space for the matrix data, initializing all entries to be 0. Return -2 if allocating memory failed.
    // 4. Set the number of rows and columns in the matrix struct according to the arguments provided.
    // 5. Set the `parent` field to NULL, since this matrix was not created from a slice.
    // 6. Set the `ref_cnt` field to 1.
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    // 8. Return 0 upon success.
    if (rows <= 0 || cols <= 0) {
        return -1;
    }
    matrix* new_matrix_struct = (matrix*) malloc (sizeof(matrix));
    if (new_matrix_struct == NULL) {
        return -2;
    }
    new_matrix_struct->data = (double*) calloc (rows * cols, sizeof(double));
    if (new_matrix_struct->data == NULL) {
        return -2;
    }
    new_matrix_struct->rows = rows;
    new_matrix_struct->cols = cols;
    new_matrix_struct->parent = NULL;
    new_matrix_struct->ref_cnt = 1;
    *mat = new_matrix_struct;

    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or that you free `mat->parent->data` if `mat` is the last existing slice of its parent matrix and its parent
 * matrix has no other references (including itself).
 */
void deallocate_matrix(matrix *mat) {
    // Task 1.3 TODO
    // HINTS: Follow these steps.
    // 1. If the matrix pointer `mat` is NULL, return.
    // 2. If `mat` has no parent: decrement its `ref_cnt` field by 1. If the `ref_cnt` field becomes 0, then free `mat` and its `data` field.
    // 3. Otherwise, recursively call `deallocate_matrix` on `mat`'s parent, then free `mat`.
    if (mat == NULL) {
        return;
    }
    if (mat->parent == NULL) {
        mat->ref_cnt = mat->ref_cnt - 1;
        if (mat->ref_cnt == 0) {
            free(mat->data);
            free(mat);
        }
    }
    return deallocate_matrix(mat->parent);
    free(mat);
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`
 * and the reference counter for `from` should be incremented. Lastly, do not forget to set the
 * matrix's row and column values as well.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 * NOTE: Here we're allocating a matrix struct that refers to already allocated data, so
 * there is no need to allocate space for matrix data.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    // Task 1.4 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    // 3. Set the `data` field of the new struct to be the `data` field of the `from` struct plus `offset`.
    // 4. Set the number of rows and columns in the new struct according to the arguments provided.
    // 5. Set the `parent` field of the new struct to the `from` struct pointer.
    // 6. Increment the `ref_cnt` field of the `from` struct by 1.
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    // 8. Return 0 upon success.
    int return_val = 0;
    if (rows <= 0 || cols <= 0) {
        return_val = -1;
        return return_val;
    }
    matrix* new_struct = (matrix*) malloc (sizeof(matrix));
    if (new_struct == NULL) {
        return_val = -2;
        return return_val;
    }
    new_struct->data = from->data + offset;
    new_struct->rows = rows;
    new_struct->cols = cols;
    new_struct->parent = from;
    from->ref_cnt += 1;
    *mat = new_struct;
    return return_val;
}

/*
 * Sets all entries in mat to val. Note that the matrix is in row-major order.
 */
void fill_matrix(matrix *mat, double val) {
    #pragma omp parallel for 
    for (int i = 0; i < (mat->rows * mat->cols) / 4 * 4; i+=4) {
        __m256d four_doubles = _mm256_set_pd(val, val, val, val);
         _mm256_storeu_pd((mat->data + i), four_doubles);
    }

    #pragma omp parallel for 
    for (int i = (mat->rows * mat->cols) / 4 * 4; i < mat->rows * mat->cols; ++i) {
         *(mat->data + i) = val;
    }
    /*
    for (int i = 0; i < mat->rows; ++i) {
        for (int j = 0; j < mat->cols; ++j) {
            set(mat, i, j, val);
        }
    }
    */
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int abs_matrix(matrix *result, matrix *mat) {
    __m256d set_neg_ones = _mm256_set_pd(-1.0, -1.0, -1.0, -1.0);
    #pragma omp parallel for shared(set_neg_ones) 
    for (int i = 0; i < (mat->rows * mat->cols) / 4 * 4; i += 4) {
        __m256d mat_elems = _mm256_loadu_pd((mat->data + i));
        __m256d neg_elems = _mm256_mul_pd(mat_elems, set_neg_ones);
        __m256d max_elems = _mm256_max_pd(mat_elems, neg_elems);
        _mm256_storeu_pd((result->data + i), max_elems);
    }

    #pragma omp parallel for 
    for(int i = (mat->rows * mat->cols) / 4 * 4; i < mat->rows * mat->cols; ++i) {
        double mat_val = *(result->data + i);
        if (mat_val < mat_val * -1.0) {
            *(result->data + i) = mat_val * -1.0;
        }
        else {
            *(result->data + i) = mat_val;
        }
    }
    /*
    //fmax(4, 5);
    double abs_curr_value = 0.0;
    for (int i = 0; i < mat->rows; ++i) {
        for (int j = 0; j < mat->cols; ++j) {
            double curr_value = get(mat, i, j);
            if (curr_value < 0) {
                abs_curr_value = curr_value * (-1.00);
            } 
            else {
                abs_curr_value = curr_value;
            }
            set(result, i, j, abs_curr_value);
        }
    }
    */
    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int neg_matrix(matrix *result, matrix *mat) {
    return 0;
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    #pragma omp parallel for 
    for (int i = 0; i < (mat1->rows * mat1->cols) / 4 * 4; i += 4) {
         __m256d mat1_elems = _mm256_loadu_pd((mat1->data + i));
        __m256d mat2_elems = _mm256_loadu_pd((mat2->data + i));
        __m256d add_elems = _mm256_add_pd(mat1_elems, mat2_elems);
        _mm256_storeu_pd(result->data + i, add_elems);
    }

    #pragma omp parallel for 
    for(int i = (mat1->rows * mat1->cols) / 4 * 4; i < mat1->rows * mat1->cols; ++i) {
        double mat1_value = *(mat1->data + i);
        double mat2_value = *(mat2->data + i);
        double result_val = mat1_value + mat2_value;
        *(result->data + i) = result_val;
    }
   /*
    #pragma omp parallel for
    for (int i = 0; i < mat1->rows; ++i) {
        for (int j = 0; j < mat1->cols; ++j) {
            double mat1_value = get(mat1, i, j);
            double mat2_value = get(mat2, i, j);
            double result_val = mat1_value + mat2_value;
            set(result, i, j, result_val);
        }
    }
    */
    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 * You may assume `mat1`'s number of columns is equal to `mat2`'s number of rows.
 * Note that the matrix is in row-major order.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    if ((mat1->rows * mat2->cols) <= 10000 || (mat2->rows * mat2->cols) <= 10000 ) {
       double running_total = 0.0;
        for (int mat2_cols = 0; mat2_cols < mat2->cols; ++mat2_cols) {
            for (int i = 0; i < mat1->rows; ++i) {
                for (int j = 0; j < mat1->cols; ++j) {
                    double mat1_curr_val = get(mat1, i, j);  
                    double mat2_curr_val = get(mat2, j, mat2_cols);
                    double mul_elems = mat1_curr_val * mat2_curr_val;
                    running_total += mul_elems;
                }
                set(result, i, mat2_cols, running_total);
                running_total = 0.0;
            }
        }
    }

    else {
        double* temp_array = (double*) malloc (sizeof(double*) * mat2->cols * mat2->rows);


        #pragma omp parallel for
        for(int i = 0; i < mat2->cols * mat2->rows; ++i) {
            int row = i / mat2->cols;
            int col = i % mat2->cols;
            *(temp_array + (col * mat2->rows) + row) = get(mat2, row, col); 
        }

        double* transpose;

        #pragma omp parallel for private(transpose) collapse(2)
        for (int i = 0; i < mat1->rows; i++) {
            for (int j = 0; j < mat2->cols; j++) {
                transpose = temp_array;
                double total = 0;
                int tracker = 0;
                int elems_to_mul = mat2->rows;
                double t_array[4];
                __m256d mul_elems = _mm256_set_pd (0.0, 0.0, 0.0, 0.0);
                for (int k = 0; k < mat2->rows; k += 28) {
                    if (elems_to_mul > 27) {
                        __m256d mat1_elems = _mm256_loadu_pd((mat1->data + (i * mat1->cols) + k));
                        __m256d mat2_elems = _mm256_loadu_pd((transpose + (j * mat2->rows) + k));
                        mul_elems = _mm256_fmadd_pd(mat1_elems, mat2_elems, mul_elems);
                        __m256d mat1_next_4 = _mm256_loadu_pd((mat1->data + (i * mat1->cols) + k + 4));
                        __m256d mat2_next_4 = _mm256_loadu_pd((transpose + (j * mat2->rows) + k + 4));
                        mul_elems = _mm256_fmadd_pd(mat1_next_4, mat2_next_4, mul_elems);
                        __m256d mat1_next_8 = _mm256_loadu_pd((mat1->data + (i * mat1->cols) + k + 8));
                        __m256d mat2_next_8 = _mm256_loadu_pd((transpose + (j * mat2->rows) + k + 8));
                        mul_elems = _mm256_fmadd_pd(mat1_next_8, mat2_next_8, mul_elems);
                        __m256d mat1_next_12 = _mm256_loadu_pd((mat1->data + (i * mat1->cols) + k + 12));
                        __m256d mat2_next_12 = _mm256_loadu_pd((transpose + (j * mat2->rows) + k + 12));
                        mul_elems = _mm256_fmadd_pd(mat1_next_12, mat2_next_12, mul_elems);
                        __m256d mat1_next_16 = _mm256_loadu_pd((mat1->data + (i * mat1->cols) + k + 16));
                        __m256d mat2_next_16 = _mm256_loadu_pd((transpose + (j * mat2->rows) + k + 16));
                        mul_elems = _mm256_fmadd_pd(mat1_next_16, mat2_next_16, mul_elems);
                        __m256d mat1_next_20 = _mm256_loadu_pd((mat1->data + (i * mat1->cols) + k + 20));
                        __m256d mat2_next_20 = _mm256_loadu_pd((transpose + (j * mat2->rows) + k + 20));
                        mul_elems = _mm256_fmadd_pd(mat1_next_20, mat2_next_20, mul_elems);
                        __m256d mat1_next_24 = _mm256_loadu_pd((mat1->data + (i * mat1->cols) + k + 24));
                        __m256d mat2_next_24 = _mm256_loadu_pd((transpose + (j * mat2->rows) + k + 24));
                        mul_elems = _mm256_fmadd_pd(mat1_next_24, mat2_next_24, mul_elems);
                        /*
                        __m256d mat1_next_28 = _mm256_loadu_pd((mat1->data + (i * mat1->cols) + k + 28));
                        __m256d mat2_next_28 = _mm256_loadu_pd((temp_array + (j * mat2->rows) + k + 28));
                        mul_elems = _mm256_fmadd_pd(mat1_next_28, mat2_next_28, mul_elems);
                        */
                        elems_to_mul = elems_to_mul - 28;
                    }
                    tracker = k;
                }

                for(int l = tracker; l < mat2->rows; l += 8) {
                    if (elems_to_mul > 7) {
                        __m256d mat1_elems = _mm256_loadu_pd((mat1->data + (i * mat1->cols) + l));
                        __m256d mat2_elems = _mm256_loadu_pd((transpose + (j * mat2->rows) + l));
                        mul_elems = _mm256_fmadd_pd(mat1_elems, mat2_elems, mul_elems);
                        __m256d mat1_next_4 = _mm256_loadu_pd((mat1->data + (i * mat1->cols) + l + 4));
                        __m256d mat2_next_4 = _mm256_loadu_pd((transpose + (j * mat2->rows) + l + 4));
                        mul_elems = _mm256_fmadd_pd(mat1_next_4, mat2_next_4, mul_elems);
                        elems_to_mul = elems_to_mul - 8;
                    }
                    tracker = l;
                }

                while (elems_to_mul > 0) {
                    double mat1_val = *(mat1->data + (i * mat1->cols) + tracker); //get(mat1, i, k);
                    double mat2_val = *(transpose + (j * mat2->rows) + tracker); //get(mat2, k, j);
                    total += mat1_val * mat2_val;
                    elems_to_mul--;
                    tracker++;
                }

                _mm256_storeu_pd(t_array, mul_elems);
                total += t_array[0] + t_array[1] + t_array[2] + t_array[3];
                set(result, i, j, total);
            }
        }

    }
        /*
        double* temp_array = (double*) malloc (sizeof(double*) * mat2->cols * mat2->rows);

        #pragma omp parallel for
        for(int i = 0; i < mat2->cols * mat2->rows; ++i) {
            int row = i / mat2->cols;
            int col = i % mat2->cols;
            *(temp_array + (col * mat2->rows) + row) = get(mat2, row, col); 
        }
        */ 

       /*
        double* transpose;

        #pragma omp parallel for private(transpose) //collapse(2)
        for (int i = 0; i < mat1->rows; i++) {
            transpose = temp_array;
            for (int j = 0; j < mat2->cols; j++) {
                double total = 0;
                int tracker = 0;
                int elems_to_mul = mat2->rows;
                double t_array[4];
                __m256d mul_elems = _mm256_set_pd (0.0, 0.0, 0.0, 0.0);
                for (int k = 0; k < mat2->rows; k += 4) {
                    if (elems_to_mul > 3) {
                        //__m256d mat1_elems = _mm256_loadu_pd((mat1->data + (i * mat1->cols) + k));
                        //__m256d mat2_elems = _mm256_loadu_pd((temp_array + (j * mat2->rows) + k));
                        mul_elems = _mm256_fmadd_pd(_mm256_loadu_pd((mat1->data + (i * mat1->cols) + k)), 
                        _mm256_loadu_pd((transpose + (j * mat2->rows) + k)), mul_elems);
                        elems_to_mul = elems_to_mul - 4;
                    }
                    tracker = k;
                }
                while (elems_to_mul > 0) {
                    double mat1_val = *(mat1->data + (i * mat1->cols) + tracker); //get(mat1, i, k);
                    double mat2_val = *(transpose + (j * mat2->rows) + tracker); //get(mat2, k, j);
                    total += mat1_val * mat2_val;
                    elems_to_mul--;
                    tracker++;
                }
                _mm256_storeu_pd(t_array, mul_elems);
                total += t_array[0] + t_array[1] + t_array[2] + t_array[3];
                set(result, i, j, total);
            }
        }
    }
    */
    //free(temp_array);
    return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 * You may assume `mat` is a square matrix and `pow` is a non-negative integer.
 * Note that the matrix is in row-major order.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    /*
    matrix* copied_result = NULL;
    allocate_matrix(&copied_result, result->rows, result->cols);
    memcpy(copied_result, mat, sizeof(matrix));
    double* data = (double*) malloc (result->rows * result->cols * sizeof(double));
    if (pow == 0) {
        fill_matrix(result, 0);
        int col_count = 0;
        for (int i = 0; i < result->rows; ++i) {
            set(result, i, col_count, 1);
            col_count++;
        }
    }
    else if (pow == 1) {
        memcpy(result->data, mat->data, mat->rows * mat->cols * sizeof(double));
    }
    else {
        for(int i = 1; i < pow; ++i) {
            mul_matrix(result, mat, copied_result);
            memcpy(copied_result, result, sizeof(matrix));      //Something wrong with this as we
            memcpy(data, result->data, result->rows * result->cols * sizeof(double));
            copied_result->data = data;

        }
    }
    */
    
    if (pow == 0) {
        fill_matrix(result, 0);
        #pragma omp parallel for
        for(int i = 0; i < mat->rows; ++i) {
            *(result->data + (i * mat->cols) + i) = 1.0;
        }
        return 0;
    }
    
    if (pow == 1) {
        memcpy(result->data, mat->data, mat->rows * mat->cols * sizeof(double));
        return 0;
    }
    double* data_x = (double*) malloc (result->rows * result->cols * sizeof(double));
    double* data_y = (double*) malloc (result->rows * result->cols * sizeof(double));

    matrix* x = NULL;
    allocate_matrix(&x, result->rows, result->cols);
    memcpy(x, mat, sizeof(matrix));

    matrix* y = NULL;
    allocate_matrix(&y, result->rows, result->cols);

    matrix* temp = NULL;
    allocate_matrix(&temp, result->rows, result->cols);

    fill_matrix(y, 0);

    #pragma omp parallel for
    for(int i = 0; i < mat->rows; ++i) {
        *(y->data + (i * mat->cols) + i) = 1.0;
    }

    while(pow > 1) {
        if (pow % 2 == 0) {
            mul_matrix(temp, x, x);
            memcpy(data_x, temp->data, result->rows * result->cols * sizeof(double));
            x->data = data_x;
            pow = pow / 2;
        }
        else {
            mul_matrix(temp, x, y);
            memcpy(data_y, temp->data, result->rows * result->cols * sizeof(double));
            y->data = data_y;
            mul_matrix(temp, x, x);
            memcpy(data_x, temp->data, result->rows * result->cols * sizeof(double));
            x->data = data_x;
            pow = (pow - 1) / 2;
        }
    }

    mul_matrix(temp, x, y);
    memcpy(data_y, temp->data, result->rows * result->cols * sizeof(double));
    result->data = data_y;
    //free(data_x);
    //free(data_y);
    return 0;
}
