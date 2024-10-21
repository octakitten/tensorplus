#pragma once
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "defines.h"
#include <stdbool.h>

typedef struct {
    unsigned int* size;
    short* data;
} Tensor;

Tensor* create_tensor(unsigned int size);
void destroy_tensor(Tensor* tensor);
Tensor* create_device_tensor(unsigned int size);
void destroy_device_tensor(Tensor* tensor);
void get_tensor_size_wrapper(Tensor* tensor, unsigned int size);
void zeros_tensor_wrapper( Tensor* tensor);
void ones_tensor_wrapper( Tensor* tensor);
void print_tensor_wrapper( Tensor* tensor);
void fill_tensor_wrapper( Tensor* tensor, short value);
void set_tensor_wrapper( Tensor* tensor, unsigned int index, short value);
void get_tensor_value_wrapper( Tensor* tensor, unsigned int index, short value);
void add_tensor_wrapper( Tensor* tensor,  Tensor* other,  Tensor* result);
void sub_tensor_wrapper( Tensor* tensor,  Tensor* other,  Tensor* result);
void mul_tensor_wrapper( Tensor* tensor,  Tensor* other,  Tensor* result);
void div_tensor_wrapper( Tensor* tensor,  Tensor* other,  Tensor* result);
void add_scalar_tensor_wrapper( Tensor* tensor, short scalar,  Tensor* result);
void sub_scalar_tensor_wrapper( Tensor* tensor, short scalar,  Tensor* result);
void mul_scalar_tensor_wrapper( Tensor* tensor, short scalar,  Tensor* result);
void div_scalar_tensor_wrapper( Tensor* tensor, short scalar,  Tensor* result);
void transpose_tensor_wrapper( Tensor* tensor,  Tensor* result);
void sum_tensor_wrapper( Tensor* tensor,  Tensor* result);
void max_tensor_wrapper( Tensor* tensor,  Tensor* result);
void min_tensor_wrapper( Tensor* tensor,  Tensor* result);
void gradient_tensor_wrapper( Tensor* tensor,  Tensor* result);
void gate_tensor_wrapper( Tensor* tensor,  Tensor* booleans,  Tensor* result);
void vector_sort_tensor_wrapper( Tensor* tensor,  Tensor* vectors,  Tensor* result);
void vector_add_wrapper( Tensor* tensor, Tensor* other, Tensor* vectors, Tensor* result);
void vector_sub_wrapper( Tensor* tensor, Tensor* other, Tensor* vectors, Tensor* result);
void vector_mul_wrapper( Tensor* tensor, Tensor* other, Tensor* vectors, Tensor* result);
void vector_div_wrapper( Tensor* tensor, Tensor* other, Tensor* vectors, Tensor* result);
void vector_gate_wrapper( Tensor* tensor, Tensor* booleans, Tensor* vectors, Tensor* result);
bool check_size_3( Tensor* tensor,  Tensor* other,  Tensor* result);
void vector_resize_tensor_wrapper(Tensor* tensor, Tensor* vectors, Tensor* result);
void tensor_enlarge_wrapper(Tensor* tensor, unsigned int scale_factor, Tensor* result);
void lesser_tensor_wrapper(Tensor* tensor, Tensor* other, Tensor* result);
void greater_tensor_wrapper(Tensor* tensor, Tensor* other, Tensor* result);
void lesser_equals_tensor_wrapper(Tensor* tensor, Tensor* other, Tensor* result);
void greater_equals_tensor_wrapper(Tensor* tensor, Tensor* other, Tensor* result);
void equals_tensor_wrapper(Tensor* tensor, Tensor* other, Tensor* result);
void copy_tensor(Tensor* tensor, Tensor* result);
void clone_tensor(Tensor* tensor, Tensor* result);
void negate_tensor_wrapper(Tensor* tensor, Tensor* result);
