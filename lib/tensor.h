#pragma once
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "defines.h"

typedef struct {
    int* size;
    short* data;
} Tensor;

Tensor* create_tensor(int size);
void destroy_tensor(Tensor* tensor);
Tensor* create_device_tensor(int size);
void destroy_device_tensor(Tensor* tensor);
void zeros_tensor_wrapper( Tensor* tensor);
void ones_tensor_wrapper( Tensor* tensor);
void print_tensor_wrapper( Tensor* tensor);
void fill_tensor_wrapper( Tensor* tensor, int value);
void add_tensor_wrapper( Tensor* tensor,  Tensor* other,  Tensor* result);
void sub_tensor_wrapper( Tensor* tensor,  Tensor* other,  Tensor* result);
void mul_tensor_wrapper( Tensor* tensor,  Tensor* other,  Tensor* result);
void div_tensor_wrapper( Tensor* tensor,  Tensor* other,  Tensor* result);
void add_scalar_tensor_wrapper( Tensor* tensor, int scalar,  Tensor* result);
void sub_scalar_tensor_wrapper( Tensor* tensor, int scalar,  Tensor* result);
void mul_scalar_tensor_wrapper( Tensor* tensor, int scalar,  Tensor* result);
void div_scalar_tensor_wrapper( Tensor* tensor, int scalar,  Tensor* result);
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
