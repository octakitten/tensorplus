#pragma once
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "defines.h"

struct Tensor {
    int size;
    short* data;
};


void print_tensor_wrapper(struct Tensor* tensor);
void fill_tensor_wrapper(struct Tensor* tensor, int value);
void add_tensor_wrapper(struct Tensor* tensor, struct Tensor* other, struct Tensor* result);
void sub_tensor_wrapper(struct Tensor* tensor, struct Tensor* other, struct Tensor* result);
void mul_tensor_wrapper(struct Tensor* tensor, struct Tensor* other, struct Tensor* result);
void div_tensor_wrapper(struct Tensor* tensor, struct Tensor* other, struct Tensor* result);
void add_scalar_tensor_wrapper(struct Tensor* tensor, int scalar, struct Tensor* result);
void sub_scalar_tensor_wrapper(struct Tensor* tensor, int scalar, struct Tensor* result);
void mul_scalar_tensor_wrapper(struct Tensor* tensor, int scalar, struct Tensor* result);
void div_scalar_tensor_wrapper(struct Tensor* tensor, int scalar, struct Tensor* result);
void transpose_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void sum_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void max_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void min_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void gradient_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void gate_tensor_wrapper(struct Tensor* tensor, struct Tensor* booleans, struct Tensor* result);
void vector_sort_tensor_wrapper(struct Tensor* tensor, struct Tensor* vectors, struct Tensor* result);
