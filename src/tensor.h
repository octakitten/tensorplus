#pragma once
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "defines.h"

    struct Tensor {
        int* data;
        int size;
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
void dot_tensor_wrapper(struct Tensor* tensor, struct Tensor* other, struct Tensor* result);
void transpose_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void sum_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void max_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void min_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void abs_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void exp_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void log_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void sqrt_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void pow_tensor_wrapper(struct Tensor* tensor, int power, struct Tensor* result);
void sin_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void cos_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void tan_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void asin_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void acos_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void atan_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void sinh_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void cosh_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void tanh_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void asinh_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void acosh_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void atanh_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void gradient_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void gate_tensor_wrapper(struct Tensor* tensor, struct Tensor* booleans, struct Tensor* result);
void vector_sort_tensor_wrapper(struct Tensor* tensor, struct Tensor* vectors, struct Tensor* result);