#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <stdbool.h>
#include "defines.h
ifdef __cplusplus
extern "C" {
    #include "tensor.h"
}
#else
    #include "tensor.h/equal"
#endif

#define THREADS_PER_BLOCK 256
#define BLOCKS_MAXIMUM 65535

extern "C" Tensor_double* create_tensor_double(unsigned int* dims) {
    Tensor_double* tensor = (Tensor_double*)malloc(sizeof(Tensor_double));
    tensor->dims = dims;
    unsigned int tmp = 1;
    (for (int i = 0, i < dims[0], i++) {
        tmp *= dims[i];
    }
    tensor->data = (double*)malloc(tmp * sizeof(double));
    return tensor;
}

extern "C" void destroy_tensor_double(Tensor_double* tensor) {
    free(tensor->data);
    free(tensor->dims);
    free(tensor);
}

extern "C" Tensor_double* create_tensor_double_device(unsigned int size, unsigned int* dims) {
    Tensor_double* tensor = (Tensor_double*)malloc(sizeof(Tensor_double));
    cudaMalloc(&tensor->size, sizeof(unsigned int));
    cudaMemcpy(tensor->size, &size, sizeof(unsigned int), cudaMemcpyHostToDevice);
    unsigned int tmp = 1;
    for (int i = 0; i < dims[0]; i++) {
        tmp *= dims[i];
    }
    cudaMalloc(&tensor->dims, sizeof(unsigned int) * dims[0]);
    cudaMemcpy(tensor->dims, dims, sizeof(unsigned int) * dims[0], cudaMemcpyHostToDevice);
    cudaMalloc(&tensor->data, sizeof(double) * tmp);
    return tensor;
}

extern "C" void destroy_tensor_double_device(Tensor_double* tensor) {
    cudaFree(tensor->data);
    cudaFree(tensor->dims);
    cudaFree(tensor->size);
    free(tensor);
}

extern "C" void get_tensor_double_size(Tensor_double* tensor, unsigned int* size) {
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

extern "C" void copy_tensor_double(Tensor_double* tensor, Tensor_double* result) {
    result = tensor;
}

extern "C" Tensor_double* clone_tensor_double(Tensor_double* tensor) {
    Tensor_double* result = create_tensor_double_device(tensor->size, tensor->dims);
    cudaMemcpy(result->data, tensor->data, sizeof(double) * tensor->size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(result->dims, tensor->dims, sizeof(unsigned int) * tensor->dims[0], cudaMemcpyDeviceToDevice);
    cudaMemcpy(result->size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    return result;
}

__global__ void add_tensor_double_kernel(unsigned int* dims, double* src, double* other, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = src[index] + other[index];
    }
}

__global__ void sub_tensor_double_kernel(unsigned int* dims, double* src, double* other, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = src[index] - other[index];
    }
}

__global__ void mul_tensor_double_kernel(unsigned int* dims, double* src, double* other, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = src[index] * other[index];
    }
}

__global__ void div_tensor_double_kernel(unsigned int* dims, double* src, double* other, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if (other[index] != 0) {
            result[index] = src[index] / other[index];
        }
    }
}

__global__ void logical_not_tensor_double_kernel(unsigned int* dims, double* src, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if (src[index] == 0) {
            result[index] = 1;
        } else {
            result[index] = 0;
        }
    }
}

__global__ void logical_and_tensor_double_kernel(unsigned int* dims, double* src, double* other, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if (src[index] != 0 && other[index] != 0) {
            result[index] = 1;
        } else {
            result[index] = 0;
        }
    }
}
__global__ void logical_or_tensor_double_kernel(unsigned int* dims, double* src, double* other, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if (src[index] != 0 || other[index] != 0) {
            result[index] = 1;
        } else {
            result[index] = 0;
        }
    }
}

__global__ void logical_xor_tensor_double_kernel(unsigned int* dims, double* src, double* other, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if ((src[index] != 0 && other[index] == 0) || (src[index] == 0 && other[index] != 0)) {
            result[index] = 1;
        } else {
            result[index] = 0;
        }
    }
}

__global__ void logical_equals_tensor_double_kernel(unsigned int* dims, double* src, double* other, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if (src[index] == other[index]) {
            result[index] = 1;
        } else {
            result[index] = 0;
        }
    }
}

__global__ void logical_not_equals_tensor_double_kernel(unsigned int* dims, double* src, double* other, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if (src[index] != other[index]) {
            result[index] = 1;
        } else {
            result[index] = 0;
        }
    }
}

__global__ void logical_greater_than_tensor_double_kernel(unsigned int* dims, double* src, double* other, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if (src[index] > other[index]) {
            result[index] = 1;
        } else {
            result[index] = 0;
        }
    }
}

__global__ void logical_less_than_tensor_double_kernel(unsigned int* dims, double* src, double* other, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if (src[index] < other[index]) {
            result[index] = 1;
        } else {
            result[index] = 0;
        }
    }
}

__global__ void logical_greater_than_equals_tensor_double_kernel(unsigned int* dims, double* src, double* other, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if (src[index] >= other[index]) {
            result[index] = 1;
        } else {
            result[index] = 0;
        }
    }
}

__global__ void logical_less_than_equals_tensor_double_kernel(unsigned int* dims, double* src, double* other, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if (src[index] <= other[index]) {
            result[index] = 1;
        } else {
            result[index] = 0;
        }
    }
}

__global__ void transpose_tensor_double_kernel(unsigned int* dims, double* src double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index <= dims[dims[0] + 1]) {
        result->data[index] = src->data[dims[dims[0] + 1] - index];
    }
}
// vector operations do the same as their standard counterparts but follow the vectors tensor to get the indices of the elements from 
// the other tensor that it needs to operate on.

__global__ void vector_logical_equals_tensor_double_kernel(unsigned int* dims, double* src, double* other, double* vectors, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = (src[index] == other[vectors[index]]) ? 1 : 0;
    }
}

__global__ void vector_logical_not_equals_tensor_double_kernel(unsigned int* dims, double* src, double* other, double* vectors, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = (src[index] != other[vectors[index]]) ? 1 : 0;
    }
}

__global__ void vector_logical_greater_than_tensor_double_kernel(unsigned int* dims, double* src, double* other, double* vectors, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = (src[index] > other[vectors[index]]) ? 1 : 0;
    }
}

__global__ void vector_logical_less_than_tensor_double_kernel(unsigned int* dims, double* src, double* other, double* vectors, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = (src[index] < other[vectors[index]]) ? 1 : 0;
    }
}

__global__ void vector_logical_greater_than_equals_tensor_double_kernel(unsigned int* dims, double* src, double* other, double* vectors, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = (src[index] >= other[vectors[index]]) ? 1 : 0;
    }
}

__global__ void vector_logical_less_than_equals_tensor_double_kernel(unsigned int* dims, double* src, double* other, double* vectors, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = (src[index] <= other[vectors[index]]) ? 1 : 0;
    }
}

__global__ void vector_logical_and_tensor_double_kernel(unsigned int* dims, double* src, double* other, double* vectors, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if (src[index] != 0 && other[vectors[index]] != 0) {
            result[index] = 1;
        } else {
            result[index] = 0;
        }
    }
}

__global__ void vector_logical_or_tensor_double_kernel(unsigned int* dims, double* src, double* other, double* vectors, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if (src[index] != 0 || other[vectors[index]] != 0) {
            result[index] = 1;
        } else {
            result[index] = 0;
        }
    }
}

__global__ void vector_logical_xor_tensor_double_kernel(unsigned int* dims, double* src, double* other, double* vectors, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if ((src[index] != 0 && other[vectors[index]] == 0) || (src[index] == 0 && other[vectors[index]] != 0)) {
            result[index] = 1;
        } else {
            result[index] = 0;
        }
    }
}

__global__ void vector_logical_nor_tensor_double_kernel(unsigned int* dims, double* src, double* other, double* vectors, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if (!(src[index] != 0 || other[vectors[index]] != 0)) {
            result[index] = 1;
        } else {
            result[index] = 0;
        }
    }
}

__global__ void vector_sort_tensor_double_kernel(unsigned int* dims, double* src, double* vectors, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = src[vectors[index]];
    }
}

__global__ void vector_add_tensor_double_kernel(unsigned int* dims, double* src, double* other, double* vectors, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = src[[index] + other[vectors[index]];
    }
}

__global__ void vector_sub_tensor_double_kernel(unsigned int* dims, double* src, double* other, double* vectors, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = src[index]- other[vectors[index]];
    }
}

__global__ void vector_mul_tensor_double_kernel(unsigned int* dims, double* src, double* other, double* vectors, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = src[index] * other[vectors[index]];
    }
}

__global__ void vector_div_tensor_double_kernel(unsigned int* dims, double* src, double* other, double* vectors, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = src[index] / other[vectors[index]];
    }
}

__global__ void vector_mod_tensor_double_unsigned(int* dims, double* src, double* other, double* vectors, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = src[index] % other[vectors[index]];
    }
}

__global__ void vector_gate_tensor_double_kernel(unsigned int* dims, double* src, double* other, double* vectors, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if (other[vectors[index]] != 0) {
            result[index] = sin(src[index]);
        }
    }
}

__global__ void sin_tensor_double_kernel(unsigned int* dims, double* src, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = sin(src[index]);
    }
}

__global__ void cos_tensor_double_kernel(unsigned int* dims, double* src, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = cos(src[index]);
    }
}

__global__ void tan_tensor_double_kernel(unsigned int* dims, double* src, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = tan(src[index]);
    }
}

__global__ void sec_tensor_double_kernel(unsigned int* dims, double* src, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = 1 / sin(src[index]);
    }
}

__global__ void csc_tensor_double_kernel(unsigned int* dims, double* src, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = 1 / sin(src[index]);
    }
}

__global__ void cot_tensor_double_kernel(unsigned int* dims, double* src, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = 1 / tan(src[index]);
    }
}
__global__ void arcsin_tensor_double_kernel(unsigned int* dims, double* src, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = asin(src[index]);
    }
}

__global__ void arccos_tensor_double_kernel(unsigned int* dims, double* src, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = acos(src[index]);
    }
}

__global__ void arctan_tensor_double_kernel(unsigned int* dims, double* src, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = atan(src[index]);
    }
}


__global__ void vector_sin_tensor_double_kernel(unsigned int* dims, double* src, double* vectors, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = sin(src[vectors[index]]);
    }
}

__global__ void vector_cos_tensor_double_kernel(unsigned int* dims, double* src, double* vectors, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = cos(src[vectors[index]]);
    }
}

__global__ void vector_tan_tensor_double_kernel(unsigned int* dims, double* src, double* vectors, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = tan(src[vectors[index]]);
    }
}

__global__ void vector_arcsin_tensor_double_kernel(unsigned int* dims, double* src, double* vectors, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = asin(src[vectors[index]]);
    }
}

__global__ void vector_arccos_tensor_double_kernel(unsigned int* dims, double* src, double* vectors, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = acos(src[vectors[index]]);
    }
}

__global__ void vector_arctan_tensor_double_kernel(unsigned int* dims, double* src, double* vectors, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = atan(src[vectors[index]]);
    }
}

__global__ void pow_tensor_double_kernel(unsigned int* dims, double* src, double* other, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = pow(src[index], other[index]);
    }
}

__global__ void sqrt_tensor_double_kernel(unsigned int* dims, double* src, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = sqrt(src[index]);
    }
}

__global__ void exp_tensor_double_kernel(unsigned int* dims, double* src, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = exp(src[index]);
    }
}

__global__ void ln_tensor_double_kernel(unsigned int* dims, double* src, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = log(src[index]);
    }
}

__global__ void abs_tensor_double_kernel(unsigned int* dims, double* src, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = fabs(src[index]);
    }
}

__global__ void vector_pow_tensor_double_kernel(unsigned int* dims, double* src, double* other, double* vectors, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = pow(src[index], other[vectors[index]]);
    }
}

__global__ void vector_sqrt_tensor_double_kernel(unsigned int* dims, double* src, double* vectors, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = sqrt(src[vectors[index]]);
    }
}

__global__ void vector_exp_tensor_double_kernel(unsigned int* dims, double* src, double* vectors, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = exp(src[vectors[index]]);
    }
}

__global__ void vector_ln_tensor_double_kernel(unsigned int* dims, double* src, double* vectors, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = log(src[vectors[index]]);
    }
}

__global__ void vector_abs_tensor_double_kernel(unsigned int* dims, double* src, double* vectors, double* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = fabs(src[vectors[index]]);
    }
}


extern "C" void add_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudamemcpy(dims, src->dims[0], sizeof(unsigned int) * (2 + dim0), cudaMemcpymdeviceToHost);
    add_tensor_double_kernel<<<(dims[dims[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, result->data);
    cudaDeviceSynchronize();
}


extern "C" void sub_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudamemcpy(dims, src->dims[0], sizeof(unsigned int) * (2 + dim0), cudaMemcpymdeviceToHost);
    sub_tensor_double_kernel<<<(dims[dims[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void mul_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudamemcpy(dims, src->dims[0], sizeof(unsigned int) * (2 + dim0), cudaMemcpymdeviceToHost);
    mul_tensor_double_kernel<<<(dims[dims[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void div_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudamemcpy(dims, src->dims[0], sizeof(unsigned int) * (2 + dim0), cudaMemcpymdeviceToHost);
    div_tensor_double_kernel<<<(dims[dims[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, result->data);
    cudaDeviceSynchronize();
}


extern "C" void logical_not_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudamemcpy(dims, src->dims[0], sizeof(unsigned int) * (2 + dim0), cudaMemcpymdeviceToHost);
    logical_not_tensor_double_kernel<<<(dims[dims[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void logical_and_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudamemcpy(dims, src->dims[0], sizeof(unsigned int) * (2 + dim0), cudaMemcpymdeviceToHost);
    logical_and_tensor_double_kernel<<<(dims[dims[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void logical_or_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudamemcpy(dims, src->dims[0], sizeof(unsigned int) * (2 + dim0), cudaMemcpymdeviceToHost);
    logical_or_tensor_double_kernel<<<(dims[dims[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void logical_xor_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudamemcpy(dims, src->dims[0], sizeof(unsigned int) * (2 + dim0), cudaMemcpymdeviceToHost);
    logical_xor_tensor_double_kernel<<<(dims[dims[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void logical_equals_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudamemcpy(dims, src->dims[0], sizeof(unsigned int) * (2 + dim0), cudaMemcpymdeviceToHost);
    logical_equals_tensor_double_kernel<<<(dims[dims[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void logical_not_equals_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudamemcpy(dims, src->dims[0], sizeof(unsigned int) * (2 + dim0), cudaMemcpymdeviceToHost);
    logical_not_equals_tensor_double_kernel<<<(dims[dims[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void logical_greater_than_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudamemcpy(dims, src->dims[0], sizeof(unsigned int) * (2 + dim0), cudaMemcpymdeviceToHost);
    logical_greater_than_tensor_double_kernel<<<(dims[dims[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void logical_less_than_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudamemcpy(dims, src->dims[0], sizeof(unsigned int) * (2 + dim0), cudaMemcpymdeviceToHost);
    div_tensor_double_kernel<<<(dims[dims[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void logical_greater_than_equals_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudamemcpy(dims, src->dims[0], sizeof(unsigned int) * (2 + dim0), cudaMemcpymdeviceToHost);
    logical_greater_than_equals_tensor_double_kernel<<<(dims[dims[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void logical_less_than_equals_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudamemcpy(dims, src->dims[0], sizeof(unsigned int) * (2 + dim0), cudaMemcpymdeviceToHost);
    logial_less_than_equals_tensor_double_kernel<<<(dims[dims[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void transpose_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudamemcpy(dims, src->dims[0], sizeof(unsigned int) * (2 + dim0), cudaMemcpymdeviceToHost);
    transpose_tensor_double_kernel<<<(dims[dims[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, result->data);
    cudaDeviceSynchronize();
}

// vector operations start here
extern "C" void vector_add_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* vectors, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_add_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}


extern "C" void vector_sub_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* vectors, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_sub_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}


extern "C" void vector_mul_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* vectors, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_mul_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_div_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* vectors, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_div_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_mod_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* vectors, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    mod_add_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_gate_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* vectors, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_gate_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_logical_equals_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* vectors, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_logical_equals_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_logical_not_equals_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* vectors, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_logical_not_equals_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_logical_less_than_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* vectors, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_logical_less_than_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_logical_greater_than_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* vectors, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_logical_greater_than_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_logical_less_than_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* vectors, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_logical_less_than_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_logical_greater_than_equals_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* vectors, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_logical_greater_than_equals_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_less_than_equals_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* vectors, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_logical_less_than_equals_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_logical_and_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* vectors, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_logical_and_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_logical_or_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* vectors, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_logical_or_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_logical_xor_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* vectors, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_logical_xor_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_logical_nor_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* vectors, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_logical_nor_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_gate_tensor_double(Tensor_double* src, Tensor_double* other, Tensor_double* vectors, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_gate_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}


extern "C" void sin_tensor_double(Tensor_double* src, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    sin_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void cos_tensor_double(Tensor_double* src, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    cos_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void tan_tensor_double(Tensor_double* src, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    tan_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void sec_tensor_double(Tensor_double* src, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    sec_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void csc_tensor_double(Tensor_double* src, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    csc_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void cot_tensor_double(Tensor_double* src, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    cot_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void arcsin_tensor_double(Tensor_double* src, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    arcsin_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, result->data);
    cudaDeviceSynchronize();
}


extern "C" void arccos_tensor_double(Tensor_double* src, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    arccos_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void arctan_tensor_double(Tensor_double* src, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    arctan_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_sin_tensor_double(Tensor_double* src, Tensor_double* vectors, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    sin_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_cos_tensor_double(Tensor_double* src, Tensor_double* vectors, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    cos_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_tan_tensor_double(Tensor_double* src, Tensor_double* vectors, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    tan_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_arcsin_tensor_double(Tensor_double* src, Tensor_double* vectors, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    arcsin_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_arccos_tensor_double(Tensor_double* src, Tensor_double* vectors, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    arccos_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_arctan_tensor_double(Tensor_double* src, Tensor_double* vectors, Tensor_double* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    arctan_tensor_double_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(src->dims, src->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

