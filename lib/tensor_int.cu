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

extern "C" Tensor_int* create_tensor_int(unsigned int* dims) {
    Tensor_int* tensor = (Tensor_int*)malloc(sizeof(Tensor_int));
    tensor->dims = dims;
    unsigned int tmp = 1;
    (for (int i = 0, i < dims[0], i++) {
        tmp *= dims[i];
    }
    tensor->data = (int*)malloc(tmp * sizeof(int));
    return tensor;
}

extern "C" void destroy_tensor_int(Tensor_int* tensor) {
    free(tensor->data);
    free(tensor->dims);
    free(tensor);
}

extern "C" Tensor_int* create_tensor_int_device(unsigned int size, unsigned int* dims) {
    Tensor_int* tensor = (Tensor_int*)malloc(sizeof(Tensor_int));
    cudaMalloc(&tensor->size, sizeof(unsigned int));
    cudaMemcpy(tensor->size, &size, sizeof(unsigned int), cudaMemcpyHostToDevice);
    unsigned int tmp = 1;
    for (int i = 0; i < dims[0]; i++) {
        tmp *= dims[i];
    }
    cudaMalloc(&tensor->dims, sizeof(unsigned int) * dims[0]);
    cudaMemcpy(tensor->dims, dims, sizeof(unsigned int) * dims[0], cudaMemcpyHostToDevice);
    cudaMalloc(&tensor->data, sizeof(int) * tmp);
    return tensor;
}

extern "C" void destroy_tensor_int_device(Tensor_int* tensor) {
    cudaFree(tensor->data);
    cudaFree(tensor->dims);
    cudaFree(tensor->size);
    free(tensor);
}

extern "C" void get_tensor_int_size(Tensor_int* tensor, unsigned int* size) {
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

extern "C" void copy_tensor_int(Tensor_int* tensor, Tensor_int* result) {
    result = tensor;
}

extern "C" Tensor_int* clone_tensor_int(Tensor_int* tensor) {
    Tensor_int* result = create_tensor_int_device(tensor->size, tensor->dims);
    cudaMemcpy(result->data, tensor->data, sizeof(int) * tensor->size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(result->dims, tensor->dims, sizeof(unsigned int) * tensor->dims[0], cudaMemcpyDeviceToDevice);
    cudaMemcpy(result->size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    return result;
}

__global__ void add_tensor_int_kernel(unsigned int* dims, int* src, int* other, int* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = src[index] + other[index];
    }
}

__global__ void sub_tensor_int_kernel(unsigned int* dims, int* src, int* other, int* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = src[index] - other[index];
    }
}

__global__ void mul_tensor_int_kernel(unsigned int* dims, int* src, int* other, int* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = src[index] * other[index];
    }
}

__global__ void div_tensor_int_kernel(unsigned int* dims, int* src, int* other, int* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if (other[index] != 0) {
            result[index] = src[index] / other[index];
        }
    }
}

__global__ void logical_not_tensor_int_kernel(unsigned int* dims, int* src, int* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if (src[index] == 0) {
            result[index] = 1;
        } else {
            result[index] = 0;
        }
    }
}

__global__ void logical_and_tensor_int_kernel(unsigned int* dims, int* src, int* other, int* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if (src[index] != 0 && other[index] != 0) {
            result[index] = 1;
        } else {
            result[index] = 0;
        }
    }
}
__global__ void logical_or_tensor_int_kernel(unsigned int* dims, int* src, int* other, int* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if (src[index] != 0 || other[index] != 0) {
            result[index] = 1;
        } else {
            result[index] = 0;
        }
    }
}

__global__ void logical_xor_tensor_int_kernel(unsigned int* dims, int* src, int* other, int* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if ((src[index] != 0 && other[index] == 0) || (src[index] == 0 && other[index] != 0)) {
            result[index] = 1;
        } else {
            result[index] = 0;
        }
    }
}

__global__ void logical_equals_tensor_int_kernel(unsigned int* dims, int* src, int* other, int* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if (src[index] == other[index]) {
            result[index] = 1;
        } else {
            result[index] = 0;
        }
    }
}

__global__ void logical_not_equals_tensor_int_kernel(unsigned int* dims, int* src, int* other, int* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if (src[index] != other[index]) {
            result[index] = 1;
        } else {
            result[index] = 0;
        }
    }
}

__global__ void logical_greater_than_tensor_int_kernel(unsigned int* dims, int* src, int* other, int* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if (src[index] > other[index]) {
            result[index] = 1;
        } else {
            result[index] = 0;
        }
    }
}

__global__ void logical_less_than_tensor_int_kernel(unsigned int* dims, int* src, int* other, int* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if (src[index] < other[index]) {
            result[index] = 1;
        } else {
            result[index] = 0;
        }
    }
}

__global__ void logical_greater_than_equals_tensor_int_kernel(unsigned int* dims, int* src, int* other, int* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if (src[index] >= other[index]) {
            result[index] = 1;
        } else {
            result[index] = 0;
        }
    }
}

__global__ void logical_less_than_equals_tensor_int_kernel(unsigned int* dims, int* src, int* other, int* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if (src[index] <= other[index]) {
            result[index] = 1;
        } else {
            result[index] = 0;
        }
    }
}

__global__ void transpose_tensor_int_kernel(unsigned int* dims, int* src int* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index <= dims[dims[0] + 1]) {
        result->data[index] = src->data[dims[dims[0] + 1] - index];
    }
}
// vector operations do the same as their standard counterparts but follow the vectors tensor to get the indices of the elements from 
// the other tensor that it needs to operate on.

__global__ void vector_logical_equals_tensor_int_kernel(unsigned int* dims, int* src, int* other, int* vectors, int* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = (src[index] == other[vectors[index]]) ? 1 : 0;
    }
}

__global__ void vector_logical_not_equals_tensor_int_kernel(unsigned int* dims, int* src, int* other, int* vectors, int* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = (src[index] != other[vectors[index]]) ? 1 : 0;
    }
}

__global__ void vector_logical_greater_than_tensor_int_kernel(unsigned int* dims, int* src, int* other, int* vectors, int* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = (src[index] > other[vectors[index]]) ? 1 : 0;
    }
}

__global__ void vector_logical_less_than_tensor_int_kernel(unsigned int* dims, int* src, int* other, int* vectors, int* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = (src[index] < other[vectors[index]]) ? 1 : 0;
    }
}

__global__ void vector_logical_greater_than_equals_tensor_int_kernel(unsigned int* dims, int* src, int* other, int* vectors, int* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = (src[index] >= other[vectors[index]]) ? 1 : 0;
    }
}

__global__ void vector_logical_less_than_equals_tensor_int_kernel(unsigned int* dims, int* src, int* other, int* vectors, int* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = (src[index] <= other[vectors[index]]) ? 1 : 0;
    }
}

__global__ void vector_logical_and_tensor_int_kernel(unsigned int* dims, int* src, int* other, int* vectors, int* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if (src[index] != 0 && other[vectors[index]] != 0) {
            result[index] = 1;
        } else {
            result[index] = 0;
        }
    }
}

__global__ void vector_logical_or_tensor_int_kernel(unsigned int* dims, int* src, int* other, int* vectors, int* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if (src[index] != 0 || other[vectors[index]] != 0) {
            result[index] = 1;
        } else {
            result[index] = 0;
        }
    }
}

__global__ void vector_logical_xor_tensor_int_kernel(unsigned int* dims, int* src, int* other, int* vectors, int* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if ((src[index] != 0 && other[vectors[index]] == 0) || (src[index] == 0 && other[vectors[index]] != 0)) {
            result[index] = 1;
        } else {
            result[index] = 0;
        }
    }
}

__global__ void vector_logical_nor_tensor_int_kernel(unsigned int* dims, int* src, int* other, int* vectors, int* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if (!(src[index] != 0 || other[vectors[index]] != 0)) {
            result[index] = 1;
        } else {
            result[index] = 0;
        }
    }
}

__global__ void vector_sort_tensor_int_kernel(unsigned int* dims, int* src, int* vectors, int* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = src[vectors[index]];
    }
}

__global__ void vector_add_tensor_int_kernel(unsigned int* dims, int* src, int* other, int* vectors, int* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = src[[index] + other[vectors[index]];
    }
}

__global__ void vector_sub_tensor_int_kernel(unsigned int* dims, int* src, int* other, int* vectors, int* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = src[index]- other[vectors[index]];
    }
}

__global__ void vector_mul_tensor_int_kernel(unsigned int* dims, int* src, int* other, int* vectors, int* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = src[index] * other[vectors[index]];
    }
}

__global__ void vector_div_tensor_int_kernel(unsigned int* dims, int* src, int* other, int* vectors, int* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = src[index] / other[vectors[index]];
    }
}

__global__ void vector_mod_tensor_int_unsigned(int* dims, int* src, int* other, int* vectors, int* result) {
    unsigned int index = blockIdx.x a* blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        result[index] = src[index] % other[vectors[index]];
    }
}
 
__global__ void vector_gate_tensor_int_kernel(unsigned int* dims, int* src, int* other, int* vectors, int* result) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dims[dims[0] + 1]) {
        if (other[vectors[index]] != 0) {
            result[index] = src[index];
        }
    }
}


extern "C" void add_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudamemcpy(dims, src->dims[0], sizeof(unsigned int) * (2 + dim0), cudaMemcpymdeviceToHost);
    add_tensor_int_kernel<<<(dims[dims[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, result->data);
    cudaDeviceSynchronize();
}


extern "C" void sub_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudamemcpy(dims, src->dims[0], sizeof(unsigned int) * (2 + dim0), cudaMemcpymdeviceToHost);
    sub_tensor_int_kernel<<<(dims[dims[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void mul_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudamemcpy(dims, src->dims[0], sizeof(unsigned int) * (2 + dim0), cudaMemcpymdeviceToHost);
    mul_tensor_int_kernel<<<(dims[dims[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void div_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudamemcpy(dims, src->dims[0], sizeof(unsigned int) * (2 + dim0), cudaMemcpymdeviceToHost);
    div_tensor_int_kernel<<<(dims[dims[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, result->data);
    cudaDeviceSynchronize();
}


extern "C" void logical_not_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudamemcpy(dims, src->dims[0], sizeof(unsigned int) * (2 + dim0), cudaMemcpymdeviceToHost);
    logical_not_tensor_int_kernel<<<(dims[dims[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void logical_and_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudamemcpy(dims, src->dims[0], sizeof(unsigned int) * (2 + dim0), cudaMemcpymdeviceToHost);
    logical_and_tensor_int_kernel<<<(dims[dims[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void logical_or_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudamemcpy(dims, src->dims[0], sizeof(unsigned int) * (2 + dim0), cudaMemcpymdeviceToHost);
    logical_or_tensor_int_kernel<<<(dims[dims[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void logical_xor_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudamemcpy(dims, src->dims[0], sizeof(unsigned int) * (2 + dim0), cudaMemcpymdeviceToHost);
    logical_xor_tensor_int_kernel<<<(dims[dims[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void logical_equals_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudamemcpy(dims, src->dims[0], sizeof(unsigned int) * (2 + dim0), cudaMemcpymdeviceToHost);
    logical_equals_tensor_int_kernel<<<(dims[dims[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void logical_not_equals_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudamemcpy(dims, src->dims[0], sizeof(unsigned int) * (2 + dim0), cudaMemcpymdeviceToHost);
    logical_not_equals_tensor_int_kernel<<<(dims[dims[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void logical_greater_than_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudamemcpy(dims, src->dims[0], sizeof(unsigned int) * (2 + dim0), cudaMemcpymdeviceToHost);
    logical_greater_than_tensor_int_kernel<<<(dims[dims[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void logical_less_than_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudamemcpy(dims, src->dims[0], sizeof(unsigned int) * (2 + dim0), cudaMemcpymdeviceToHost);
    div_tensor_int_kernel<<<(dims[dims[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void logical_greater_than_equals_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudamemcpy(dims, src->dims[0], sizeof(unsigned int) * (2 + dim0), cudaMemcpymdeviceToHost);
    logical_greater_than_equals_tensor_int_kernel<<<(dims[dims[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void logical_less_than_equals_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudamemcpy(dims, src->dims[0], sizeof(unsigned int) * (2 + dim0), cudaMemcpymdeviceToHost);
    logial_less_than_equals_tensor_int_kernel<<<(dims[dims[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void transpose_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudamemcpy(dims, src->dims[0], sizeof(unsigned int) * (2 + dim0), cudaMemcpymdeviceToHost);
    transpose_tensor_int_kernel<<<(dims[dims[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, result->data);
    cudaDeviceSynchronize();
}

// vector operations start here
extern "C" void vector_add_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* vectors, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_add_tensor_int_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}


extern "C" void vector_sub_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* vectors, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_sub_tensor_int_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}


extern "C" void vector_mul_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* vectors, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_mul_tensor_int_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_div_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* vectors, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_div_tensor_int_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_mod_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* vectors, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    mod_add_tensor_int_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_gate_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* vectors, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_gate_tensor_int_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_logical_equals_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* vectors, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_logical_equals_tensor_int_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_logical_not_equals_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* vectors, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_logical_not_equals_tensor_int_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_logical_less_than_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* vectors, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_logical_less_than_tensor_int_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_logical_greater_than_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* vectors, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_logical_greater_than_tensor_int_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_logical_less_than_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* vectors, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_logical_less_than_tensor_int_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_logical_greater_than_equals_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* vectors, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_logical_greater_than_equals_tensor_int_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_less_than_equals_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* vectors, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_logical_less_than_equals_tensor_int_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_logical_and_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* vectors, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_logical_and_tensor_int_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_logical_or_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* vectors, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_logical_or_tensor_int_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_logical_xor_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* vectors, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_logical_xor_tensor_int_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_logical_nor_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* vectors, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_logical_nor_tensor_int_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}

extern "C" void vector_gate_tensor_int(Tensor_int* src, Tensor_int* other, Tensor_int* vectors, Tensor_int* result) {
    unsigned int dim0;
    cudaMemcpy(dim0, src->dims[0], sizeof(unsigned int), cudamemcpyDeviceToHost);
    unsigned int* dims = (unsigned int*) malloc(sizeof(unsigned int) * (2 + dim0));
    cudaMemcpy(dims, src->dims[0],sizeof(unsigned int) * (2 + dim0) cudaMemcpyDeviceToHost);
    vector_gate_tensor_int_kernel<<<(dims[dime[0] + 1] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(src->dims, src->data, other->data, vectors->data, result->data);
    cudaDeviceSynchronize();
}
