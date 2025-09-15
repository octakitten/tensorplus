#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "defines.h"
#ifdef __cplusplus
extern "C" {
    #include "tensor.h"
}
#else
    #include "tensor.h/equal"
#endif
#ifdef __DEBUG__
#endif

#define THREADS_PER_BLOCK 256
#define BLOCKS_MAXIMUM  65536

void set_cpu_to_device_tensor(Tensor* tensor);
void set_device_to_cpu_tensor(Tensor* tensor);
void get_tensor_size_wrapper(Tensor* tensor, unsigned int size);
void copy_device_to_device_tensor(Tensor* tensor, Tensor* result);
void copy_device_to_cpu_tensor(Tensor* tensor, Tensor* result);
void copy_cpu_to_device_tensor(Tensor* tensor, Tensor* result);
void copy_cpu_to_cpu_tensor(Tensor* tensor, Tensor* result);
__global__ void vector_sort_tensor(unsigned int* size, short* src, short* other, short* vectors, short* result);
__global__ void vector_add_tensor(unsigned int* size, short* src, short* other, short* vectors, short* result);
__global__ void vector_sub_tensor(unsigned int* size, short* src, short* other, short* vectors, short* result);
__global__ void vector_mul_tensor(unsigned int* size, short* src, short* other, short* vectors, short* result);
__global__ void vector_div_tensor(unsigned int* size, short* src, short* other, short* vectors, short* result);
__global__ void vector_gate_tensor(unsigned int* size, short* src, short* booleans, short* vectors, short* result);
__global__ void set_tensor(unsigned int* size, short* src, unsigned int* index, short* value);
__global__ void get_tensor_value(unsigned int* size, short* src, unsigned int* index, short* result);
__global__ void zeros_tensor( unsigned int* size, short* data );
__global__ void ones_tensor( unsigned int* size, short* data);
__global__ void print_tensor( unsigned int* size, short* data);
__global__ void fill_tensor( unsigned int* size, short* data, short* value);
__global__ void add_tensor( unsigned int* size, short* data1, short* data2, short* data3);
__global__ void sub_tensor( unsigned int* size, short* data1, short* data2, short* data3);
__global__ void mul_tensor( unsigned int* size, short* data1, short* data2, short* data3);
__global__ void div_tensor( unsigned int* size, short* data1, short* data2, short* data3);
__global__ void add_scalar_tensor(unsigned int* size, short* src, short* value, short* result);
__global__ void sub_scalar_tensor(unsigned int* size, short* src, short* value, short* result);
__global__ void mul_scalar_tensor(unsigned int* size, short* src, short* value, short* result);
__global__ void div_scalar_tensor(unsigned int* size, short* src, short* value, short* result);
__global__ void transpose_tensor(unsigned int* size, short* src, short* result);
__global__ void sum_tensor(unsigned int* size, short* src, short* result);
__global__ void mean_tensor(unsigned int* size, short* src, short* result);
__global__ void max_tensor(unsigned int* size, short* src, short* result);
__global__ void min_tensor(unsigned int* size, short* src, short* result);
__global__ void gradient_tensor(unsigned int* size, short* src, short* result);
__global__ void gate_tensor(unsigned int* size, short* src, short* bools, short* result);
bool check_if_cuda_tensor(Tensor* tensor);
__global__ void negate_tensor(Tensor* tensor, Tensor* result);
bool check_size( Tensor* tensor,  Tensor* other);
int get_device_dim(unsigned int size);
int get_device_dim_remainder(unsigned int size);



extern "C" Tensor* create_tensor(unsigned int size) {
    Tensor *tensor = (Tensor*) malloc(sizeof(Tensor));
    tensor->size = (unsigned int*) malloc(sizeof(unsigned int));
    tensor->size = &size;
    tensor->data = (short*) malloc(size*sizeof(short)); 
    return tensor;
}

extern "C" void destroy_tensor(Tensor* tensor) {
    free(tensor->data);
    free(tensor->size);
    free(tensor);
    tensor = NULL;
}

extern "C" void get_tensor_size_wrapper(Tensor* tensor, unsigned int size) {
    cudaError_t err = cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("Printing the size from get_tensor_size_wrapper() function: %d\n", size);
}

void set_cpu_to_device_tensor(Tensor* tensor) {
    Tensor* d_tensor = (Tensor*) malloc(sizeof(Tensor));
    cudaMalloc((void **) &d_tensor->size, sizeof(unsigned int));
    cudaMemcpy(d_tensor->size, tensor->size, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_tensor->data, tensor->size[0] * sizeof(short));
    cudaMemcpy(d_tensor->data, tensor->data, tensor->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    free(tensor->data);
    free(tensor->size);
    tensor = d_tensor;
    d_tensor = NULL;
    free(d_tensor);
}

void set_device_to_cpu_tensor(Tensor* tensor) {
    Tensor* c_tensor = (Tensor*) malloc(sizeof(Tensor*));
    c_tensor->size = (unsigned int*) malloc(sizeof(unsigned int));
    cudaMemcpy(&c_tensor->size, &tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    c_tensor->data = (short*) malloc(sizeof(short) * c_tensor->size[0]);
    cudaMemcpy(&c_tensor->data, &tensor->data, sizeof(short) * c_tensor->size[0], cudaMemcpyDeviceToHost);
    cudaFree(tensor->size);
    cudaFree(tensor->data);
    tensor = c_tensor;
    c_tensor = NULL;
}

void copy_device_to_device_tensor(Tensor* tensor, Tensor* result) {
    cudaMemcpy(&result->size, &tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(&result->data, &tensor->data, sizeof(short) * tensor->size[0], cudaMemcpyDeviceToDevice);
}

void copy_device_to_cpu_tensor(Tensor* tensor, Tensor* result) {
    cudaMemcpy(&result->size, &tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result->data, &tensor->data, sizeof(short) * tensor->size[0], cudaMemcpyDeviceToHost);
}

void copy_cpu_to_device_tensor(Tensor* tensor, Tensor* result) {
    cudaMemcpy(&result->size, &tensor->size, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(&result->data, &tensor->data, sizeof(short) * tensor->size[0], cudaMemcpyHostToDevice);
}

void copy_cpu_to_cpu_tensor(Tensor* tensor, Tensor* result) {
    memcpy(&result->size, &tensor->size, sizeof(unsigned int));
    memcpy(&result->data, &tensor->data, sizeof(short) * tensor->size[0]);
}

extern "C" void clone_tensor(Tensor* tensor, Tensor* result) {
  bool is_cuda1 = check_if_cuda_tensor(tensor);
  bool is_cuda2 = check_if_cuda_tensor(result);
  if (is_cuda1 && is_cuda2) {
    copy_device_to_device_tensor(tensor, result);
  } else if (!is_cuda1 && !is_cuda2) {
    copy_cpu_to_cpu_tensor(tensor, result);
  } else if (is_cuda1 && !is_cuda2) {
    copy_device_to_cpu_tensor(tensor, result);
  } else if (!is_cuda1 && is_cuda2) {
    copy_cpu_to_device_tensor(tensor, result);
  }
}

extern "C" void copy_tensor(Tensor* tensor, Tensor* result) {
  result = tensor;
}

bool check_if_cuda_tensor(Tensor* tensor) {
  cudaPointerAttributes attributes;
  cudaPointerGetAttributes(&attributes, tensor->size);
  if (attributes.type == cudaMemoryTypeHost) {
    return false;
  } else if (attributes.type == cudaMemoryTypeDevice) {
    return true;
  } else {
    return false;
  }
}

extern "C" Tensor* create_device_tensor(unsigned int size) {
    //printf("step 1\n");
    unsigned int temp = size;
    Tensor *tensor = (Tensor*) malloc(sizeof(Tensor*));
    //printf("step 2\n");
    //printf("%d\n", size);
    cudaError_t err = cudaMalloc((void **) &tensor->size, sizeof(unsigned int));
    //printf("%s\n", cudaGetErrorString(err));
    //printf("step 3\n");
    err = cudaMemcpy(tensor->size, &temp, sizeof(unsigned int), cudaMemcpyHostToDevice);
    //printf("%s\n", cudaGetErrorString(err));
    //printf("step 4\n");
    err = cudaMalloc((void **) &tensor->data, sizeof(short) * size);
    //printf("%s\n", cudaGetErrorString(err));
    //printf("step 5\n");
    err = cudaDeviceSynchronize();
    //printf("%s\n", cudaGetErrorString(err));
    
    return tensor;
}

extern "C" void destroy_device_tensor(Tensor* tensor) {
    cudaFree(tensor->data);
    cudaFree(tensor->size);
    free(tensor);
    tensor = NULL;
}

int init_cpu_tensor(Tensor* tensor, unsigned int size) {
    tensor->size = &size;
    tensor->data = (short*)malloc(size * sizeof(short));
    if (!tensor->data) {
        return -1;
    }
    return 0;
}

__global__
void vector_sort_tensor(unsigned int* size, short* src, short* vectors, short* result) {
    unsigned int index = threadIdx.x;
    if (index < size[0]) {
        result[index] = src[vectors[index]];
    }
}

__global__
void vector_add_tensor(unsigned int* size, short* src, short* other, short* vectors, short* result) {
    extern __shared__ short tmp_src[];
    extern __shared__ short tmp_other[];
    extern __shared__ short tmp_vects[];
    extern __shared__ short tmp_result[];
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int block_index = threadIdx.x;
    if (index < size[0]) {
        tmp_src[block_index] = src[index];
        tmp_other[block_index] = other[index];
        tmp_vects[block_index] = vectors[index];
        tmp_result[block_index] = tmp_src[block_index] + tmp_other[tmp_vects[block_index]];
        
        __syncthreads();
        result[index] = tmp_result[0];
    }
}

__global__
void vector_sub_tensor(unsigned int* size, short* src, short* other, short* vectors, short* result) {
    extern __shared__ short tmp_src[];
    extern __shared__ short tmp_other[];
    extern __shared__ short tmp_vects[];
    extern __shared__ short tmp_result[];
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int block_index = threadIdx.x;
    if (index < size[0]) {
        tmp_src[block_index] = src[index];
        tmp_other[block_index] = other[index];
        tmp_vects[block_index] = vectors[index];
        tmp_result[block_index] = tmp_src[block_index] - tmp_other[tmp_vects[block_index]];
        
        __syncthreads();
        result[index] = tmp_result[0];
    }
}

__global__
void vector_mul_tensor(unsigned int* size, short* src, short* other, short* vectors, short* result) {
    extern __shared__ short tmp_src[];
    extern __shared__ short tmp_other[];
    extern __shared__ short tmp_vects[];
    extern __shared__ short tmp_result[];
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int block_index = threadIdx.x;
    if (index < size[0]) {
        tmp_src[block_index] = src[index];
        tmp_other[block_index] = other[index];
        tmp_vects[block_index] = vectors[index];
        tmp_result[block_index] = tmp_src[block_index] * tmp_other[tmp_vects[block_index]];
        
        __syncthreads();
        result[index] = tmp_result[0];
    }
}

__global__
void vector_div_tensor(unsigned int* size, short* src, short* other, short* vectors, short* result) {
    extern __shared__ short tmp_src[];
    extern __shared__ short tmp_other[];
    extern __shared__ short tmp_vects[];
    extern __shared__ short tmp_result[];
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int block_index = threadIdx.x;
    if (index < size[0]) {
        tmp_src[block_index] = src[index];
        tmp_other[block_index] = other[index];
        tmp_vects[block_index] = vectors[index];
        tmp_result[block_index] = tmp_src[block_index] / tmp_other[tmp_vects[block_index]];
        
        __syncthreads();
        result[index] = tmp_result[0];
    }
}

__global__
void vector_gate_tensor(unsigned int* size, short* src, short* booleans, short* vectors, short* result) {
    extern __shared__ short tmp_src[];
    extern __shared__ short tmp_bools[];
    extern __shared__ short tmp_vects[];
    extern __shared__ short tmp_result[];
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int block_index = threadIdx.x;
    if (index < size[0]) {
        tmp_src[block_index] = src[index];
        tmp_bools[block_index] = booleans[index];
        tmp_vects[block_index] = vectors[index];
        if (tmp_bools[block_index] == 1) {
            tmp_result[block_index] = tmp_src[tmp_vects[block_index]];
        }
        __syncthreads();
        result[index] = tmp_result[0];
    }
}

__global__
void set_tensor(unsigned int* size, short* src, unsigned int* index, short* value) {
    if (index[0] < size[0]) {
        src[index[0]] = value[0];
    }
}
 
__global__ void get_tensor_value(unsigned int* size, short* src, unsigned int* index, short* result) {
    if (index[0] < size[0]) {
        result[0] = src[index[0]];
    }
    //printf("Value gotten from within the kernel: %d\n", src[index[0]]);
}

        
        
        __global__ void zeros_tensor( unsigned int* size, short* data) {
            extern __shared__ short tmp[];
            unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int block_index = threadIdx.x;
            //printf("Printing from within zeros tensor kernel! Index: %d\n", index);

            if (index < size[0]) {
                tmp[block_index] = 0;
                //printf("setting to %d\n", value);
            }

            __syncthreads();
            data[index] = tmp[0];
            //data[index] = tmp[0];
        }

        __global__
        void ones_tensor( unsigned int* size, short* data) {
            extern __shared__ short tmp[];
            unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int block_index = threadIdx.x;
            //printf("printing from within ones tensor kernel!\n");
            if (index < size[0]) {
                tmp[block_index] = 1;
                //printf("setting to %d\n", value);
            }
             __syncthreads();
            data[index] = tmp[0];
        }

        __global__
        void print_tensor( unsigned int* size, short* data) {
            extern __shared__ short tmp[];
            unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int block_index = threadIdx.x;
            if (index < size[0]) {
                tmp[block_index] = data[index];
                printf("index: %d, value: %d\n", index, tmp[block_index]);
            }
            __syncthreads();
        }
            

        __global__
        void fill_tensor( unsigned int* size, short* data, short* value) {
            extern __shared__ short tmp[];
            unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int block_index = threadIdx.x;
            if (index < size[0]) {
                short tmp2 = value[0];
                tmp[block_index] = tmp2;
                //printf("setting to %d\n", value[0]);
            }
            __syncthreads();
            data[index] = tmp[0];
            //unsigned int stride = blockDim.x - 1;
            //printf("Thread index: %d\n", index);
            //printf("Stride: %d\n", stride);
            //for (unsigned int i = index; i < tensor->size[0]; i += stride) {
            //    tensor->data[i] = value;
            //    printf("Setting to %d\n", value);
            //}
        }

        __global__
        void add_tensor( unsigned int* size, short* data1, short* data2, short* data3) {
            extern __shared__ short tmp1[];
            extern __shared__ short tmp2[];
            unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int block_index = threadIdx.x;
            if (index < size[0]) {
                tmp1[block_index] = tmp1[block_index] + tmp2[block_index];
            }
            __syncthreads();
            data3[index] = tmp1[0];
        }

        __global__
        void sub_tensor( unsigned int* size, short* data1, short* data2, short* data3) {
            extern __shared__ short tmp1[];
            extern __shared__ short tmp2[];
            unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int block_index = threadIdx.x;
            if (index < size[0]) {
                tmp1[block_index] = tmp1[block_index] - tmp2[block_index];
            }
            __syncthreads();
            data3[index] = tmp1[0];
        }

        __global__
        void mul_tensor( unsigned int* size, short* data1, short* data2, short* data3) {
            extern __shared__ short tmp1[];
            extern __shared__ short tmp2[];
            unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int block_index = threadIdx.x;
            if (index < size[0]) {
                tmp1[block_index] = tmp1[block_index] * tmp2[block_index];
            }
            __syncthreads();
            data3[index] = tmp1[0];
        }

        __global__
        void div_tensor( unsigned int* size, short* data1, short* data2, short* data3) {
            extern __shared__ short tmp1[];
            extern __shared__ short tmp2[];
            unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int block_index = threadIdx.x;
            if (index < size[0]) {
                tmp1[block_index] = tmp1[block_index] / tmp2[block_index];
            }
            __syncthreads();
            data3[index] = tmp1[0];
        }

        __global__
        void add_scalar_tensor(unsigned int* size, short* src, short* value,  short* result) {
            extern __shared__ short tmp[];
            extern __shared__ short tmp_value;
            unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int block_index = threadIdx.x;
            if (index < size[0]) {
                tmp[block_index] = tmp[block_index] + tmp_value;
            }
            __syncthreads();
            result[index] = tmp[0];
        }

        __global__
        void sub_scalar_tensor(unsigned int* size, short* src, short* value, short* result) {
            extern __shared__ short tmp[];
            extern __shared__ short tmpval;
            unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int block_index = threadIdx.x;
            if (index < size[0]) {
                tmp[block_index] = tmp[block_index] - tmpval;
            }
            __syncthreads();
            result[index] = tmp[0];
        }

        __global__
        void mul_scalar_tensor(unsigned int* size, short* src, short* value, short* result) {
            extern __shared__ short tmp[];
            extern __shared__ short tmpval;
            unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int block_index = threadIdx.x;
            if (index < size[0]) {
                tmp[block_index] = tmp[block_index] * tmpval;
            }
            __syncthreads();
            result[index] = tmp[0];
        }

        __global__
        void div_scalar_tensor(unsigned int* size, short* src, short* value, short* result) {
            extern __shared__ short tmp[];
            extern __shared__ short tmpval;
            unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int block_index = threadIdx.x;
            if (index < size[0]) {
                tmp[block_index] = tmp[block_index] / tmpval;
            }
            __syncthreads();
            result[index] = tmp[0];
        }

        __global__
        void transpose_tensor(unsigned int* size, short* src, short* result) {
            extern __shared__ short tmp[];
            extern __shared__ short tmp2[];
            unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int block_index = threadIdx.x;
            if (index < size[0]) {
                tmp[block_index] = src[index];
                tmp2[block_index] = tmp[size[0] - index - 1];
            }
            __syncthreads();
            result[index] = tmp2[0];
        }

        __global__
        void sum_tensor( unsigned int* size, short* src, short* result) {
            extern __shared__ short tmp[];
            unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int block_index = threadIdx.x;
            if (index < size[0]) {
                result[0] += src[index];
            }
        }

        __global__
        void mean_tensor( unsigned int* size, short* src, short* result) {
            extern __shared__ short tmp[];
            unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int block_index = threadIdx.x;
            if (index < size[0]) {
                result[0] += src[index];
            }
            result[0] /= size[0];
        }

        __global__
        void max_tensor(unsigned int* size, short* src, short* result) {
            extern __shared__ short tmp[];
            unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int block_index = threadIdx.x;
            if (index < size[0]) {
                if (src[index] > result[0]) {
                    result[0] = src[index];
                }
            }
        }

        __global__
        void min_tensor(unsigned int* size, short* src, short* result) {
            extern __shared__ short tmp[];
            unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int block_index = threadIdx.x;
            if (index < size[0]) {
                if (src[index] < result[0]) {
                    result[0] = src[index];
                }
            }
        }

        __global__
        void gradient_tensor(unsigned int* size, short* src, short* result) {
            extern __shared__ short tmp[];
            unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int block_index = threadIdx.x;
            if (index < size[0]) {
                result[index] = src[index + 1] - src[index];
            }
        }

        __global__
        void gate_tensor(unsigned int* size, short* src, short* bools, short* result) {
            extern __shared__ short tmp[];
            extern __shared__ short tmp2[];
            extern __shared__ short tmp3[];
            unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int block_index = threadIdx.x;
            if (index < size[0]) {
              tmp[block_index] = src[index];
              tmp2[block_index] = bools[index];
                if (tmp2[block_index] == 0) {
                    tmp[block_index] = 0;
                }
            }
            __syncthreads();
            result[index] = tmp[0];
        }

        __global__ void lesser_tensor(unsigned int* size, short* src, short* other, short* result) {
            extern __shared__ short tmp_src[];
            extern __shared__ short tmp_other[];
            unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int block_index = threadIdx.x;
            if (index < size[0]) {
              tmp_src[block_index] = src[index];
              tmp_other[block_index] = other[index];
                if (tmp_src[block_index] < tmp_other[block_index]) {
                    tmp_src[block_index] = 1;
                } else {
                    tmp_src[block_index] = 0;
                }
            }
            __syncthreads();
            result[index] = tmp_src[0];
        }

        __global__ void greater_tensor(unsigned int* size, short* src, short* other, short* result) {
            extern __shared__ short tmp_src[];
            extern __shared__ short tmp_other[];
            unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int block_index = threadIdx.x;
            if (index < size[0]) {
              tmp_src[block_index] = src[index];
              tmp_other[block_index] = other[index];
                if (tmp_src[block_index] > tmp_other[block_index]) {
                    tmp_src[block_index] = 1;
                } else {
                    tmp_src[block_index] = 0;
                }
            }
            __syncthreads();
            result[index] = tmp_src[0];
        }


        __global__ void lesser_equals_tensor(unsigned int* size, short* src, short* other, short* result) {
            extern __shared__ short tmp_src[];
            extern __shared__ short tmp_other[];
            unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int block_index = threadIdx.x;
            if (index < size[0]) {
              tmp_src[block_index] = src[index];
              tmp_other[block_index] = other[index];
                if (tmp_src[block_index] <= tmp_other[block_index]) {
                    tmp_src[block_index] = 1;
                } else {
                    tmp_src[block_index] = 0;
                }
            }
            __syncthreads();
            result[index] = tmp_src[0];
        }

        __global__ void greater_equals_tensor(unsigned int* size, short* src, short* other, short* result) {
            extern __shared__ short tmp_src[];
            extern __shared__ short tmp_other[];
            unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int block_index = threadIdx.x;
            if (index < size[0]) {
              tmp_src[block_index] = src[index];
              tmp_other[block_index] = other[index];
                if (tmp_src[block_index] >= tmp_other[block_index]) {
                    tmp_src[block_index] = 1;
                } else {
                    tmp_src[block_index] = 0;
                }
            }
            __syncthreads();
            result[index] = tmp_src[0];
        }

         __global__ void equals_tensor(unsigned int* size, short* src, short* other, short* result) {
            extern __shared__ short tmp_src[];
            extern __shared__ short tmp_other[];
            unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int block_index = threadIdx.x;
            if (index < size[0]) {
              tmp_src[block_index] = src[index];
              tmp_other[block_index] = other[index];
                if (tmp_src[block_index] == tmp_other[block_index]) {
                    tmp_src[block_index] = 1;
                } else {
                    tmp_src[block_index] = 0;
                }
            }
            __syncthreads();
            result[index] = tmp_src[0];
        }

       bool check_size( Tensor* tensor,  Tensor* other) {
            if (tensor->size[0] != other->size[0]) {
                //printf("Error: size mismatch\n");
                return false;
            }
            if (sizeof(&tensor->data) != sizeof(&other->data)) {
                //printf("Error: size mismatch\n");
                return false;
            }
            return true;
        }

        extern "C" bool check_size_3( Tensor* tensor,  Tensor* other,  Tensor* result) {
            if (tensor->size[0] != other->size[0] || tensor->size[0] != result->size[0]) {
                //printf("Error: size mismatch\n");
                return false;
            }
            if (sizeof(&tensor->data) != sizeof(&other->data) || sizeof(&tensor->data) != sizeof(&result->data)) {
                //printf("Error: size mismatch\n");
                return false;
            }
            return true;
        }

        int get_device_dim(unsigned int size) {
            return ((size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        }

        int get_device_dim_remainder(unsigned int size) {
            return (size - THREADS_PER_BLOCK*(size/THREADS_PER_BLOCK));
        }

        __global__ void vector_resize_tensor(unsigned int* size_small, unsigned int* size_large, short* src, short* vectors, short* result) {
            extern __shared__ short tmp[];
            unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int block_index = threadIdx.x;
            if (index < size_large[0]) {
                result[index] = src[vectors[index]];
            }
        }

        __global__ void tensor_enlarge(unsigned int* size_small, unsigned int* size_large, unsigned int* scale_factor, short* src, short* result) {
            extern __shared__ short tmp[];
            unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int block_index = threadIdx.x;
            if (index < size_large[0]) {
                result[index] = src[int(index/scale_factor[0])];
            }
        }

        __global__ void bilinear_interpolate_tensor_enlarge(unsigned int* size_small, unsigned int* dims_small, unsigned int* size_large, unsigned int* dims_large, double* scale_factor, double* sqrt2, short* src, short* result) {
            extern __shared__ short tmp[];
            unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int block_index = threadIdx.x;
            if (index < size_large[0]) {

            }
        }
        
__global__ void negate_tensor(unsigned int* size, short* data, short* result) {
  extern __shared__ short tmp[];
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int block_index = threadIdx.x;
  if (threadIdx.x < size[0]) {
    if (data[threadIdx.x] == 0) {
      result[threadIdx.x] = 0;
    } else {
      result[threadIdx.x] = 1;
    }
  }
}

extern "C" void negate_tensor_wrapper(Tensor* tensor, Tensor* result) {
    unsigned int* size = (unsigned int*) malloc(sizeof(unsigned int));
    cudaMemcpy(size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size[0] < THREADS_PER_BLOCK) {
        negate_tensor<<<1, size[0], THREADS_PER_BLOCK * sizeof(short)>>>(tensor->size, tensor->data, result->data);
    } else if (size[0] / THREADS_PER_BLOCK < BLOCKS_MAXIMUM) {
        negate_tensor<<<get_device_dim(size[0]), THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(short)>>>(tensor->size, tensor->data, result->data);
    } else {
        negate_tensor<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(short)>>>(tensor->size, tensor->data, result->data);
    }
    cudaDeviceSynchronize();
    free(size);
}

extern "C" void vector_add_wrapper( Tensor* tensor,  Tensor* other,  Tensor* vectors,  Tensor* result) {
    unsigned int* size = (unsigned int*) malloc(sizeof(unsigned int));
    cudaMemcpy(size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size[0] < THREADS_PER_BLOCK) {
        vector_add_tensor<<<1, size[0], 4*THREADS_PER_BLOCK*sizeof(short)>>>(tensor->size, tensor->data, other->data, vectors->data, result->data);
    } else if (size[0] / THREADS_PER_BLOCK < BLOCKS_MAXIMUM) {
        vector_add_tensor<<<get_device_dim(size[0]), THREADS_PER_BLOCK, 4*THREADS_PER_BLOCK*sizeof(short)>>>(tensor->size, tensor->data, other->data, vectors->data, result->data);
    } else {
        vector_add_tensor<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK, 4*THREADS_PER_BLOCK*sizeof(short)>>>(tensor->size, tensor->data, other->data, vectors->data, result->data);
    }
    cudaDeviceSynchronize();
    free(size);
}

extern "C" void vector_sub_wrapper( Tensor* tensor,  Tensor* other,  Tensor* vectors,  Tensor* result) {
    unsigned int size = 0;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size < THREADS_PER_BLOCK) {
        vector_sub_tensor<<<1, size, 4 * THREADS_PER_BLOCK * sizeof(short)>>>(tensor->size, tensor->data, other->data, vectors->data, result->data);
    } else if (size / THREADS_PER_BLOCK < BLOCKS_MAXIMUM) {
        vector_sub_tensor<<<get_device_dim(size), THREADS_PER_BLOCK, 4 * THREADS_PER_BLOCK * sizeof(short)>>>(tensor->size, tensor->data, other->data, vectors->data, result->data);
    } else {
        vector_sub_tensor<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK, 4 * THREADS_PER_BLOCK * sizeof(short)>>>(tensor->size, tensor->data, other->data, vectors->data, result->data);
    }
    cudaDeviceSynchronize();
}

extern "C" void vector_mul_wrapper( Tensor* tensor,  Tensor* other,  Tensor* vectors,  Tensor* result) {
    unsigned int size = 0;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size < THREADS_PER_BLOCK) {
        vector_mul_tensor<<<1, size, 4 * THREADS_PER_BLOCK * sizeof(short)>>>(tensor->size, tensor->data, other->data, vectors->data, result->data);
    } else if (size / THREADS_PER_BLOCK < BLOCKS_MAXIMUM) {
        vector_mul_tensor<<<get_device_dim(size), THREADS_PER_BLOCK, 4 * THREADS_PER_BLOCK * sizeof(short)>>>(tensor->size, tensor->data, other->data, vectors->data, result->data);
    } else {
        vector_mul_tensor<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK, 4 * THREADS_PER_BLOCK * sizeof(short)>>>(tensor->size, tensor->data, other->data, vectors->data, result->data);
    }
    cudaDeviceSynchronize();
}

extern "C" void vector_div_wrapper( Tensor* tensor,  Tensor* other,  Tensor* vectors,  Tensor* result) {
    unsigned int size = 0;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size < THREADS_PER_BLOCK) {
        vector_div_tensor<<<1, size, 4 * THREADS_PER_BLOCK * sizeof(short)>>>(tensor->size, tensor->data, other->data, vectors->data, result->data);
    } else if (size / THREADS_PER_BLOCK < BLOCKS_MAXIMUM) {
        vector_div_tensor<<<get_device_dim(size), THREADS_PER_BLOCK, 4 * THREADS_PER_BLOCK * sizeof(short)>>>(tensor->size, tensor->data, other->data, vectors->data, result->data);
    } else {
        vector_div_tensor<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK, 4 * THREADS_PER_BLOCK * sizeof(short)>>>(tensor->size, tensor->data, other->data, vectors->data, result->data);
    }
    cudaDeviceSynchronize();
}

extern "C" void vector_gate_wrapper( Tensor* tensor,  Tensor* booleans,  Tensor* vectors,  Tensor* result) {
    unsigned int size = 0;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size < THREADS_PER_BLOCK) {
        vector_gate_tensor<<<1, size, 4 * THREADS_PER_BLOCK * sizeof(short)>>>(tensor->size, tensor->data, booleans->data, vectors->data, result->data);
    } else if (size / THREADS_PER_BLOCK < BLOCKS_MAXIMUM) {
        vector_gate_tensor<<<get_device_dim(size), THREADS_PER_BLOCK>>>(tensor->size, tensor->data, booleans->data, vectors->data, result->data);
    } else {
        vector_gate_tensor<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK>>>(tensor->size, tensor->data, booleans->data, vectors->data, result->data);
    }
    cudaDeviceSynchronize();
}

extern "C" void zeros_tensor_wrapper( Tensor* tensor) {
    unsigned int size = 0;
    
    cudaError_t err = cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    //printf("Error: %s\n", cudaGetErrorString(err));
    //printf("%d\n", size);
    if (size < THREADS_PER_BLOCK) {
        zeros_tensor<<<1, size>>>(tensor->size, tensor->data);
        //printf("Ran the zeros tensor kernel!\n");
    } else if (size / THREADS_PER_BLOCK < BLOCKS_MAXIMUM) {
        zeros_tensor<<<get_device_dim(size), THREADS_PER_BLOCK>>>(tensor->size, tensor->data);
    } else {
        zeros_tensor<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK>>>(tensor->size, tensor->data);
    }
    err = cudaDeviceSynchronize();
    //printf("Error: %s\n", cudaGetErrorString(err));
}

extern "C" void ones_tensor_wrapper( Tensor* tensor) {
    unsigned int size = 0;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    //printf("Size: %d\n", size);
    if (size < THREADS_PER_BLOCK) {
        ones_tensor<<<1, size>>>(tensor->size, tensor->data);
    } else if (size / THREADS_PER_BLOCK < BLOCKS_MAXIMUM) {
        ones_tensor<<<get_device_dim(size), THREADS_PER_BLOCK>>>(tensor->size, tensor->data);
    } else {
        ones_tensor<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK>>>(tensor->size, tensor->data);
    }
    cudaDeviceSynchronize();
}

extern "C" void vector_sort_tensor_wrapper( Tensor* tensor,  Tensor* vectors,  Tensor* result)  {
    unsigned int size = 0;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);

   if (size < THREADS_PER_BLOCK) {
        vector_sort_tensor<<<1, size>>>(tensor->size, tensor->data, vectors->data, result->data);
    } else if (size / THREADS_PER_BLOCK < BLOCKS_MAXIMUM) {
        vector_sort_tensor<<<get_device_dim(size), THREADS_PER_BLOCK>>>(tensor->size, tensor->data, vectors->data, result->data);
    } else {
        vector_sort_tensor<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK>>>(tensor->size, tensor->data, vectors->data, result->data);
    }
    cudaDeviceSynchronize();
}

extern "C" void print_tensor_wrapper( Tensor* tensor) {
    unsigned int size = 0;
    cudaError_t err = cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    //printf("Error: %s\n", cudaGetErrorString(err));
    //printf("Tensor size: %d\n", size);
    if (size == 4) {
        print_tensor<<<1, 4>>>(tensor->size, tensor->data);
    }
    err = cudaDeviceSynchronize();
    //printf("Error: %s\n", cudaGetErrorString(err));
    short val = -1;
    err = cudaMemcpy(&val, &tensor->data[0], sizeof(short), cudaMemcpyDeviceToHost);
    //printf("Error: %s\n", cudaGetErrorString(err));
    //printf("Value at index 0: %d\n", val);
}

extern "C" void fill_tensor_wrapper( Tensor* tensor, short value) {
    unsigned int size = 0;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    //printf("Size: %d\n", size);
    short* device_value;
    cudaMalloc((void**) &device_value, sizeof(short));
    cudaMemcpy(device_value, &value, sizeof(short), cudaMemcpyHostToDevice);
    cudaError_t err = cudaDeviceSynchronize();
    //printf("Error: %s\n", cudaGetErrorString(err));
    //printf("Size: %d\n", size);
    //printf("Value: %d\n", value);
    if (size < THREADS_PER_BLOCK) {
        fill_tensor<<<1, size>>>(tensor->size, tensor->data, device_value);
        //printf("Size is less than THREADS_PER_BLOCK.\n");
    } else if (size / THREADS_PER_BLOCK < BLOCKS_MAXIMUM) {
        fill_tensor<<<get_device_dim(size), THREADS_PER_BLOCK>>>(tensor->size, tensor->data, device_value);
    } else {
        fill_tensor<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK>>>(tensor->size, tensor->data, device_value);
    }
    err = cudaDeviceSynchronize();
    //printf("Error: %s\n", cudaGetErrorString(err));
}

extern "C" void set_tensor_wrapper( Tensor* tensor, unsigned int index, short value) {
    //printf("Retrieving size...");
    unsigned int size = 0;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    //printf("Setting device index...");
    unsigned int* device_index;
    cudaMalloc((void**) &device_index, sizeof(unsigned int));
    cudaMemcpy(device_index, &index, sizeof(unsigned int), cudaMemcpyHostToDevice);
    //printf("Setting device value...");
    short* device_value;
    cudaMalloc((void**) &device_value, sizeof(short));
    cudaMemcpy(device_value, &value, sizeof(short), cudaMemcpyHostToDevice);
    
    //printf("Calling device function...");
    if (index < size) {
      set_tensor<<<1, 1>>>(tensor->size, tensor->data, device_index, device_value);
    }

    //printf("Calling cuda synchronize...");
    cudaDeviceSynchronize();
    //printf("Freeing unused device memory...");
    cudaFree(device_index);
    cudaFree(device_value);
}

extern "C" void get_tensor_value_wrapper( Tensor* tensor, unsigned int index, short* value) {


    //cudaError_t err = cudaGetLastError();
    //printf("Get_tensor_value_wrapper function...\n");
    //printf("Getting size...\n");
    unsigned int size = 0;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    //err = cudaGetLastError();
    //printf("Error: %s\n", cudaGetErrorString(err));
    //printf("Setting device index...\n");
    unsigned int* device_index = NULL;
    cudaMalloc((void**) &device_index, sizeof(unsigned int));
    //err = cudaGetLastError();
    //printf("Error: %s\n", cudaGetErrorString(err));
    cudaMemcpy(device_index, &index, sizeof(unsigned int), cudaMemcpyHostToDevice);
    //err = cudaGetLastError();
    //printf("Error: %s\n", cudaGetErrorString(err));
    //printf("Setting device_value...\n");
    short* device_value = NULL;
    cudaMalloc((void**) &device_value, sizeof(short));
    //err = cudaGetLastError();
    //printf("Error: %s\n", cudaGetErrorString(err));
    if (index < size) {
        get_tensor_value<<<1, 1>>>(tensor->size, tensor->data, device_index, device_value);
    }
    //err = cudaGetLastError();
    //printf("Error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
    //err = cudaGetLastError();
    //printf("Error: %s\n", cudaGetErrorString(err));
    //printf("Setting host value...\n");
    //short host_value = 0;
    cudaMemcpy(value, device_value, sizeof(short), cudaMemcpyDeviceToHost);
    printf("Value at index %d: %d\n", index, value[0]);
    err = cudaGetLastError();
    printf("Error: %s\n", cudaGetErrorString(err));
    //printf("Freeing pointers...\n");
    cudaFree(device_value);
    cudaFree(device_index);
    //err = cudaGetLastError();
    //printf("Error: %s\n", cudaGetErrorString(err));
    //printf("Returning value: %d\n", host_value);
    //value[0] = host_value;
    //err = cudaGetLastError();
    //printf("Error: %s\n", cudaGetErrorString(err));
}

extern "C" void add_tensor_wrapper( Tensor* tensor,  Tensor* other,  Tensor* result) {
    unsigned int size = 0;
    
    cudaError_t err = cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    //printf("Error: %s\n", cudaGetErrorString(err));
    //printf("Size: %d\n", size);
    
    //if (size == 4) { 
        if (size < THREADS_PER_BLOCK) {
            add_tensor<<<1, size>>>(tensor->size, tensor->data, other->data, result->data);
            //printf("adding choice 1\n");
        } else if (size / THREADS_PER_BLOCK < BLOCKS_MAXIMUM) {
            add_tensor<<<get_device_dim(size), THREADS_PER_BLOCK>>>(tensor->size, tensor->data, other->data, result->data);
            //printf("adding choice 2\n");
        } else {
            add_tensor<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK>>>(tensor->size, tensor->data, other->data, result->data);
            //printf("adding choice 3\n");
        }
    //}

    err = cudaDeviceSynchronize();
    //printf("Error: %s\n", cudaGetErrorString(err));
}

extern "C" void sub_tensor_wrapper( Tensor* tensor,  Tensor* other,  Tensor* result) {
    unsigned int size = 0;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size / THREADS_PER_BLOCK == 0) {
          sub_tensor<<<1, size>>>(tensor->size, tensor->data, other->data, result->data);
     } else if (size / THREADS_PER_BLOCK < BLOCKS_MAXIMUM) {
          sub_tensor<<<get_device_dim(size), THREADS_PER_BLOCK>>>(tensor->size, tensor->data, other->data, result->data);
     } else {
          sub_tensor<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK>>>(tensor->size, tensor->data, other->data, result->data);
     }
    cudaDeviceSynchronize();
}

extern "C" void mul_tensor_wrapper( Tensor* tensor,  Tensor* other,  Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size / THREADS_PER_BLOCK == 0) {
          mul_tensor<<<1, size>>>(tensor->size, tensor->data, other->data, result->data);
     } else if (size / THREADS_PER_BLOCK < BLOCKS_MAXIMUM) {
          mul_tensor<<<get_device_dim(size), THREADS_PER_BLOCK>>>(tensor->size, tensor->data, other->data, result->data);
     } else {
          mul_tensor<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK>>>(tensor->size, tensor->data, other->data, result->data);
     }
    cudaDeviceSynchronize();
}

extern "C" void div_tensor_wrapper( Tensor* tensor,  Tensor* other,  Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size / THREADS_PER_BLOCK == 0) {
          div_tensor<<<1, size>>>(tensor->size, tensor->data, other->data, result->data);
     } else if (size / THREADS_PER_BLOCK < 8) {
          div_tensor<<<(size / THREADS_PER_BLOCK), 256>>>(tensor->size, tensor->data, other->data, result->data);
     } else {
          div_tensor<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK>>>(tensor->size, tensor->data, other->data, result->data);
     }
    cudaDeviceSynchronize();
}

extern "C" void add_scalar_tensor_wrapper(Tensor* tensor, short value, Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    short* device_value = NULL;
    cudaMalloc((void**) device_value, sizeof(short));
    cudaMemcpy(&device_value, &value, sizeof(short), cudaMemcpyHostToDevice);
    if (size / THREADS_PER_BLOCK == 0) {
        add_scalar_tensor<<<1, size>>>(tensor->size, tensor->data, device_value, result->data);
    } else if (size / THREADS_PER_BLOCK < BLOCKS_MAXIMUM) {
        add_scalar_tensor<<<get_device_dim(size), THREADS_PER_BLOCK>>>(tensor->size, tensor->data, device_value, result->data);
    } else {
        add_scalar_tensor<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK>>>(tensor->size, tensor->data, device_value, result->data);
    }
    cudaDeviceSynchronize();
}

extern "C" void sub_scalar_tensor_wrapper(Tensor* tensor, short value, Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    short* device_value = NULL;
    cudaMalloc((void**) device_value, sizeof(short));
    cudaMemcpy(&device_value, &value, sizeof(short), cudaMemcpyHostToDevice);
    if (size / THREADS_PER_BLOCK == 0) {
        sub_scalar_tensor<<<1, size>>>(tensor->size, tensor->data, device_value, result->data);
    } else if (size / THREADS_PER_BLOCK < BLOCKS_MAXIMUM) {
        sub_scalar_tensor<<<get_device_dim(size), THREADS_PER_BLOCK>>>(tensor->size, tensor->data, device_value, result->data);
    } else {
        sub_scalar_tensor<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK>>>(tensor->size, tensor->data, device_value, result->data);
    }
    cudaDeviceSynchronize();
}

extern "C" void mul_scalar_tensor_wrapper(Tensor* tensor, short value, Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    short* device_value = NULL;
    cudaMalloc((void**) device_value, sizeof(short));
    cudaMemcpy(&device_value, &value, sizeof(short), cudaMemcpyHostToDevice);
    if (size / THREADS_PER_BLOCK == 0) {
        mul_scalar_tensor<<<1, size>>>(tensor->size, tensor->data, device_value, result->data);
    } else if (size / THREADS_PER_BLOCK < BLOCKS_MAXIMUM) {
        mul_scalar_tensor<<<get_device_dim(size), THREADS_PER_BLOCK>>>(tensor->size, tensor->data, device_value, result->data);
    } else {
        mul_scalar_tensor<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK>>>(tensor->size, tensor->data, device_value, result->data);
    }
    cudaDeviceSynchronize();
}

extern "C" void div_scalar_tensor_wrapper(Tensor* tensor, short scalar, Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    short* device_value = NULL;
    cudaMalloc((void**) device_value, sizeof(short));
    cudaMemcpy(&device_value, &scalar, sizeof(short), cudaMemcpyHostToDevice);
    if (size / THREADS_PER_BLOCK == 0) {
        div_scalar_tensor<<<1, size>>>(tensor->size, tensor->data, device_value, result->data);
    } else if (size / THREADS_PER_BLOCK < BLOCKS_MAXIMUM) {
        div_scalar_tensor<<<get_device_dim(size), THREADS_PER_BLOCK>>>(tensor->size, tensor->data, device_value, result->data);
    } else {
        div_scalar_tensor<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK>>>(tensor->size, tensor->data, device_value, result->data);
    }
    cudaDeviceSynchronize();
}

extern "C" void transpose_tensor_wrapper(Tensor* tensor, Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size / THREADS_PER_BLOCK == 0) {
        transpose_tensor<<<1, size>>>(tensor->size, tensor->data, result->data);
    } else if (size / THREADS_PER_BLOCK < BLOCKS_MAXIMUM) {
        transpose_tensor<<<get_device_dim(size), THREADS_PER_BLOCK>>>(tensor->size, tensor->data, result->data);
    } else {
        transpose_tensor<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK>>>(tensor->size, tensor->data, result->data);
    }
    cudaDeviceSynchronize();
}

extern "C" void sum_tensor_wrapper(Tensor* tensor, Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size / THREADS_PER_BLOCK == 0) {
        sum_tensor<<<1, size>>>(tensor->size, tensor->data, result->data);
    } else if (size / THREADS_PER_BLOCK < BLOCKS_MAXIMUM) {
        sum_tensor<<<get_device_dim(size), THREADS_PER_BLOCK>>>(tensor->size, tensor->data, result->data);
    } else {
        sum_tensor<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK>>>(tensor->size, tensor->data, result->data);
    }
    cudaDeviceSynchronize();
}

extern "C" void mean_tensor_wrapper(Tensor* tensor, Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size / THREADS_PER_BLOCK == 0) {
        mean_tensor<<<1, size>>>(tensor->size, tensor->data, result->data);
    } else if (size / THREADS_PER_BLOCK < BLOCKS_MAXIMUM) {
        mean_tensor<<<get_device_dim(size), THREADS_PER_BLOCK>>>(tensor->size, tensor->data, result->data);
    } else {
        mean_tensor<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK>>>(tensor->size, tensor->data, result->data);
    }
    cudaDeviceSynchronize();
}

extern "C" void max_tensor_wrapper(Tensor* tensor, Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size / THREADS_PER_BLOCK == 0) {
        max_tensor<<<1, size>>>(tensor->size, tensor->data, result->data);
    } else if (size / THREADS_PER_BLOCK < BLOCKS_MAXIMUM) {
        max_tensor<<<get_device_dim(size), THREADS_PER_BLOCK>>>(tensor->size, tensor->data, result->data);
    } else {
        max_tensor<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK>>>(tensor->size, tensor->data, result->data);
    }
}

extern "C" void min_tensor_wrapper(Tensor* tensor, Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size / THREADS_PER_BLOCK == 0) {
        min_tensor<<<1, size>>>(tensor->size, tensor->data, result->data);
    } else if (size / THREADS_PER_BLOCK < BLOCKS_MAXIMUM) {
        min_tensor<<<get_device_dim(size), THREADS_PER_BLOCK>>>(tensor->size, tensor->data, result->data);
    } else {
        min_tensor<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK>>>(tensor->size, tensor->data, result->data);
    }
}

extern "C" void gradient_tensor_wrapper(Tensor* tensor, Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size / THREADS_PER_BLOCK == 0) {
        gradient_tensor<<<1, size>>>(tensor->size, tensor->data, result->data);
    } else if (size / THREADS_PER_BLOCK < BLOCKS_MAXIMUM) {
        gradient_tensor<<<get_device_dim(size), THREADS_PER_BLOCK>>>(tensor->size, tensor->data, result->data);
    } else {
        gradient_tensor<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK>>>(tensor->size, tensor->data, result->data);
    }
}

extern "C" void gate_tensor_wrapper(Tensor* tensor, Tensor* booleans, Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size / THREADS_PER_BLOCK == 0) {
        gate_tensor<<<1, size>>>(tensor->size, tensor->data, booleans->data, result->data);
    } else if (size / THREADS_PER_BLOCK < BLOCKS_MAXIMUM) {
        gate_tensor<<<get_device_dim(size), THREADS_PER_BLOCK>>>(tensor->size, tensor->data, booleans->data, result->data);
    } else {
        gate_tensor<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK>>>(tensor->size, tensor->data, booleans->data, result->data);
    }
}

extern "C" void vector_resize_tensor_wrapper(Tensor* tensor, Tensor* vectors, Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size / THREADS_PER_BLOCK == 0) {
        vector_resize_tensor<<<1, size>>>(tensor->size, result->size, tensor->data, vectors->data, result->data);
    } else if (size / THREADS_PER_BLOCK < BLOCKS_MAXIMUM) {
        vector_resize_tensor<<<get_device_dim(size), THREADS_PER_BLOCK>>>(tensor->size, result->size, tensor->data, vectors->data, result->data);
    } else {
        vector_resize_tensor<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK>>>(tensor->size, result->size, tensor->data, vectors->data, result->data);
    }
}

extern "C" void tensor_enlarge_wrapper(Tensor* tensor, unsigned int scale_factor, Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* device_scale = NULL;
    cudaMalloc((void**) device_scale, sizeof(unsigned int));
    cudaMemcpy(device_scale, &scale_factor, sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (size / THREADS_PER_BLOCK == 0) {
        tensor_enlarge<<<1, size>>>(tensor->size, result->size, device_scale, tensor->data, result->data);
    } else if (size / THREADS_PER_BLOCK < BLOCKS_MAXIMUM) {
        tensor_enlarge<<<get_device_dim(size), THREADS_PER_BLOCK>>>(tensor->size, result->size, device_scale, tensor->data, result->data);
    } else {
        tensor_enlarge<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK>>>(tensor->size, result->size, device_scale, tensor->data, result->data);
    }
}

extern "C" void lesser_tensor_wrapper(Tensor* tensor, Tensor* other, Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size / THREADS_PER_BLOCK == 0) {
        lesser_tensor<<<1, size>>>(tensor->size, tensor->data, other->data, result->data);
    } else if (size / THREADS_PER_BLOCK < BLOCKS_MAXIMUM) {
        lesser_tensor<<<get_device_dim(size), THREADS_PER_BLOCK>>>(tensor->size, tensor->data, other->data, result->data);
    } else {
        lesser_tensor<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK>>>(tensor->size, tensor->data, other->data, result->data);
    }
}

extern "C" void greater_tensor_wrapper(Tensor* tensor, Tensor* other, Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size / THREADS_PER_BLOCK == 0) {
        greater_tensor<<<1, size>>>(tensor->size, tensor->data, other->data, result->data);
    } else if (size / THREADS_PER_BLOCK < BLOCKS_MAXIMUM) {
        greater_tensor<<<get_device_dim(size), THREADS_PER_BLOCK>>>(tensor->size, tensor->data, other->data, result->data);
    } else {
        greater_tensor<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK>>>(tensor->size, tensor->data, other->data, result->data);
    }
}

extern "C" void lesser_equals_tensor_wrapper(Tensor* tensor, Tensor* other, Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size / THREADS_PER_BLOCK == 0) {
        lesser_equals_tensor<<<1, size>>>(tensor->size, tensor->data, other->data, result->data);
    } else if (size / THREADS_PER_BLOCK < BLOCKS_MAXIMUM) {
        lesser_equals_tensor<<<get_device_dim(size), THREADS_PER_BLOCK>>>(tensor->size, tensor->data, other->data, result->data);
    } else {
        lesser_equals_tensor<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK>>>(tensor->size, tensor->data, other->data, result->data);
    }
}

extern "C" void greater_equals_tensor_wrapper(Tensor* tensor, Tensor* other, Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size / THREADS_PER_BLOCK == 0) {
        greater_equals_tensor<<<1, size>>>(tensor->size, tensor->data, other->data, result->data);
    } else if (size / THREADS_PER_BLOCK < BLOCKS_MAXIMUM) {
        greater_equals_tensor<<<get_device_dim(size), THREADS_PER_BLOCK>>>(tensor->size, tensor->data, other->data, result->data);
    } else {
        greater_equals_tensor<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK>>>(tensor->size, tensor->data, other->data, result->data);
    }
}

extern "C" void equals_tensor_wrapper(Tensor* tensor, Tensor* other, Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size / THREADS_PER_BLOCK == 0) {
        equals_tensor<<<1, size>>>(tensor->size, tensor->data, other->data, result->data);
    } else if (size / THREADS_PER_BLOCK < BLOCKS_MAXIMUM) {
        equals_tensor<<<get_device_dim(size), THREADS_PER_BLOCK>>>(tensor->size, tensor->data, other->data, result->data);
    } else {
        equals_tensor<<<BLOCKS_MAXIMUM, THREADS_PER_BLOCK>>>(tensor->size, tensor->data, other->data, result->data);
    }
}
