#include "cuda.h"
#include "cuda_runtime.h"
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
    #include "tensor.h"
#endif
#ifdef __DEBUG__
#endif

void set_cpu_to_device_tensor(Tensor* tensor);
void set_device_to_cpu_tensor(Tensor* tensor);
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

extern "C" Tensor* create_device_tensor(unsigned int size) {
    printf("step 1\n");
    unsigned int temp = size;
    Tensor *tensor = (Tensor*) malloc(sizeof(Tensor*));
    printf("step 2\n");
    printf("%d\n", size);
    cudaError_t err = cudaMalloc((void **) &tensor->size, sizeof(unsigned int));
    printf("%s\n", cudaGetErrorString(err));
    printf("step 3\n");
    err = cudaMemcpy(tensor->size, &temp, sizeof(unsigned int), cudaMemcpyHostToDevice);
    printf("%s\n", cudaGetErrorString(err));
    printf("step 4\n");
    err = cudaMalloc((void **) &tensor->data, sizeof(short) * size);
    printf("%s\n", cudaGetErrorString(err));
    printf("step 5\n");
    err = cudaDeviceSynchronize();
    printf("%s\n", cudaGetErrorString(err));
    
    return tensor;
}

extern "C" void destroy_device_tensor(Tensor* tensor) {
    cudaFree(tensor->data);
    cudaFree(tensor->size);
    free(tensor);
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
    unsigned int index = threadIdx.x;
    if (index < size[0]) {
        result[index] = src[index] + other[vectors[index]];
    }
}

__global__
void vector_sub_tensor(unsigned int* size, short* src, short* other, short* vectors, short* result) {
    unsigned int index = threadIdx.x;
    if (index < size[0]) {
        result[index] = src[index] + other[vectors[index]];
    }
}

__global__
void vector_mul_tensor(unsigned int* size, short* src, short* other, short* vectors, short* result) {
    unsigned int index = threadIdx.x;
    if (index < size[0]) {
        result[index] = src[index] + other[vectors[index]];
    }
}

__global__
void vector_div_tensor(unsigned int* size, short* src, short* other, short* vectors, short* result) {
    unsigned int index = threadIdx.x;
    if (index < size[0]) {
        result[index] = src[index] + other[vectors[index]];
    }
}

__global__
void vector_gate_tensor(unsigned int* size, short* src, short* booleans, short* vectors, short* result) {
    unsigned int i = threadIdx.x;
    if (i < size[0]) {
        if (booleans[i] == 1) {
            result[i] = src[vectors[i]];
        }
    }
}

__global__
void set_tensor(unsigned int* size, short* src, unsigned int* index, short* value) {

    if (index[0] > size[0]) {
        printf("Error: index out of bounds\n");
        return;
    }
    src[index[0]] = value[0];
}       

        
        
        __global__ void zeros_tensor( unsigned int* size, short* data) {
            unsigned int index = threadIdx.x;
            short value = 0;
            printf("Printing from within zeros tensor kernel! Index: %d\n", index);
            if (index < size[0]) {
                data[index] = value;
                printf("setting to %d\n", value);
            }
        }

        __global__
        void ones_tensor( unsigned int* size, short* data) {
            unsigned int index = threadIdx.x;
            short value = 1;
            printf("printing from within ones tensor kernel!\n");
            if (index < size[0]) {
                data[index] = value;
                printf("setting to %d\n", value);
            }
        }

        __global__
        void print_tensor( unsigned int* size, short* data) {

            unsigned int index = threadIdx.x;
            if (index < size[0]) {
                printf("index: %d, value: %d\n", index, data[index]);
            }
        }
            

        __global__
        void fill_tensor( unsigned int* size, short* data, short* value) {
            unsigned int index = threadIdx.x;
            if (index < size[0]) {
                data[index] = value[0];
                printf("setting to %d\n", value[0]);
            }
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
            unsigned int index = threadIdx.x;
            if (index < size[0]) {
                data3[index] = data1[index] + data2[index];
            }
        }

        __global__
        void sub_tensor( unsigned int* size, short* data1, short* data2, short* data3) {
            unsigned int index = threadIdx.x;
            if (index < size[0]) {
                data3[index] = data1[index] - data2[index];
            }
        }

        __global__
        void mul_tensor( unsigned int* size, short* data1, short* data2, short* data3) {
            unsigned int index = threadIdx.x;
            if (index < size[0]) {
                data3[index] = data1[index] * data2[index];
            }
        }

        __global__
        void div_tensor( unsigned int* size, short* data1, short* data2, short* data3) {
            unsigned int index = threadIdx.x;
            if (index < size[0]) {
                data3[index] = data1[index] / data2[index];
            }
        }

        __global__
        void add_scalar_tensor(unsigned int* size, short* src, short* value,  short* result) {
            unsigned int index = threadIdx.x;
            if (index < size[0]) {
                result[index] = src[index] + value[0];
            }
        }

        __global__
        void sub_scalar_tensor(unsigned int* size, short* src, short* value, short* result) {
            unsigned int index = threadIdx.x;
            if (index < size[0]) {
                result[index] = src[index] - value[0];
            }
        }

        __global__
        void mul_scalar_tensor(unsigned int* size, short* src, short* value, short* result) {
            unsigned int index = threadIdx.x;
            if (index < size[0]) {
                result[index] = src[index] * value[0];
            }
        }

        __global__
        void div_scalar_tensor(unsigned int* size, short* src, short* value, short* result) {
            unsigned int index = threadIdx.x;
            if (index < size[0]) {
                result[index] = src[index] / value[0];
            }
        }

        __global__
        void transpose_tensor(unsigned int* size, short* src, short* result) {
            unsigned int index = threadIdx.x;
            if (index < size[0]) {
                result[index] = src[size[0] - index - 1];
            }
        }

        __global__
        void sum_tensor( unsigned int* size, short* src, short* result) {
            unsigned int index = threadIdx.x;
            if (index < size[0]) {
                result[0] += src[index];
            }
        }

        __global__
        void mean_tensor( unsigned int* size, short* src, short* result) {
            unsigned int index = threadIdx.x;
            if (index < size[0]) {
                result[0] += src[index];
            }
            result[0] /= size[0];
        }

        __global__
        void max_tensor(unsigned int* size, short* src, short* result) {
            unsigned int index = threadIdx.x;
            if (index < size[0]) {
                if (src[index] > result[0]) {
                    result[0] = src[index];
                }
            }
        }

        __global__
        void min_tensor(unsigned int* size, short* src, short* result) {
            unsigned int index = threadIdx.x;
            if (index < size[0]) {
                if (src[index] < result[0]) {
                    result[0] = src[index];
                }
            }
        }

        __global__
        void gradient_tensor(unsigned int* size, short* src, short* result) {
            unsigned int index = threadIdx.x;
            if (index < size[0]) {
                result[index] = src[index + 1] - src[index];
            }
        }

        __global__
        void gate_tensor(unsigned int* size, short* src, short* bools, short* result) {
            unsigned int index = threadIdx.x;
            if (index < size[0]) {
                if (bools[index] == 1) {
                    result[index] = src[index];
                }
            }
        }

        bool check_size( Tensor* tensor,  Tensor* other) {
            if (tensor->size[0] != other->size[0]) {
                printf("Error: size mismatch\n");
                return false;
            }
            if (sizeof(&tensor->data) != sizeof(&other->data)) {
                printf("Error: size mismatch\n");
                return false;
            }
            return true;
        }

        extern "C" bool check_size_3( Tensor* tensor,  Tensor* other,  Tensor* result) {
            if (tensor->size[0] != other->size[0] || tensor->size[0] != result->size[0]) {
                printf("Error: size mismatch\n");
                return false;
            }
            if (sizeof(&tensor->data) != sizeof(&other->data) || sizeof(&tensor->data) != sizeof(&result->data)) {
                printf("Error: size mismatch\n");
                return false;
            }
            return true;
        }

        int get_device_dim(unsigned int size) {
            return (1 +(size/256));
        }

        int get_device_dim_remainder(unsigned int size) {
            return (size - 256*(size/256));
        }
        

extern "C" void vector_add_wrapper( Tensor* tensor,  Tensor* other,  Tensor* vectors,  Tensor* result) {
    unsigned int* size = (unsigned int*) malloc(sizeof(unsigned int));
    cudaMemcpy(size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size[0] < 256 == 0) {
        vector_add_tensor<<<1, size[0]>>>(tensor->size, tensor->data, other->data, vectors->data, result->data);
    } else if (size[0] / 256 < 7) {
        vector_add_tensor<<<get_device_dim(size[0]), 256>>>(tensor->size, tensor->data, other->data, vectors->data, result->data);
    } else {
        vector_add_tensor<<<7, 256>>>(tensor->size, tensor->data, other->data, vectors->data, result->data);
    }
    cudaDeviceSynchronize();
}

extern "C" void vector_sub_wrapper( Tensor* tensor,  Tensor* other,  Tensor* vectors,  Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size < 256 == 0) {
        vector_sub_tensor<<<1, size>>>(tensor->size, tensor->data, other->data, vectors->data, result->data);
    } else if (size / 256 < 7) {
        vector_sub_tensor<<<get_device_dim(size), 256>>>(tensor->size, tensor->data, other->data, vectors->data, result->data);
    } else {
        vector_sub_tensor<<<7, 256>>>(tensor->size, tensor->data, other->data, vectors->data, result->data);
    }
    cudaDeviceSynchronize();
}

extern "C" void vector_mul_wrapper( Tensor* tensor,  Tensor* other,  Tensor* vectors,  Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size < 256 == 0) {
        vector_mul_tensor<<<1, size>>>(tensor->size, tensor->data, other->data, vectors->data, result->data);
    } else if (size / 256 < 7) {
        vector_mul_tensor<<<get_device_dim(size), 256>>>(tensor->size, tensor->data, other->data, vectors->data, result->data);
    } else {
        vector_mul_tensor<<<7, 256>>>(tensor->size, tensor->data, other->data, vectors->data, result->data);
    }
    cudaDeviceSynchronize();
}

extern "C" void vector_div_wrapper( Tensor* tensor,  Tensor* other,  Tensor* vectors,  Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size < 256 == 0) {
        vector_div_tensor<<<1, size>>>(tensor->size, tensor->data, other->data, vectors->data, result->data);
    } else if (size / 256 < 7) {
        vector_div_tensor<<<get_device_dim(size), 256>>>(tensor->size, tensor->data, other->data, vectors->data, result->data);
    } else {
        vector_div_tensor<<<7, 256>>>(tensor->size, tensor->data, other->data, vectors->data, result->data);
    }
    cudaDeviceSynchronize();
}

extern "C" void vector_gate_wrapper( Tensor* tensor,  Tensor* booleans,  Tensor* vectors,  Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size < 256 == 0) {
        vector_gate_tensor<<<1, size>>>(tensor->size, tensor->data, booleans->data, vectors->data, result->data);
    } else if (size / 256 < 7) {
        vector_gate_tensor<<<get_device_dim(size), 256>>>(tensor->size, tensor->data, booleans->data, vectors->data, result->data);
    } else {
        vector_gate_tensor<<<7, 256>>>(tensor->size, tensor->data, booleans->data, vectors->data, result->data);
    }
    cudaDeviceSynchronize();
}

extern "C" void zeros_tensor_wrapper( Tensor* tensor) {
    unsigned int size = 0;
    
    cudaError_t err = cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("Error: %s\n", cudaGetErrorString(err));
    printf("%d\n", size);
    if (size < 256) {
        zeros_tensor<<<1, size>>>(tensor->size, tensor->data);
        printf("Ran the zeros tensor kernel!\n");
    } else if (size / 256 < 7) {
        zeros_tensor<<<get_device_dim(size), 256>>>(tensor->size, tensor->data);
    } else {
        zeros_tensor<<<7, 256>>>(tensor->size, tensor->data);
    }
    err = cudaDeviceSynchronize();
    printf("Error: %s\n", cudaGetErrorString(err));
}

extern "C" void ones_tensor_wrapper( Tensor* tensor) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("Size: %d\n", size);
    if (size < 256) {
        ones_tensor<<<1, size>>>(tensor->size, tensor->data);
    } else if (size / 256 < 7) {
        ones_tensor<<<get_device_dim(size), 256>>>(tensor->size, tensor->data);
    } else {
        ones_tensor<<<7, 256>>>(tensor->size, tensor->data);
    }
    cudaDeviceSynchronize();
}

extern "C" void vector_sort_tensor_wrapper( Tensor* tensor,  Tensor* vectors,  Tensor* result)  {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);

   if (size < 256) {
        vector_sort_tensor<<<1, size>>>(tensor->size, tensor->data, vectors->data, result->data);
    } else if (size / 256 < 7) {
        vector_sort_tensor<<<get_device_dim(size), 256>>>(tensor->size, tensor->data, vectors->data, result->data);
    } else {
        vector_sort_tensor<<<7, 256>>>(tensor->size, tensor->data, vectors->data, result->data);
    }
    cudaDeviceSynchronize();
}

extern "C" void print_tensor_wrapper( Tensor* tensor) {
    unsigned int size = 0;
    cudaError_t err = cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("Error: %s\n", cudaGetErrorString(err));
    printf("Tensor size: %d\n", size);
    if (size == 4) {
        print_tensor<<<1, 4>>>(tensor->size, tensor->data);
    }
    err = cudaDeviceSynchronize();
    printf("Error: %s\n", cudaGetErrorString(err));
    short val = -1;
    err = cudaMemcpy(&val, &tensor->data[0], sizeof(short), cudaMemcpyDeviceToHost);
    printf("Error: %s\n", cudaGetErrorString(err));
    printf("Value at index 0: %d\n", val);
}

extern "C" void fill_tensor_wrapper( Tensor* tensor, short value) {
    unsigned int size = 0;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("Size: %d\n", size);
    short* device_value = NULL;
    cudaMalloc((void**) &device_value, sizeof(short));
    cudaMemcpy(device_value, &value, sizeof(short), cudaMemcpyHostToDevice);
    cudaError_t err = cudaDeviceSynchronize();
    printf("Error: %s\n", cudaGetErrorString(err));
    printf("Size: %d\n", size);
    printf("Value: %d\n", value);
    if (size < 256) {
        fill_tensor<<<1, size>>>(tensor->size, tensor->data, device_value);
        printf("Size is less than 256.\n");
    } else if (size / 256 < 7) {
        fill_tensor<<<get_device_dim(size), 256>>>(tensor->size, tensor->data, device_value);
    } else {
        fill_tensor<<<7, 256>>>(tensor->size, tensor->data, device_value);
    }
    err = cudaDeviceSynchronize();
    printf("Error: %s\n", cudaGetErrorString(err));
}

extern "C" void add_tensor_wrapper( Tensor* tensor,  Tensor* other,  Tensor* result) {
    unsigned int size = 0;
    
    cudaError_t err = cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("Error: %s\n", cudaGetErrorString(err));
    printf("Size: %d\n", size);
    
    //if (size == 4) { 
        if (size < 256) {
            add_tensor<<<1, size>>>(tensor->size, tensor->data, other->data, result->data);
            printf("adding choice 1\n");
        } else if (size / 256 < 7) {
            add_tensor<<<get_device_dim(size), 256>>>(tensor->size, tensor->data, other->data, result->data);
            printf("adding choice 2\n");
        } else {
            add_tensor<<<7, 256>>>(tensor->size, tensor->data, other->data, result->data);
            printf("adding choice 3\n");
        }
    //}

    err = cudaDeviceSynchronize();
    printf("Error: %s\n", cudaGetErrorString(err));
}

extern "C" void sub_tensor_wrapper( Tensor* tensor,  Tensor* other,  Tensor* result) {
    unsigned int size = 0;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size / 256 == 0) {
          sub_tensor<<<1, 256>>>(tensor->size, tensor->data, other->data, result->data);
     } else if (size / 256 < 7) {
          sub_tensor<<<get_device_dim(size), 256>>>(tensor->size, tensor->data, other->data, result->data);
     } else {
          sub_tensor<<<7, 256>>>(tensor->size, tensor->data, other->data, result->data);
     }
    cudaDeviceSynchronize();
}

extern "C" void mul_tensor_wrapper( Tensor* tensor,  Tensor* other,  Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size / 256 == 0) {
          mul_tensor<<<1, 256>>>(tensor->size, tensor->data, other->data, result->data);
     } else if (size / 256 < 7) {
          mul_tensor<<<get_device_dim(size), 256>>>(tensor->size, tensor->data, other->data, result->data);
     } else {
          mul_tensor<<<7, 256>>>(tensor->size, tensor->data, other->data, result->data);
     }
    cudaDeviceSynchronize();
}

extern "C" void div_tensor_wrapper( Tensor* tensor,  Tensor* other,  Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size / 256 == 0) {
          div_tensor<<<1, 256>>>(tensor->size, tensor->data, other->data, result->data);
     } else if (size / 256 < 8) {
          div_tensor<<<(size / 256), 256>>>(tensor->size, tensor->data, other->data, result->data);
     } else {
          div_tensor<<<7, 256>>>(tensor->size, tensor->data, other->data, result->data);
     }
    cudaDeviceSynchronize();
}

extern "C" void add_scalar_tensor_wrapper(Tensor* tensor, short value, Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    short* device_value = NULL;
    cudaMalloc((void**) device_value, sizeof(short));
    cudaMemcpy(&device_value, &value, sizeof(short), cudaMemcpyHostToDevice);
    if (size / 256 == 0) {
        add_scalar_tensor<<<1, 256>>>(tensor->size, tensor->data, device_value, result->data);
    } else if (size / 256 < 7) {
        add_scalar_tensor<<<get_device_dim(size), 256>>>(tensor->size, tensor->data, device_value, result->data);
    } else {
        add_scalar_tensor<<<7, 256>>>(tensor->size, tensor->data, device_value, result->data);
    }
    cudaDeviceSynchronize();
}

extern "C" void sub_scalar_tensor_wrapper(Tensor* tensor, short value, Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    short* device_value = NULL;
    cudaMalloc((void**) device_value, sizeof(short));
    cudaMemcpy(&device_value, &value, sizeof(short), cudaMemcpyHostToDevice);
    if (size / 256 == 0) {
        sub_scalar_tensor<<<1, 256>>>(tensor->size, tensor->data, device_value, result->data);
    } else if (size / 256 < 7) {
        sub_scalar_tensor<<<get_device_dim(size), 256>>>(tensor->size, tensor->data, device_value, result->data);
    } else {
        sub_scalar_tensor<<<7, 256>>>(tensor->size, tensor->data, device_value, result->data);
    }
    cudaDeviceSynchronize();
}

extern "C" void mul_scalar_tensor_wrapper(Tensor* tensor, short value, Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    short* device_value = NULL;
    cudaMalloc((void**) device_value, sizeof(short));
    cudaMemcpy(&device_value, &value, sizeof(short), cudaMemcpyHostToDevice);
    if (size / 256 == 0) {
        mul_scalar_tensor<<<1, 256>>>(tensor->size, tensor->data, device_value, result->data);
    } else if (size / 256 < 7) {
        mul_scalar_tensor<<<get_device_dim(size), 256>>>(tensor->size, tensor->data, device_value, result->data);
    } else {
        mul_scalar_tensor<<<7, 256>>>(tensor->size, tensor->data, device_value, result->data);
    }
    cudaDeviceSynchronize();
}

extern "C" void div_scalar_tensor_wrapper(Tensor* tensor, short scalar, Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    short* device_value = NULL;
    cudaMalloc((void**) device_value, sizeof(short));
    cudaMemcpy(&device_value, &scalar, sizeof(short), cudaMemcpyHostToDevice);
    if (size / 256 == 0) {
        div_scalar_tensor<<<1, 256>>>(tensor->size, tensor->data, device_value, result->data);
    } else if (size / 256 < 7) {
        div_scalar_tensor<<<get_device_dim(size), 256>>>(tensor->size, tensor->data, device_value, result->data);
    } else {
        div_scalar_tensor<<<7, 256>>>(tensor->size, tensor->data, device_value, result->data);
    }
    cudaDeviceSynchronize();
}

extern "C" void transpose_tensor_wrapper(Tensor* tensor, Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size / 256 == 0) {
        transpose_tensor<<<1, 256>>>(tensor->size, tensor->data, result->data);
    } else if (size / 256 < 7) {
        transpose_tensor<<<get_device_dim(size), 256>>>(tensor->size, tensor->data, result->data);
    } else {
        transpose_tensor<<<7, 256>>>(tensor->size, tensor->data, result->data);
    }
    cudaDeviceSynchronize();
}

extern "C" void sum_tensor_wrapper(Tensor* tensor, Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size / 256 == 0) {
        sum_tensor<<<1, 256>>>(tensor->size, tensor->data, result->data);
    } else if (size / 256 < 7) {
        sum_tensor<<<get_device_dim(size), 256>>>(tensor->size, tensor->data, result->data);
    } else {
        sum_tensor<<<7, 256>>>(tensor->size, tensor->data, result->data);
    }
    cudaDeviceSynchronize();
}

extern "C" void mean_tensor_wrapper(Tensor* tensor, Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size / 256 == 0) {
        mean_tensor<<<1, 256>>>(tensor->size, tensor->data, result->data);
    } else if (size / 256 < 7) {
        mean_tensor<<<get_device_dim(size), 256>>>(tensor->size, tensor->data, result->data);
    } else {
        mean_tensor<<<7, 256>>>(tensor->size, tensor->data, result->data);
    }
    cudaDeviceSynchronize();
}

extern "C" void max_tensor_wrapper(Tensor* tensor, Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size / 256 == 0) {
        max_tensor<<<1, 256>>>(tensor->size, tensor->data, result->data);
    } else if (size / 256 < 7) {
        max_tensor<<<get_device_dim(size), 256>>>(tensor->size, tensor->data, result->data);
    } else {
        max_tensor<<<7, 256>>>(tensor->size, tensor->data, result->data);
    }
}

extern "C" void min_tensor_wrapper(Tensor* tensor, Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size / 256 == 0) {
        min_tensor<<<1, 256>>>(tensor->size, tensor->data, result->data);
    } else if (size / 256 < 7) {
        min_tensor<<<get_device_dim(size), 256>>>(tensor->size, tensor->data, result->data);
    } else {
        min_tensor<<<7, 256>>>(tensor->size, tensor->data, result->data);
    }
}

extern "C" void gradient_tensor_wrapper(Tensor* tensor, Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size / 256 == 0) {
        gradient_tensor<<<1, 256>>>(tensor->size, tensor->data, result->data);
    } else if (size / 256 < 7) {
        gradient_tensor<<<get_device_dim(size), 256>>>(tensor->size, tensor->data, result->data);
    } else {
        gradient_tensor<<<7, 256>>>(tensor->size, tensor->data, result->data);
    }
}

extern "C" void gate_tensor_wrapper(Tensor* tensor, Tensor* booleans, Tensor* result) {
    unsigned int size;
    cudaMemcpy(&size, tensor->size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (size / 256 == 0) {
        gate_tensor<<<1, 256>>>(tensor->size, tensor->data, booleans->data, result->data);
    } else if (size / 256 < 7) {
        gate_tensor<<<get_device_dim(size), 256>>>(tensor->size, tensor->data, booleans->data, result->data);
    } else {
        gate_tensor<<<7, 256>>>(tensor->size, tensor->data, booleans->data, result->data);
    }
}
