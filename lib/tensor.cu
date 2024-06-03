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

Tensor* create_tensor(int size) {
    Tensor *tensor = (Tensor*) malloc(sizeof(Tensor));
    tensor->size = (int*) malloc(sizeof(int));
    tensor->size = &size;
    tensor->data = (short*) malloc(size*sizeof(short)); 
    return tensor;
}

void destroy_tensor(Tensor* tensor) {
    free(tensor->data);
    free(tensor->size);
    free(tensor);
    tensor = NULL;
}

void set_cpu_to_device_tensor(Tensor* tensor) {
    Tensor* d_tensor;
    cudaMalloc((void **) &d_tensor->size, sizeof(int));
    cudaMemcpy(&d_tensor->size, &tensor->size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_tensor->data, tensor->size[0] * sizeof(short));
    cudaMemcpy(d_tensor->data, tensor->data, tensor->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    free(tensor->data);
    free(tensor->size);
    tensor = d_tensor;
    d_tensor = NULL;
}

void set_device_to_cpu_tensor(Tensor* tensor) {
    Tensor* c_tensor = (Tensor*) malloc(sizeof(Tensor*));
    c_tensor->size = (int*) malloc(sizeof(int));
    cudaMemcpy(&c_tensor->size, &tensor->size, sizeof(int), cudaMemcpyDeviceToHost);
    c_tensor->data = (short*) malloc(sizeof(short) * c_tensor->size[0]);
    cudaMemcpy(&c_tensor->data, &tensor->data, sizeof(short) * c_tensor->size[0], cudaMemcpyDeviceToHost);
    cudaFree(tensor->size);
    cudaFree(tensor->data);
    tensor = c_tensor;
    c_tensor = NULL;
}

void copy_device_to_device_tensor(Tensor* tensor, Tensor* result) {
    cudaMemcpy(&result->size, &tensor->size, sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(&result->data, &tensor->data, sizeof(short) * tensor->size[0], cudaMemcpyDeviceToDevice);
}

void copy_device_to_cpu_tensor(Tensor* tensor, Tensor* result) {
    cudaMemcpy(&result->size, &tensor->size, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result->data, &tensor->data, sizeof(short) * tensor->size[0], cudaMemcpyDeviceToHost);
}

void copy_cpu_to_device_tensor(Tensor* tensor, Tensor* result) {
    cudaMemcpy(&result->size, &tensor->size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&result->data, &tensor->data, sizeof(short) * tensor->size[0], cudaMemcpyHostToDevice);
}

void copy_cpu_to_cpu_tensor(Tensor* tensor, Tensor* result) {
    memcpy(&result->size[0], &tensor->size[0], sizeof(int));
    memcpy(&result->data, &tensor->data, sizeof(short) * tensor->size[0]);
}

Tensor* create_device_tensor(int size) {
    Tensor *tensor = (Tensor*) malloc(sizeof(Tensor));
    cudaMalloc((void **) &tensor->size, sizeof(int));
    cudaMemcpy(&tensor->size, &size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void **) &tensor->data, size * sizeof(short));
    return tensor;
}

void destroy_device_tensor(Tensor* tensor) {
    cudaFree(&tensor->data);
    cudaFree(&tensor->size);
    tensor = NULL;
}

int init_tensor(Tensor* tensor, int size) {
    tensor->size = &size;
    tensor->data = (short*)malloc(size * sizeof(short));
    if (!tensor->data) {
        return -1;
    }
    return 0;
}

__global__
void vector_sort_tensor(Tensor* tensor,  Tensor* vectors,  Tensor* result) {
    if (vectors->size[0] != tensor->size[0] || result->size[0] != tensor->size[0]) {
        printf("Error: index out of bounds\n");
        return; 
    }
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < tensor->size[0]; i+= stride) {
        result->data[i] = tensor->data[vectors->data[i]];
    }
}

__global__
void vector_add_tensor(Tensor* tensor, Tensor* other, Tensor* vectors, Tensor* result) {
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = 0; i < tensor->size[0]; i++) {
        result->data[i] = tensor->data[i] + other->data[vectors->data[i]];
    }
}

__global__
void vector_sub_tensor(Tensor* tensor, Tensor* other, Tensor* vectors, Tensor* result) {
    for (int i = 0; i < tensor->size[0]; i++) {
        result->data[i] = tensor->data[i] - other->data[vectors->data[i]];
    }
}

__global__
void vector_mul_tensor(Tensor* tensor, Tensor* other, Tensor* vectors, Tensor* result) {
    for (int i = 0; i < tensor->size[0]; i++) {
        result->data[i] = tensor->data[i] * other->data[vectors->data[i]];
    }
}

__global__
void vector_div_tensor(Tensor* tensor, Tensor* other, Tensor* vectors, Tensor* result) {
    for (int i = 0; i < tensor->size[0]; i++) {
        result->data[i] = tensor->data[i] / other->data[vectors->data[i]];
    }
}

__global__
void vector_gate_tensor(Tensor* tensor, Tensor* booleans, Tensor* vectors, Tensor* result) {
    for (int i = 0; i < tensor->size[0]; i++) {
        if (booleans->data[i] == 1) {
            result->data[i] = tensor->data[vectors->data[i]];
        }
    }
}

__global__
void set_tensor( Tensor* tensor, int index, short value) {
    if (index > tensor->size[0]) {
        printf("Error: index out of bounds\n");
        return;
    }
    tensor->data[index] = value;
}       

        __global__
        void zeros_tensor( Tensor* tensor) {

            int index = threadIdx.x;
            tensor->data[index] = 0;
        }

        __global__
        void ones_tensor( Tensor* tensor) {
            int index = threadIdx.x;
            tensor->data[index] = 1;
        }

        void rm_Tensor( Tensor* tensor) {
            cudaFree(&tensor->data);
        }

        __global__
        void print_tensor( Tensor* tensor) {

            int index = threadIdx.x;
            int stride = blockDim.x;
            printf("Value at index: %d is %d\n", index, tensor->data[index]);
        }

        __global__
        void fill_tensor( Tensor* tensor, int value) {
            int index = threadIdx.x;
            tensor->data[index] = value;
        }

        __global__
        void add_tensor( Tensor* tensor,  Tensor* other,  Tensor* result) {
            int index = threadIdx.x;
            result->data[index] = tensor->data[index], other->data[index];
        }

        __global__
        void sub_tensor( Tensor* tensor,  Tensor* other,  Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            if (tensor->size[0] != other->size[0]) {
                printf("Error: size mismatch\n");
                return;
            }
            for (int i = index; i < tensor->size[0]; i+= stride) {
                result->data[i] = tensor->data[i] - other->data[i];
            }
        }

        __global__
        void mul_tensor( Tensor* tensor,  Tensor* other,  Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            if (tensor->size[0] != other->size[0]) {
                printf("Error: size mismatch\n");
                return;
            }
            for (int i = index; i < tensor->size[0]; i+= stride) {
                result->data[i] = tensor->data[i] * other->data[i];
            }
        }

        __global__
        void div_tensor( Tensor* tensor,  Tensor* other,  Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            if (tensor->size[0] != other->size[0]) {
                printf("Error: size mismatch\n");
                return;
            }
            for (int i = index; i < tensor->size[0]; i+= stride) {
                result->data[i] = tensor->data[i] / other->data[i];
            }
        }

        __global__
        void add_scalar_tensor( Tensor* tensor, int value,  Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size[0]; i+= stride) {
                result->data[i] = tensor->data[i] + value;
            }
        }

        __global__
        void sub_scalar_tensor( Tensor* tensor, int value,  Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size[0]; i+= stride) {
                result->data[i] = tensor->data[i] - value;
            }
        }

        __global__
        void mul_scalar_tensor( Tensor* tensor, int value,  Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size[0]; i+= stride) {
                result->data[i] = tensor->data[i] * value;
            }
        }

        __global__
        void div_scalar_tensor( Tensor* tensor, int value,  Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size[0]; i+= stride) {
                result->data[i] = tensor->data[i] / value;
            }
        }

        __global__
        void transpose_tensor( Tensor* tensor,  Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size[0]; i+= stride) {
                result->data[i] = tensor->data[tensor->size[0] - i - 1];
            }
        }

        __global__
        void sum_tensor( Tensor* tensor,  Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size[0]; i+= stride) {
                result->data[0] += tensor->data[i];
            }
        }

        __global__
        void mean_tensor( Tensor* tensor,  Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size[0]; i+= stride) {
                result->data[0] += tensor->data[i];
            }
            result->data[0] /= tensor->size[0];
        }

        __global__
        void max_tensor( Tensor* tensor,  Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size[0]; i+= stride) {
                if (tensor->data[i] > result->data[0]) {
                    result->data[0] = tensor->data[i];
                }
            }
        }

        __global__
        void min_tensor( Tensor* tensor,  Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size[0]; i+= stride) {
                if (tensor->data[i] < result->data[0]) {
                    result->data[0] = tensor->data[i];
                }
            }
        }

        __global__
        void gradient_tensor( Tensor* tensor,  Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size[0]; i+= stride) {
                result->data[i] = tensor->data[i + 1] - tensor->data[i];
            }
        }

        __global__
        void gate_tensor( Tensor* tensor,  Tensor* bools,  Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size[0]; i+= stride) {
                if (bools->data[i] == 1) {
                    result->data[i] = tensor->data[i];
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

extern "C" void vector_add_wrapper( Tensor* tensor,  Tensor* other,  Tensor* vectors,  Tensor* result) {
    int size;
    cudaMemcpy(&size, tensor->size, sizeof(int), cudaMemcpyDeviceToHost);
    if (size < 256 == 0) {
        vector_add_tensor<<<1, 256>>>(tensor, other, vectors, result);
    } else if (size / 256 < 8) {
        vector_add_tensor<<<(size / 256), 256>>>(tensor, other, vectors, result);
    } else {
        vector_add_tensor<<<7, 256>>>(tensor, other, vectors, result);
    }
    cudaDeviceSynchronize();
}

extern "C" void vector_sub_wrapper( Tensor* tensor,  Tensor* other,  Tensor* vectors,  Tensor* result) {
    int size;
    cudaMemcpy(&size, tensor->size, sizeof(int), cudaMemcpyDeviceToHost);
    if (size < 256 == 0) {
        vector_sub_tensor<<<1, 256>>>(tensor, other, vectors, result);
    } else if (size / 256 < 8) {
        vector_sub_tensor<<<(size / 256), 256>>>(tensor, other, vectors, result);
    } else {
        vector_sub_tensor<<<7, 256>>>(tensor, other, vectors, result);
    }
    cudaDeviceSynchronize();
}

extern "C" void vector_mul_wrapper( Tensor* tensor,  Tensor* other,  Tensor* vectors,  Tensor* result) {
    int size;
    cudaMemcpy(&size, tensor->size, sizeof(int), cudaMemcpyDeviceToHost);
    if (size < 256 == 0) {
        vector_mul_tensor<<<1, 256>>>(tensor, other, vectors, result);
    } else if (size / 256 < 8) {
        vector_mul_tensor<<<(size / 256), 256>>>(tensor, other, vectors, result);
    } else {
        vector_mul_tensor<<<7, 256>>>(tensor, other, vectors, result);
    }
    cudaDeviceSynchronize();
}

extern "C" void vector_div_wrapper( Tensor* tensor,  Tensor* other,  Tensor* vectors,  Tensor* result) {
    int size;
    cudaMemcpy(&size, tensor->size, sizeof(int), cudaMemcpyDeviceToHost);
    if (size < 256 == 0) {
        vector_div_tensor<<<1, 256>>>(tensor, other, vectors, result);
    } else if (size / 256 < 8) {
        vector_div_tensor<<<(size / 256), 256>>>(tensor, other, vectors, result);
    } else {
        vector_div_tensor<<<7, 256>>>(tensor, other, vectors, result);
    }
    cudaDeviceSynchronize();
}

extern "C" void vector_gate_wrapper( Tensor* tensor,  Tensor* booleans,  Tensor* vectors,  Tensor* result) {
    int size;
    cudaMemcpy(&size, tensor->size, sizeof(int), cudaMemcpyDeviceToHost);
    if (size < 256 == 0) {
        vector_gate_tensor<<<1, 256>>>(tensor, booleans, vectors, result);
    } else if (size / 256 < 8) {
        vector_gate_tensor<<<(size / 256), 256>>>(tensor, booleans, vectors, result);
    } else {
        vector_gate_tensor<<<7, 256>>>(tensor, booleans, vectors, result);
    }
    cudaDeviceSynchronize();
}

extern "C" void zeros_tensor_wrapper( Tensor* tensor) {
    int size;
    cudaMemcpy(&size, tensor->size, sizeof(int), cudaMemcpyDeviceToHost);
    zeros_tensor<<<1,size>>>(tensor);
    cudaDeviceSynchronize();
}

extern "C" void ones_tensor_wrapper( Tensor* tensor) {
    int size;
    cudaMemcpy(&size, tensor->size, sizeof(int), cudaMemcpyDeviceToHost);
    ones_tensor<<<1, size>>>(tensor);
    cudaDeviceSynchronize();
}

extern "C" void vector_sort_tensor_wrapper( Tensor* tensor,  Tensor* vectors,  Tensor* result)  {
    int size;
    cudaMemcpy(&size, tensor->size, sizeof(int), cudaMemcpyDeviceToHost);

   if (size / 256 == 0) {
        vector_sort_tensor<<<1, 256>>>(tensor, vectors, result);
    } else if (size / 256 < 8) {
        vector_sort_tensor<<<(size / 256), 256>>>(tensor, vectors, result);
    } else {
        vector_sort_tensor<<<7, 256>>>(tensor, vectors, result);
    }
    cudaDeviceSynchronize();
}

extern "C" void print_tensor_wrapper( Tensor* tensor) {
     int size;
    cudaMemcpy(&size, tensor->size, sizeof(int), cudaMemcpyDeviceToHost);
    print_tensor<<<1, size>>>(tensor);
    cudaDeviceSynchronize();
}

extern "C" void fill_tensor_wrapper( Tensor* tensor, int value) {
    int size;
    cudaMemcpy(&size, tensor->size, sizeof(int), cudaMemcpyDeviceToHost);
    if (size / 256 == 0) {
        fill_tensor<<<1, 256>>>(tensor, value);
    } else if (size / 256 < 8) {
        fill_tensor<<<(size / 256), 256>>>(tensor, value);
    } else {
        fill_tensor<<<7, 256>>>(tensor, value);
    }
    cudaDeviceSynchronize();
    cudaFree(tensor);
}

extern "C" void add_tensor_wrapper( Tensor* tensor,  Tensor* other,  Tensor* result) {
    int size;
    cudaMemcpy(&size, tensor->size, sizeof(int), cudaMemcpyDeviceToHost);
    add_tensor<<<1, size>>>(tensor, other, result);
    cudaDeviceSynchronize();
}

extern "C" void sub_tensor_wrapper( Tensor* tensor,  Tensor* vectors,  Tensor* result) {
    int size;
    cudaMemcpy(&size, tensor->size, sizeof(int), cudaMemcpyDeviceToHost);
    if (size / 256 == 0) {
          sub_tensor<<<1, 256>>>(tensor, vectors, result);
     } else if (size / 256 < 8) {
          sub_tensor<<<(size / 256), 256>>>(tensor, vectors, result);
     } else {
          sub_tensor<<<7, 256>>>(tensor, vectors, result);
     }
    cudaDeviceSynchronize();
}

extern "C" void mul_tensor_wrapper( Tensor* tensor,  Tensor* vectors,  Tensor* result) {
    int size;
    cudaMemcpy(&size, tensor->size, sizeof(int), cudaMemcpyDeviceToHost);
    if (size / 256 == 0) {
          mul_tensor<<<1, 256>>>(tensor, vectors, result);
     } else if (size / 256 < 8) {
          mul_tensor<<<(size / 256), 256>>>(tensor, vectors, result);
     } else {
          mul_tensor<<<7, 256>>>(tensor, vectors, result);
     }
    cudaDeviceSynchronize();
}

extern "C" void div_tensor_wrapper( Tensor* tensor,  Tensor* vectors,  Tensor* result) {
    int size;
    cudaMemcpy(&size, tensor->size, sizeof(int), cudaMemcpyDeviceToHost);
    if (size / 256 == 0) {
          div_tensor<<<1, 256>>>(tensor, vectors, result);
     } else if (size / 256 < 8) {
          div_tensor<<<(size / 256), 256>>>(tensor, vectors, result);
     } else {
          div_tensor<<<7, 256>>>(tensor, vectors, result);
     }
    cudaDeviceSynchronize();
}

extern "C" void add_scalar_tensor_wrapper(Tensor* tensor, int value, Tensor* result) {
    int size;
    cudaMemcpy(&size, tensor->size, sizeof(int), cudaMemcpyDeviceToHost);
    if (size / 256 == 0) {
        add_scalar_tensor<<<1, 256>>>(tensor, value, result);
    } else if (size / 256 < 8) {
        add_scalar_tensor<<<(size / 256), 256>>>(tensor, value, result);
    } else {
        add_scalar_tensor<<<7, 256>>>(tensor, value, result);
    }
    cudaDeviceSynchronize();
}

extern "C" void sub_scalar_tensor_wrapper(Tensor* tensor, int value, Tensor* result) {
    int size;
    cudaMemcpy(&size, tensor->size, sizeof(int), cudaMemcpyDeviceToHost);
    if (size / 256 == 0) {
        sub_scalar_tensor<<<1, 256>>>(tensor, value, result);
    } else if (size / 256 < 8) {
        sub_scalar_tensor<<<(size / 256), 256>>>(tensor, value, result);
    } else {
        sub_scalar_tensor<<<7, 256>>>(tensor, value, result);
    }
    cudaDeviceSynchronize();
}

extern "C" void mul_scalar_tensor_wrapper(Tensor* tensor, int value, Tensor* result) {
    int size;
    cudaMemcpy(&size, tensor->size, sizeof(int), cudaMemcpyDeviceToHost);
    if (size / 256 == 0) {
        mul_scalar_tensor<<<1, 256>>>(tensor, value, result);
    } else if (size / 256 < 8) {
        mul_scalar_tensor<<<(size / 256), 256>>>(tensor, value, result);
    } else {
        mul_scalar_tensor<<<7, 256>>>(tensor, value, result);
    }
    cudaDeviceSynchronize();
}

extern "C" void div_scalar_tensor_wrapper(Tensor* tensor, int scalar, Tensor* result) {
    int size;
    cudaMemcpy(&size, tensor->size, sizeof(int), cudaMemcpyDeviceToHost);
    if (size / 256 == 0) {
        div_scalar_tensor<<<1, 256>>>(tensor, scalar, result);
    } else if (size / 256 < 8) {
        div_scalar_tensor<<<(size / 256), 256>>>(tensor, scalar, result);
    } else {
        div_scalar_tensor<<<7, 256>>>(tensor, scalar, result);
    }
    cudaDeviceSynchronize();
}

extern "C" void transpose_tensor_wrapper(Tensor* tensor, Tensor* result) {
    int size;
    cudaMemcpy(&size, tensor->size, sizeof(int), cudaMemcpyDeviceToHost);
    if (size / 256 == 0) {
        transpose_tensor<<<1, 256>>>(tensor, result);
    } else if (size / 256 < 8) {
        transpose_tensor<<<(size / 256), 256>>>(tensor, result);
    } else {
        transpose_tensor<<<7, 256>>>(tensor, result);
    }
    cudaDeviceSynchronize();
}

extern "C" void sum_tensor_wrapper(Tensor* tensor, Tensor* result) {
    int size;
    cudaMemcpy(&size, tensor->size, sizeof(int), cudaMemcpyDeviceToHost);
    if (size / 256 == 0) {
        sum_tensor<<<1, 256>>>(tensor, result);
    } else if (size / 256 < 8) {
        sum_tensor<<<(size / 256), 256>>>(tensor, result);
    } else {
        sum_tensor<<<7, 256>>>(tensor, result);
    }
    cudaDeviceSynchronize();
}

extern "C" void mean_tensor_wrapper(Tensor* tensor, Tensor* result) {
    int size;
    cudaMemcpy(&size, tensor->size, sizeof(int), cudaMemcpyDeviceToHost);
    if (size / 256 == 0) {
        mean_tensor<<<1, 256>>>(tensor, result);
    } else if (size / 256 < 8) {
        mean_tensor<<<(size / 256), 256>>>(tensor, result);
    } else {
        mean_tensor<<<7, 256>>>(tensor, result);
    }
    cudaDeviceSynchronize();
}

extern "C" void max_tensor_wrapper(Tensor* tensor, Tensor* result) {
    int size;
    cudaMemcpy(&size, tensor->size, sizeof(int), cudaMemcpyDeviceToHost);
    if (size / 256 == 0) {
        max_tensor<<<1, 256>>>(tensor, result);
    } else if (size / 256 < 8) {
        max_tensor<<<(size / 256), 256>>>(tensor, result);
    } else {
        max_tensor<<<7, 256>>>(tensor, result);
    }
}

extern "C" void min_tensor_wrapper(Tensor* tensor, Tensor* result) {
    int size;
    cudaMemcpy(&size, tensor->size, sizeof(int), cudaMemcpyDeviceToHost);
    if (size / 256 == 0) {
        min_tensor<<<1, 256>>>(tensor, result);
    } else if (size / 256 < 8) {
        min_tensor<<<(size / 256), 256>>>(tensor, result);
    } else {
        min_tensor<<<7, 256>>>(tensor, result);
    }
}

extern "C" void gradient_tensor_wrapper(Tensor* tensor, Tensor* result) {
    int size;
    cudaMemcpy(&size, tensor->size, sizeof(int), cudaMemcpyDeviceToHost);
    if (size / 256 == 0) {
        gradient_tensor<<<1, 256>>>(tensor, result);
    } else if (size / 256 < 8) {
        gradient_tensor<<<(size / 256), 256>>>(tensor, result);
    } else {
        gradient_tensor<<<7, 256>>>(tensor, result);
    }
}

extern "C" void gate_tensor_wrapper(Tensor* tensor, Tensor* booleans, Tensor* result) {
    int size;
    cudaMemcpy(&size, tensor->size, sizeof(int), cudaMemcpyDeviceToHost);
    if (size / 256 == 0) {
        gate_tensor<<<1, 256>>>(tensor, booleans, result);
    } else if (size / 256 < 8) {
        gate_tensor<<<(size / 256), 256>>>(tensor, booleans, result);
    } else {
        gate_tensor<<<7, 256>>>(tensor, booleans, result);
    }
}
