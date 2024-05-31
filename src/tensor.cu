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

Tensor* destroy_tensor(Tensor* tensor) {
    free(tensor->data);
    free(tensor);
    return NULL;
}

Tensor* create_device_tensor(int size) {
    Tensor *tensor;
    cudaError_t cuda_fun_err = cudaMalloc(&tensor, sizeof(Tensor));
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
            int stride = blockDim.x;
            for (int i = index; i < tensor->size[0] - 1; i+= stride) {
                tensor->data[i] = 0;
            }
        }

        __global__
        void ones_tensor( Tensor* tensor) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size[0]; i+= stride) {
                tensor->data[i] = 1;
            }
        }

        
        void rm_Tensor( Tensor* tensor) {
            cudaFree(&tensor->data);
        }

        __global__
        void print_tensor( Tensor* tensor) {

            int index = threadIdx.x;
            int stride = blockDim.x;
            printf("%d\n", tensor->size[0]);
            for (int i = index; i < tensor->size[0] - 1; i+= stride) {
                printf("Printing from da GPU...\n");
                printf("%hd ", tensor->data[i]);
            }
            printf("\n");
        }

        __global__
        void fill_tensor( Tensor* tensor, int value) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size[0]; i+= stride) {
                tensor->data[i] = value;
            }
        }

        __global__
        void add_tensor( Tensor* tensor,  Tensor* other,  Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            if (tensor->size[0] != other->size[0]) {
                printf("Error: size mismatch\n");
                return;
            }
            for (int i = index; i < tensor->size[0]; i+= stride) {
                result->data[i] = tensor->data[i] + other->data[i];
            }
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

extern "C" Tensor* create_tensor_wrapper(int size) {
    return create_tensor(size);
}

extern "C" void zeros_tensor_wrapper( Tensor* tensor) {
#ifdef __DEBUG__

    printf("making cpu tensor\n");
#endif
    Tensor* d_tensor = (Tensor*) malloc(sizeof(Tensor));
    cudaError_t cuda_err = cudaMalloc((void **) &d_tensor->size, sizeof(int));
#ifdef __DEBUG__
    printf("%s\n", cudaGetErrorString(cuda_err));
    #endif
    if (cuda_err != cudaSuccess) {
#ifdef __DEBUG__
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuda_err));
#endif
        return;
    }
    else {
#ifdef __DEBUG__
        printf("%p\n", &tensor->size[0]);
        printf("%p\n", &d_tensor->size[0]);
        printf("%d\n", sizeof(int));
#endif
        cuda_err = cudaMemcpy(&d_tensor->size[0], &tensor->size, sizeof(int), cudaMemcpyHostToDevice);
#ifdef __DEBUG__
        printf("%s\n", cudaGetErrorString(cuda_err));
#endif
        if (cuda_err != cudaSuccess) {
#ifdef __DEBUG__
            printf("Error in cudaMemcpy \n");
            printf("Error: %s\n", cudaGetErrorString(cuda_err));
#endif
            return;
        }
    }
    cuda_err = cudaMalloc((void **) &d_tensor->data, tensor->size[0] * sizeof(short));
#ifdef __DEBUG__
    printf("%s\n", cudaGetErrorString(cuda_err));
#endif
    if (cuda_err != cudaSuccess) {
#ifdef __DEBUG__
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuda_err));
#endif
        return;
    }
    else {
        cuda_err = cudaMemcpy(d_tensor->data, tensor->data, tensor->size[0] * sizeof(short), cudaMemcpyHostToDevice);
#ifdef __DEBUG__
        printf("%s\n", cudaGetErrorString(cuda_err));
#endif
    }
    if (cuda_err != cudaSuccess) {
#ifdef __DEBUG__
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuda_err));
#endif
        return;
    }
    if (tensor->size[0] < 256 == 0) {
        zeros_tensor<<<1, 256>>>(d_tensor);
    } else if (tensor->size[0] / 256 < 8) {
        zeros_tensor<<<(tensor->size[0] / 256), 256>>>(d_tensor);
    } else {
        zeros_tensor<<<7, 256>>>(d_tensor);
    }
    cuda_err = cudaDeviceSynchronize();
#ifdef __DEBUG__
    printf("%s\n", cudaGetErrorString(cuda_err));
#endif
    if (cuda_err != cudaSuccess) {
#ifdef __DEBUG__
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cuda_err));
#endif
    }
    cudaFree(d_tensor->data);
    cudaFree(d_tensor->size);
}

extern "C" void ones_tensor_wrapper( Tensor* tensor) {
#ifdef __DEBUG__

    printf("making cpu tensor\n");
#endif
    Tensor* d_tensor = (Tensor*) malloc(sizeof(Tensor));
    cudaError_t cuda_err = cudaMalloc((void **) &d_tensor->size, sizeof(int));
#ifdef __DEBUG__
    printf("%s\n", cudaGetErrorString(cuda_err));
    #endif
    if (cuda_err != cudaSuccess) {
#ifdef __DEBUG__
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuda_err));
#endif
        return;
    }
    else {
#ifdef __DEBUG__
        printf("%p\n", &tensor->size[0]);
        printf("%p\n", &d_tensor->size[0]);
        printf("%d\n", sizeof(int));
#endif
        cuda_err = cudaMemcpy(&d_tensor->size[0], &tensor->size, sizeof(int), cudaMemcpyHostToDevice);
#ifdef __DEBUG__
        printf("%s\n", cudaGetErrorString(cuda_err));
#endif
        if (cuda_err != cudaSuccess) {
#ifdef __DEBUG__
            printf("Error in cudaMemcpy \n");
            printf("Error: %s\n", cudaGetErrorString(cuda_err));
#endif
            return;
        }
    }
    cuda_err = cudaMalloc((void **) &d_tensor->data, tensor->size[0] * sizeof(short));
#ifdef __DEBUG__
    printf("%s\n", cudaGetErrorString(cuda_err));
#endif
    if (cuda_err != cudaSuccess) {
#ifdef __DEBUG__
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuda_err));
#endif
        return;
    }
    else {
        cuda_err = cudaMemcpy(d_tensor->data, tensor->data, tensor->size[0] * sizeof(short), cudaMemcpyHostToDevice);
#ifdef __DEBUG__
        printf("%s\n", cudaGetErrorString(cuda_err));
#endif
    }
    if (cuda_err != cudaSuccess) {
#ifdef __DEBUG__
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuda_err));
#endif
        return;
    }
    if (tensor->size[0] < 256 == 0) {
        ones_tensor<<<1, 256>>>(d_tensor);
    } else if (tensor->size[0] / 256 < 8) {
        ones_tensor<<<(tensor->size[0] / 256), 256>>>(d_tensor);
    } else {
        ones_tensor<<<7, 256>>>(d_tensor);
    }
    cuda_err = cudaDeviceSynchronize();
#ifdef __DEBUG__
    printf("%s\n", cudaGetErrorString(cuda_err));
#endif
    if (cuda_err != cudaSuccess) {
#ifdef __DEBUG__
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cuda_err));
#endif
    }
    cudaFree(d_tensor->data);
    cudaFree(d_tensor->size);
}

extern "C" void vector_sort_tensor_wrapper( Tensor* tensor,  Tensor* vectors,  Tensor* result) {
    if (!check_size_3(tensor, vectors, result)) {
        return;
    }
    Tensor* d_tensor = (Tensor*) malloc(sizeof(Tensor));
    Tensor* d_vectors = (Tensor*) malloc(sizeof(Tensor));
    Tensor* d_result = (Tensor*) malloc(sizeof(Tensor));
    cudaError_t cuderr = cudaMalloc((void **) &d_tensor->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_tensor->size[0], &tensor->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_tensor->data, tensor->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_tensor->data, tensor->data, tensor->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_vectors->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_vectors->size[0], &vectors->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_vectors->data, vectors->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_vectors->data, vectors->data, vectors->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_result->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_result->size[0], &result->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_result->data, result->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_result->data, result->data, result->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif

   if (tensor->size[0] / 256 == 0) {
        vector_sort_tensor<<<1, 256>>>(tensor, vectors, result);
    } else if (tensor->size[0] / 256 < 8) {
        vector_sort_tensor<<<(tensor->size[0] / 256), 256>>>(tensor, vectors, result);
    } else {
        vector_sort_tensor<<<7, 256>>>(tensor, vectors, result);
    }
    cuderr = cudaDeviceSynchronize();
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cuderr));
    }
    #endif
    cudaFree(d_tensor->data);
    cudaFree(d_tensor->size);
    cudaFree(d_vectors->data);
    cudaFree(d_vectors->size);
    cudaFree(d_result->data);
    cudaFree(d_result->size);
}

extern "C" void print_tensor_wrapper( Tensor* tensor) {
    #ifdef __DEBUG__
    printf("making cpu tensor\n");
    #endif
    Tensor* d_tensor = (Tensor*) malloc(sizeof(Tensor));
    #ifdef __DEBUG__
    printf("making gpu tensor\n");
    #endif
    cudaError_t cuderr = cudaMalloc((void **) &d_tensor->size, sizeof(int));
    #ifdef __DEBUG__
    printf("%s\n", cudaGetErrorString(cuderr));
    #endif
    if (cuderr != cudaSuccess) {
        #ifdef __DEBUG__
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        #endif
        return;
    }
    else {
        cuderr = cudaMemcpy(&d_tensor->size[0], &tensor->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    printf("%s\n", cudaGetErrorString(cuderr));
    #endif
        if (cuderr != cudaSuccess) {
            #ifdef __DEBUG__
            printf("Error in cudaMemcpy \n");
            printf("Error: %s\n", cudaGetErrorString(cuderr));
            #endif
            return;
        }
    
    }

    cuderr = cudaMalloc((void **) &d_tensor->data, tensor->size[0]*sizeof(short));
    #ifdef __DEBUG__
    printf("%s\n", cudaGetErrorString(cuderr));
    #endif
    if (cuderr != cudaSuccess) {
        #ifdef __DEBUG__
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        #endif
        return;
    }
    cuderr = cudaMemcpy(d_tensor->data, tensor->data, tensor->size[0]*sizeof(short), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    printf("%s\n", cudaGetErrorString(cuderr));
    #endif
    if (cuderr != cudaSuccess) {
        #ifdef __DEBUG__
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        #endif
        return;
    }
    if (tensor->size[0] < 256 == 0) {
        print_tensor<<<1, 256>>>(d_tensor);
    } else if (tensor->size[0] / 256 < 8) {
        print_tensor<<<(tensor->size[0] / 256), 256>>>(d_tensor);
    } else {
        print_tensor<<<7, 256>>>(d_tensor);
    }
    
    cuderr = cudaDeviceSynchronize();
    #ifdef __DEBUG__
    printf("%s\n", cudaGetErrorString(cuderr));
    #endif
    if (cuderr != cudaSuccess) {
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cuderr));
    }
    cudaFree(d_tensor->data);
    cudaFree(d_tensor->size);
}

extern "C" void fill_tensor_wrapper( Tensor* tensor, int value) {
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    if (tensor->size[0] % 256 == 0) {
        fill_tensor<<<1, 256>>>(tensor, value);
    } else if (tensor->size[0] % 256 < 8) {
        fill_tensor<<<(tensor->size[0] % 256), 256>>>(tensor, value);
    } else {
        fill_tensor<<<7, 256>>>(tensor, value);
    }
    cudaDeviceSynchronize();
    cudaFree(tensor);
}

extern "C" void add_tensor_wrapper( Tensor* tensor,  Tensor* vectors,  Tensor* result) {
    if (!check_size_3(tensor, vectors, result)) {
        return;
    }
    Tensor* d_tensor = (Tensor*) malloc(sizeof(Tensor));
    Tensor* d_vectors = (Tensor*) malloc(sizeof(Tensor));
    Tensor* d_result = (Tensor*) malloc(sizeof(Tensor));
    cudaError_t cuderr = cudaMalloc((void **) &d_tensor->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_tensor->size[0], &tensor->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_tensor->data, tensor->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_tensor->data, tensor->data, tensor->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_vectors->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_vectors->size[0], &vectors->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_vectors->data, vectors->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_vectors->data, vectors->data, vectors->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_result->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_result->size[0], &result->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_result->data, result->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_result->data, result->data, result->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif

   if (tensor->size[0] / 256 == 0) {
        add_tensor<<<1, 256>>>(tensor, vectors, result);
    } else if (tensor->size[0] / 256 < 8) {
        add_tensor<<<(tensor->size[0] / 256), 256>>>(tensor, vectors, result);
    } else {
        add_tensor<<<7, 256>>>(tensor, vectors, result);
    }
    cuderr = cudaDeviceSynchronize();
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cuderr));
    }
    #endif
    cudaFree(d_tensor->data);
    cudaFree(d_tensor->size);
    cudaFree(d_vectors->data);
    cudaFree(d_vectors->size);
    cudaFree(d_result->data);
    cudaFree(d_result->size);
}

extern "C" void sub_tensor_wrapper( Tensor* tensor,  Tensor* vectors,  Tensor* result) {
    if (!check_size_3(tensor, vectors, result)) {
        return;
    }
    Tensor* d_tensor = (Tensor*) malloc(sizeof(Tensor));
    Tensor* d_vectors = (Tensor*) malloc(sizeof(Tensor));
    Tensor* d_result = (Tensor*) malloc(sizeof(Tensor));
    cudaError_t cuderr = cudaMalloc((void **) &d_tensor->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_tensor->size[0], &tensor->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_tensor->data, tensor->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_tensor->data, tensor->data, tensor->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_vectors->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_vectors->size[0], &vectors->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_vectors->data, vectors->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_vectors->data, vectors->data, vectors->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_result->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_result->size[0], &result->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_result->data, result->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_result->data, result->data, result->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif

    if (tensor->size[0] / 256 == 0) {
          sub_tensor<<<1, 256>>>(tensor, vectors, result);
     } else if (tensor->size[0] / 256 < 8) {
          sub_tensor<<<(tensor->size[0] / 256), 256>>>(tensor, vectors, result);
     } else {
          sub_tensor<<<7, 256>>>(tensor, vectors, result);
     }
    cuderr = cudaDeviceSynchronize();
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cuderr));
    }
    #endif
    cudaFree(d_tensor->data);
    cudaFree(d_tensor->size);
    cudaFree(d_vectors->data);
    cudaFree(d_vectors->size);
    cudaFree(d_result->data);
    cudaFree(d_result->size);
}

extern "C" void mul_tensor_wrapper( Tensor* tensor,  Tensor* vectors,  Tensor* result) {
    if (!check_size_3(tensor, vectors, result)) {
        return;
    }
    Tensor* d_tensor = (Tensor*) malloc(sizeof(Tensor));
    Tensor* d_vectors = (Tensor*) malloc(sizeof(Tensor));
    Tensor* d_result = (Tensor*) malloc(sizeof(Tensor));
    cudaError_t cuderr = cudaMalloc((void **) &d_tensor->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_tensor->size[0], &tensor->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_tensor->data, tensor->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_tensor->data, tensor->data, tensor->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_vectors->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_vectors->size[0], &vectors->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_vectors->data, vectors->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_vectors->data, vectors->data, vectors->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_result->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_result->size[0], &result->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_result->data, result->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_result->data, result->data, result->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif

    if (tensor->size[0] / 256 == 0) {
          mul_tensor<<<1, 256>>>(tensor, vectors, result);
     } else if (tensor->size[0] / 256 < 8) {
          mul_tensor<<<(tensor->size[0] / 256), 256>>>(tensor, vectors, result);
     } else {
          mul_tensor<<<7, 256>>>(tensor, vectors, result);
     }
    cuderr = cudaDeviceSynchronize();
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cuderr));
    }
    #endif
    cudaFree(d_tensor->data);
    cudaFree(d_tensor->size);
    cudaFree(d_vectors->data);
    cudaFree(d_vectors->size);
    cudaFree(d_result->data);
    cudaFree(d_result->size);
}

extern "C" void div_tensor_wrapper( Tensor* tensor,  Tensor* vectors,  Tensor* result) {
    if (!check_size_3(tensor, vectors, result)) {
        return;
    }
    Tensor* d_tensor = (Tensor*) malloc(sizeof(Tensor));
    Tensor* d_vectors = (Tensor*) malloc(sizeof(Tensor));
    Tensor* d_result = (Tensor*) malloc(sizeof(Tensor));
    cudaError_t cuderr = cudaMalloc((void **) &d_tensor->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_tensor->size[0], &tensor->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_tensor->data, tensor->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_tensor->data, tensor->data, tensor->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_vectors->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_vectors->size[0], &vectors->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_vectors->data, vectors->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_vectors->data, vectors->data, vectors->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_result->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_result->size[0], &result->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_result->data, result->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_result->data, result->data, result->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif

    if (tensor->size[0] / 256 == 0) {
          div_tensor<<<1, 256>>>(tensor, vectors, result);
     } else if (tensor->size[0] / 256 < 8) {
          div_tensor<<<(tensor->size[0] / 256), 256>>>(tensor, vectors, result);
     } else {
          div_tensor<<<7, 256>>>(tensor, vectors, result);
     }
    cuderr = cudaDeviceSynchronize();
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cuderr));
    }
    #endif
    cudaFree(d_tensor->data);
    cudaFree(d_tensor->size);
    cudaFree(d_vectors->data);
    cudaFree(d_vectors->size);
    cudaFree(d_result->data);
    cudaFree(d_result->size);
}

extern "C" void add_scalar_tensor_wrapper(Tensor* tensor, int value, Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    Tensor* d_tensor = (Tensor*) malloc(sizeof(Tensor));
    Tensor* d_result = (Tensor*) malloc(sizeof(Tensor));
    cudaError_t cuderr = cudaMalloc((void **) &d_tensor->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_tensor->size[0], &tensor->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_tensor->data, tensor->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_tensor->data, tensor->data, tensor->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_result->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_result->size[0], &result->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_result->data, result->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_result->data, result->data, result->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif

    if (tensor->size[0] / 256 == 0) {
        add_scalar_tensor<<<1, 256>>>(tensor, value, result);
    } else if (tensor->size[0] / 256 < 8) {
        add_scalar_tensor<<<(tensor->size[0] / 256), 256>>>(tensor, value, result);
    } else {
        add_scalar_tensor<<<7, 256>>>(tensor, value, result);
    }
    cuderr = cudaDeviceSynchronize();
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cuderr));
    }
    #endif
    cudaFree(d_tensor->data);
    cudaFree(d_tensor->size);
    cudaFree(d_result->data);
    cudaFree(d_result->size);
}

extern "C" void sub_scalar_tensor_wrapper(Tensor* tensor, int value, Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    Tensor* d_tensor = (Tensor*) malloc(sizeof(Tensor));
    Tensor* d_result = (Tensor*) malloc(sizeof(Tensor));
    cudaError_t cuderr = cudaMalloc((void **) &d_tensor->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_tensor->size[0], &tensor->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_tensor->data, tensor->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_tensor->data, tensor->data, tensor->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_result->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_result->size[0], &result->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_result->data, result->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_result->data, result->data, result->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif

    if (tensor->size[0] / 256 == 0) {
        sub_scalar_tensor<<<1, 256>>>(tensor, value, result);
    } else if (tensor->size[0] / 256 < 8) {
        sub_scalar_tensor<<<(tensor->size[0] / 256), 256>>>(tensor, value, result);
    } else {
        sub_scalar_tensor<<<7, 256>>>(tensor, value, result);
    }
    cuderr = cudaDeviceSynchronize();
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cuderr));
    }
    #endif
    cudaFree(d_tensor->data);
    cudaFree(d_tensor->size);
    cudaFree(d_result->data);
    cudaFree(d_result->size);
}

extern "C" void mul_scalar_tensor_wrapper(Tensor* tensor, int value, Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    Tensor* d_tensor = (Tensor*) malloc(sizeof(Tensor));
    Tensor* d_result = (Tensor*) malloc(sizeof(Tensor));
    cudaError_t cuderr = cudaMalloc((void **) &d_tensor->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_tensor->size[0], &tensor->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_tensor->data, tensor->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_tensor->data, tensor->data, tensor->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_result->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_result->size[0], &result->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_result->data, result->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_result->data, result->data, result->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif

    if (tensor->size[0] / 256 == 0) {
        mul_scalar_tensor<<<1, 256>>>(tensor, value, result);
    } else if (tensor->size[0] / 256 < 8) {
        mul_scalar_tensor<<<(tensor->size[0] / 256), 256>>>(tensor, value, result);
    } else {
        mul_scalar_tensor<<<7, 256>>>(tensor, value, result);
    }
    cuderr = cudaDeviceSynchronize();
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cuderr));
    }
    #endif
    cudaFree(d_tensor->data);
    cudaFree(d_tensor->size);
    cudaFree(d_result->data);
    cudaFree(d_result->size);
}

extern "C" void div_scalar_tensor_wrapper(Tensor* tensor, int scalar, Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    Tensor* d_tensor = (Tensor*) malloc(sizeof(Tensor));
    Tensor* d_result = (Tensor*) malloc(sizeof(Tensor));
    cudaError_t cuderr = cudaMalloc((void **) &d_tensor->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_tensor->size[0], &tensor->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_tensor->data, tensor->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_tensor->data, tensor->data, tensor->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_result->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_result->size[0], &result->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_result->data, result->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_result->data, result->data, result->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif

    if (tensor->size[0] / 256 == 0) {
        div_scalar_tensor<<<1, 256>>>(tensor, scalar, result);
    } else if (tensor->size[0] / 256 < 8) {
        div_scalar_tensor<<<(tensor->size[0] / 256), 256>>>(tensor, scalar, result);
    } else {
        div_scalar_tensor<<<7, 256>>>(tensor, scalar, result);
    }
    cuderr = cudaDeviceSynchronize();
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cuderr));
    }
    #endif
    cudaFree(d_tensor->data);
    cudaFree(d_tensor->size);
    cudaFree(d_result->data);
    cudaFree(d_result->size);
}

extern "C" void transpose_tensor_wrapper(Tensor* tensor, Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    Tensor* d_tensor = (Tensor*) malloc(sizeof(Tensor));
    Tensor* d_result = (Tensor*) malloc(sizeof(Tensor));
    cudaError_t cuderr = cudaMalloc((void **) &d_tensor->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_tensor->size[0], &tensor->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_tensor->data, tensor->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_tensor->data, tensor->data, tensor->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_result->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_result->size[0], &result->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_result->data, result->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_result->data, result->data, result->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif

    if (tensor->size[0] / 256 == 0) {
        transpose_tensor<<<1, 256>>>(tensor, result);
    } else if (tensor->size[0] / 256 < 8) {
        transpose_tensor<<<(tensor->size[0] / 256), 256>>>(tensor, result);
    } else {
        transpose_tensor<<<7, 256>>>(tensor, result);
    }
    cuderr = cudaDeviceSynchronize();
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cuderr));
    }
    #endif
    cudaFree(d_tensor->data);
    cudaFree(d_tensor->size);
    cudaFree(d_result->data);
    cudaFree(d_result->size);
}

extern "C" void sum_tensor_wrapper(Tensor* tensor, Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    Tensor* d_tensor = (Tensor*) malloc(sizeof(Tensor));
    Tensor* d_result = (Tensor*) malloc(sizeof(Tensor));
    cudaError_t cuderr = cudaMalloc((void **) &d_tensor->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_tensor->size[0], &tensor->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_tensor->data, tensor->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_tensor->data, tensor->data, tensor->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_result->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_result->size[0], &result->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_result->data, result->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_result->data, result->data, result->size[0] * sizeof(short), cudaMemcpyHostToDevice);

    if (tensor->size[0] / 256 == 0) {
        sum_tensor<<<1, 256>>>(tensor, result);
    } else if (tensor->size[0] / 256 < 8) {
        sum_tensor<<<(tensor->size[0] / 256), 256>>>(tensor, result);
    } else {
        sum_tensor<<<7, 256>>>(tensor, result);
    }
    cuderr = cudaDeviceSynchronize();
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cuderr));
    }
    #endif
    cudaFree(d_tensor->data);
    cudaFree(d_tensor->size);
    cudaFree(d_result->data);
    cudaFree(d_result->size);
}

extern "C" void mean_tensor_wrapper(Tensor* tensor, Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    Tensor* d_tensor = (Tensor*) malloc(sizeof(Tensor));
    Tensor* d_result = (Tensor*) malloc(sizeof(Tensor));
    cudaError_t cuderr = cudaMalloc((void **) &d_tensor->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_tensor->size[0], &tensor->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_tensor->data, tensor->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_tensor->data, tensor->data, tensor->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_result->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_result->size[0], &result->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_result->data, result->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_result->data, result->data, result->size[0] * sizeof(short), cudaMemcpyHostToDevice);

    if (tensor->size[0] / 256 == 0) {
        mean_tensor<<<1, 256>>>(tensor, result);
    } else if (tensor->size[0] / 256 < 8) {
        mean_tensor<<<(tensor->size[0] / 256), 256>>>(tensor, result);
    } else {
        mean_tensor<<<7, 256>>>(tensor, result);
    }
    cuderr = cudaDeviceSynchronize();
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cuderr));
    }
    #endif
    cudaFree(d_tensor->data);
    cudaFree(d_tensor->size);
    cudaFree(d_result->data);
    cudaFree(d_result->size);
}

extern "C" void max_tensor_wrapper(Tensor* tensor, Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    Tensor* d_tensor = (Tensor*) malloc(sizeof(Tensor));
    Tensor* d_result = (Tensor*) malloc(sizeof(Tensor));
    cudaError_t cuderr = cudaMalloc((void **) &d_tensor->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_tensor->size[0], &tensor->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_tensor->data, tensor->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_tensor->data, tensor->data, tensor->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_result->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_result->size[0], &result->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_result->data, result->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_result->data, result->data, result->size[0] * sizeof(short), cudaMemcpyHostToDevice);

    if (tensor->size[0] / 256 == 0) {
        max_tensor<<<1, 256>>>(tensor, result);
    } else if (tensor->size[0] / 256 < 8) {
        max_tensor<<<(tensor->size[0] / 256), 256>>>(tensor, result);
    } else {
        max_tensor<<<7, 256>>>(tensor, result);
    }
    cuderr = cudaDeviceSynchronize();
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cuderr));
    }
    #endif
    cudaFree(d_tensor->data);
    cudaFree(d_tensor->size);
    cudaFree(d_result->data);
    cudaFree(d_result->size);
}

extern "C" void min_tensor_wrapper(Tensor* tensor, Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    Tensor* d_tensor = (Tensor*) malloc(sizeof(Tensor));
    Tensor* d_result = (Tensor*) malloc(sizeof(Tensor));
    cudaError_t cuderr = cudaMalloc((void **) &d_tensor->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_tensor->size[0], &tensor->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_tensor->data, tensor->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_tensor->data, tensor->data, tensor->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_result->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_result->size[0], &result->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_result->data, result->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_result->data, result->data, result->size[0] * sizeof(short), cudaMemcpyHostToDevice);

    if (tensor->size[0] / 256 == 0) {
        min_tensor<<<1, 256>>>(tensor, result);
    } else if (tensor->size[0] / 256 < 8) {
        min_tensor<<<(tensor->size[0] / 256), 256>>>(tensor, result);
    } else {
        min_tensor<<<7, 256>>>(tensor, result);
    }
    cuderr = cudaDeviceSynchronize();
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cuderr));
    }
    #endif
    cudaFree(d_tensor->data);
    cudaFree(d_tensor->size);
    cudaFree(d_result->data);
    cudaFree(d_result->size);
}

extern "C" void gradient_tensor_wrapper(Tensor* tensor, Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    Tensor* d_tensor = (Tensor*) malloc(sizeof(Tensor));
    Tensor* d_result = (Tensor*) malloc(sizeof(Tensor));
    cudaError_t cuderr = cudaMalloc((void **) &d_tensor->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_tensor->size[0], &tensor->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_tensor->data, tensor->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_tensor->data, tensor->data, tensor->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_result->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_result->size[0], &result->size, sizeof(int), cudaMemcpyHostToDevice);
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMemcpy \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMalloc((void **) &d_result->data, result->size[0] * sizeof(short));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(d_result->data, result->data, result->size[0] * sizeof(short), cudaMemcpyHostToDevice);

    if (tensor->size[0] / 256 == 0) {
        gradient_tensor<<<1, 256>>>(tensor, result);
    } else if (tensor->size[0] / 256 < 8) {
        gradient_tensor<<<(tensor->size[0] / 256), 256>>>(tensor, result);
    } else {
        gradient_tensor<<<7, 256>>>(tensor, result);
    }
    cuderr = cudaDeviceSynchronize();
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cuderr));
    }
    #endif
    cudaFree(d_tensor->data);
    cudaFree(d_tensor->size);
    cudaFree(d_result->data);
    cudaFree(d_result->size);
}

extern "C" void gate_tensor_wrapper(Tensor* tensor, Tensor* booleans, Tensor* result) {
    if (!check_size_3(tensor, booleans, result)) {
        return;
    }
    Tensor* d_tensor = (Tensor*) malloc(sizeof(Tensor));
    Tensor* d_booleans = (Tensor*) malloc(sizeof(Tensor));
    Tensor* d_result = (Tensor*) malloc(sizeof(Tensor));
    cudaError_t cuderr = cudaMalloc((void **) &d_tensor->size, sizeof(int));
    #ifdef __DEBUG__
    if (cuderr != cudaSuccess) {
        printf("Error in cudaMalloc \n");
        printf("Error: %s\n", cudaGetErrorString(cuderr));
        return;
    }
    #endif
    cuderr = cudaMemcpy(&d_tensor->size[0], &tensor->size, sizeof(int), cudaMemcpyHostToDevice);
    cuderr = cudaMalloc((void **) &d_booleans->size, sizeof(int));
    cuderr = cudaMemcpy(&d_booleans->size[0], &booleans->size, sizeof(int), cudaMemcpyHostToDevice);
    cuderr = cudaMalloc((void **) &d_result->size, sizeof(int));
    cuderr = cudaMemcpy(&d_result->size[0], &result->size, sizeof(int), cudaMemcpyHostToDevice);
    cuderr = cudaMalloc((void **) &d_tensor->data, tensor->size[0] * sizeof(short));
    cuderr = cudaMemcpy(d_tensor->data, tensor->data, tensor->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    cuderr = cudaMalloc((void **) &d_booleans->data, booleans->size[0] * sizeof(short));
    cuderr = cudaMemcpy(d_booleans->data, booleans->data, booleans->size[0] * sizeof(short), cudaMemcpyHostToDevice);
    cuderr = cudaMalloc((void **) &d_result->data, result->size[0] * sizeof(short));
    cuderr = cudaMemcpy(d_result->data, result->data, result->size[0] * sizeof(short), cudaMemcpyHostToDevice);

    if (tensor->size[0] / 256 == 0) {
        gate_tensor<<<1, 256>>>(tensor, booleans, result);
    } else if (tensor->size[0] / 256 < 8) {
        gate_tensor<<<(tensor->size[0] / 256), 256>>>(tensor, booleans, result);
    } else {
        gate_tensor<<<7, 256>>>(tensor, booleans, result);
    }
    cuderr = cudaDeviceSynchronize();
    cudaFree(d_tensor->data);
    cudaFree(d_tensor->size);
    cudaFree(d_booleans->data);
    cudaFree(d_booleans->size);
    cudaFree(d_result->data);
    cudaFree(d_result->size);
}
