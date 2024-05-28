#include "cuda.h"
#include "cuda_runtime.h"
#include <stdio.h>
#include <math.h>

struct Tensor {
    int size;
    short* data;
};

__global__
void vector_sort_tensor(struct Tensor* tensor, struct Tensor* vectors, struct Tensor* result) {
    if (vectors->size != tensor->size || result->size != tensor->size) {
        printf("Error: index out of bounds\n");
        return; 
    }
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < tensor->size; i+= stride) {
        result->data[i] = tensor->data[vectors->data[i]];
    }

}

__global__
void set_tensor(struct Tensor* tensor, int index, short value) {
    if (index >= tensor->size) {
        printf("Error: index out of bounds\n");
        return;
    }
    tensor->data[index] = value;
}       

        
        void Tensor(struct Tensor* tensor, int size) {
        cudaMallocManaged(&tensor->data, size * sizeof(short));
        }

        
        void rm_Tensor(struct Tensor* tensor) {
            cudaFree(&tensor->data);
        }

        __global__
        void print_tensor(struct Tensor* tensor) {

            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size; i+= stride) {
                printf("%d ", tensor->data[i]);
            }
            printf("\n");
        }

        __global__
        void fill_tensor(struct Tensor* tensor, int value) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size; i+= stride) {
                tensor->data[i] = value;
            }
        }

        __global__
        void add_tensor(struct Tensor* tensor, struct Tensor* other, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            if (tensor->size != other->size) {
                printf("Error: size mismatch\n");
                return;
            }
            for (int i = index; i < tensor->size; i+= stride) {
                result->data[i] = tensor->data[i] + other->data[i];
            }
        }

        __global__
        void sub_tensor(struct Tensor* tensor, struct Tensor* other, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            if (tensor->size != other->size) {
                printf("Error: size mismatch\n");
                return;
            }
            for (int i = index; i < tensor->size; i+= stride) {
                result->data[i] = tensor->data[i] - other->data[i];
            }
        }

        __global__
        void mul_tensor(struct Tensor* tensor, struct Tensor* other, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            if (tensor->size != other->size) {
                printf("Error: size mismatch\n");
                return;
            }
            for (int i = index; i < tensor->size; i+= stride) {
                result->data[i] = tensor->data[i] * other->data[i];
            }
        }

        __global__
        void div_tensor(struct Tensor* tensor, struct Tensor* other, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            if (tensor->size != other->size) {
                printf("Error: size mismatch\n");
                return;
            }
            for (int i = index; i < tensor->size; i+= stride) {
                result->data[i] = tensor->data[i] / other->data[i];
            }
        }

        __global__
        void add_scalar_tensor(struct Tensor* tensor, int value, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size; i+= stride) {
                result->data[i] = tensor->data[i] + value;
            }
        }

        __global__
        void sub_scalar_tensor(struct Tensor* tensor, int value, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size; i+= stride) {
                result->data[i] = tensor->data[i] - value;
            }
        }

        __global__
        void mul_scalar_tensor(struct Tensor* tensor, int value, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size; i+= stride) {
                result->data[i] = tensor->data[i] * value;
            }
        }

        __global__
        void div_scalar_tensor(struct Tensor* tensor, int value, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size; i+= stride) {
                result->data[i] = tensor->data[i] / value;
            }
        }

        __global__
        void transpose_tensor(struct Tensor* tensor, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size; i+= stride) {
                result->data[i] = tensor->data[tensor->size - i - 1];
            }
        }

        __global__
        void sum_tensor(struct Tensor* tensor, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size; i+= stride) {
                result->data[0] += tensor->data[i];
            }
        }

        __global__
        void mean_tensor(struct Tensor* tensor, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size; i+= stride) {
                result->data[0] += tensor->data[i];
            }
            result->data[0] /= tensor->size;
        }

        __global__
        void max_tensor(struct Tensor* tensor, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size; i+= stride) {
                if (tensor->data[i] > result->data[0]) {
                    result->data[0] = tensor->data[i];
                }
            }
        }

        __global__
        void min_tensor(struct Tensor* tensor, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size; i+= stride) {
                if (tensor->data[i] < result->data[0]) {
                    result->data[0] = tensor->data[i];
                }
            }
        }

        __global__
        void gradient_tensor(struct Tensor* tensor, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size; i+= stride) {
                result->data[i] = tensor->data[i + 1] - tensor->data[i];
            }
        }

        __global__
        void gate_tensor(struct Tensor* tensor, struct Tensor* bools, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size; i+= stride) {
                if (bools->data[i] == 1) {
                    result->data[i] = tensor->data[i];
                }
            }
        }

        bool check_size(struct Tensor* tensor, struct Tensor* other) {
            if (tensor->size != other->size) {
                printf("Error: size mismatch\n");
                return false;
            }
            if (sizeof(&tensor->data) != sizeof(&other->data)) {
                printf("Error: size mismatch\n");
                return false;
            }
            return true;
        }

        bool check_size(struct Tensor* tensor, struct Tensor* other, struct Tensor* result) {
            if (tensor->size != other->size || tensor->size != result->size) {
                printf("Error: size mismatch\n");
                return false;
            }
            if (sizeof(&tensor->data) != sizeof(&other->data) || sizeof(&tensor->data) != sizeof(&result->data)) {
                printf("Error: size mismatch\n");
                return false;
            }
            return true;
        }

extern "C" void vector_sort_tensor_wrapper(struct Tensor* tensor, struct Tensor* vectors, struct Tensor* result) {
    if (!check_size(tensor, vectors, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&vectors, sizeof(vectors));
    cudaMemcpy(vectors, vectors, sizeof(vectors), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);
    if (tensor->size % 256 == 0) {
        vector_sort_tensor<<<1, 256>>>(tensor, vectors, result);
    } else if (tensor->size % 256 < 8) {
        vector_sort_tensor<<<(tensor->size % 256), 256>>>(tensor, vectors, result);
    } else {
        vector_sort_tensor<<<7, 256>>>(tensor, vectors, result);
    }
    cudaDeviceSynchronize();
    cudaFree(tensor);
    cudaFree(vectors);
    cudaFree(result);
}

extern "C" void print_tensor_wrapper(struct Tensor* tensor) {
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    if (tensor->size % 256 == 0) {
        print_tensor<<<1, 256>>>(tensor);
    } else if (tensor->size % 256 < 8) {
        print_tensor<<<(tensor->size % 256), 256>>>(tensor);
    } else {
        print_tensor<<<7, 256>>>(tensor);
    }
    cudaDeviceSynchronize();
    cudaFree(tensor);
}

extern "C" void fill_tensor_wrapper(struct Tensor* tensor, int value) {
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    if (tensor->size % 256 == 0) {
        fill_tensor<<<1, 256>>>(tensor, value);
    } else if (tensor->size % 256 < 8) {
        fill_tensor<<<(tensor->size % 256), 256>>>(tensor, value);
    } else {
        fill_tensor<<<7, 256>>>(tensor, value);
    }
    cudaDeviceSynchronize();
    cudaFree(tensor);
}

extern "C" void add_tensor_wrapper(struct Tensor* tensor, struct Tensor* other, struct Tensor* result) {
    if (!check_size(tensor, other, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&other, sizeof(other));
    cudaMemcpy(other, other, sizeof(other), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);
    if (tensor->size % 256 == 0) {
        add_tensor<<<1, 256>>>(tensor, other, result);
    } else if (tensor->size % 256 < 8) {
        add_tensor<<<(tensor->size % 256), 256>>>(tensor, other, result);
    } else {
        add_tensor<<<7, 256>>>(tensor, other, result);
    }

    cudaDeviceSynchronize();
    cudaFree(tensor);
    cudaFree(other);
    cudaFree(result);
}

extern "C" void sub_tensor_wrapper(struct Tensor* tensor, struct Tensor* other, struct Tensor* result) {
    if (!check_size(tensor, other, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&other, sizeof(other));
    cudaMemcpy(other, other, sizeof(other), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);
if (tensor->size % 256 == 0) {
        sub_tensor<<<1, 256>>>(tensor, other, result);
    } else if (tensor->size % 256 < 8) {
        sub_tensor<<<(tensor->size % 256), 256>>>(tensor, other, result);
    } else {
        sub_tensor<<<7, 256>>>(tensor, other, result);
    }

    cudaDeviceSynchronize();
    cudaFree(tensor);
    cudaFree(other);
    cudaFree(result);
}

extern "C" void mul_tensor_wrapper(struct Tensor* tensor, struct Tensor* other, struct Tensor* result) {
    if (!check_size(tensor, other, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&other, sizeof(other));
    cudaMemcpy(other, other, sizeof(other), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);
if (tensor->size % 256 == 0) {
        mul_tensor<<<1, 256>>>(tensor, other, result);
    } else if (tensor->size % 256 < 8) {
        mul_tensor<<<(tensor->size % 256), 256>>>(tensor, other, result);
    } else {
        mul_tensor<<<7, 256>>>(tensor, other, result);
    }

    cudaDeviceSynchronize();
    cudaFree(tensor);
    cudaFree(other);
    cudaFree(result);
}

extern "C" void div_tensor_wrapper(struct Tensor* tensor, struct Tensor* other, struct Tensor* result) {
    if (!check_size(tensor, other, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&other, sizeof(other));
    cudaMemcpy(other, other, sizeof(other), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);
if (tensor->size % 256 == 0) {
        div_tensor<<<1, 256>>>(tensor, other, result);
    } else if (tensor->size % 256 < 8) {
        div_tensor<<<(tensor->size % 256), 256>>>(tensor, other, result);
    } else {
        div_tensor<<<7, 256>>>(tensor, other, result);
    }

    cudaDeviceSynchronize();
    cudaFree(tensor);
    cudaFree(other);
    cudaFree(result);
}

extern "C" void add_scalar_tensor_wrapper(struct Tensor* tensor, int value, struct Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);
    
    if (tensor->size % 256 == 0) {
        add_scalar_tensor<<<1, 256>>>(tensor, value, result);
    } else if (tensor->size % 256 < 8) {
        add_scalar_tensor<<<(tensor->size % 256), 256>>>(tensor, value, result);
    } else {
        add_scalar_tensor<<<7, 256>>>(tensor, value, result);
    }
    cudaDeviceSynchronize();
    cudaFree(tensor);
    cudaFree(result);
}

extern "C" void sub_scalar_tensor_wrapper(struct Tensor* tensor, int value, struct Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);

    if (tensor->size % 256 == 0) {
        sub_scalar_tensor<<<1, 256>>>(tensor, value, result);
    } else if (tensor->size % 256 < 8) {
        sub_scalar_tensor<<<(tensor->size % 256), 256>>>(tensor, value, result);
    } else {
        sub_scalar_tensor<<<7, 256>>>(tensor, value, result);
    }
    cudaDeviceSynchronize();
    cudaFree(tensor);
    cudaFree(result);
}

extern "C" void mul_scalar_tensor_wrapper(struct Tensor* tensor, int value, struct Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);

    if (tensor->size % 256 == 0) {
        mul_scalar_tensor<<<1, 256>>>(tensor, value, result);
    } else if (tensor->size % 256 < 8) {
        mul_scalar_tensor<<<(tensor->size % 256), 256>>>(tensor, value, result);
    } else {
        mul_scalar_tensor<<<7, 256>>>(tensor, value, result);
    }
    cudaDeviceSynchronize();
    cudaFree(tensor);
    cudaFree(result);
}

extern "C" void div_scalar_tensor_wrapper(struct Tensor* tensor, int value, struct Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);

    if (tensor->size % 256 == 0) {
        div_scalar_tensor<<<1, 256>>>(tensor, value, result);
    } else if (tensor->size % 256 < 8) {
        div_scalar_tensor<<<(tensor->size % 256), 256>>>(tensor, value, result);
    } else {
        div_scalar_tensor<<<7, 256>>>(tensor, value, result);
    }
    cudaDeviceSynchronize();
    cudaFree(tensor);
    cudaFree(result);
}

extern "C" void transpose_tensor_wrapper(struct Tensor* tensor, struct Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);
    transpose_tensor<<<1, 256>>>(tensor, result);

    if (tensor->size % 256 == 0) {
        transpose_tensor<<<1, 256>>>(tensor, result);
    } else if (tensor->size % 256 < 8) {
        transpose_tensor<<<(tensor->size % 256), 256>>>(tensor, result);
    } else {
        transpose_tensor<<<7, 256>>>(tensor, result);
    }
    cudaDeviceSynchronize();
    cudaFree(tensor);
    cudaFree(result);
}

extern "C" void sum_tensor_wrapper(struct Tensor* tensor, struct Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);
    sum_tensor<<<1, 256>>>(tensor, result);

    if (tensor->size % 256 == 0) {
        sum_tensor<<<1, 256>>>(tensor, result);
    } else if (tensor->size % 256 < 8) {
        sum_tensor<<<(tensor->size % 256), 256>>>(tensor, result);
    } else {
        sum_tensor<<<7, 256>>>(tensor, result);
    }
    cudaDeviceSynchronize();
    cudaFree(tensor);
    cudaFree(result);
}

extern "C" void max_tensor_wrapper(struct Tensor* tensor, struct Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);

    if (tensor->size % 256 == 0) {
        max_tensor<<<1, 256>>>(tensor, result);
    } else if (tensor->size % 256 < 8) {
        max_tensor<<<(tensor->size % 256), 256>>>(tensor, result);
    } else {
        max_tensor<<<7, 256>>>(tensor, result);
    }
    cudaDeviceSynchronize();
    cudaFree(tensor);
    cudaFree(result);
}

extern "C" void min_tensor_wrapper(struct Tensor* tensor, struct Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);

    if (tensor->size % 256 == 0) {
        min_tensor<<<1, 256>>>(tensor, result);
    } else if (tensor->size % 256 < 8) {
        min_tensor<<<(tensor->size % 256), 256>>>(tensor, result);
    } else {
        min_tensor<<<7, 256>>>(tensor, result);
    }
    cudaDeviceSynchronize();
    cudaFree(tensor);
    cudaFree(result);
}

extern "C" void gradient_tensor_wrapper(struct Tensor* tensor, struct Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);

    if (tensor->size % 256 == 0) {
        gradient_tensor<<<1, 256>>>(tensor, result);
    } else if (tensor->size % 256 < 8) {
        gradient_tensor<<<(tensor->size % 256), 256>>>(tensor, result);
    } else {
        gradient_tensor<<<7, 256>>>(tensor, result);
    }
    cudaDeviceSynchronize();
    cudaFree(tensor);
    cudaFree(result);
}

extern "C" void gate_tensor_wrapper(struct Tensor* tensor, struct Tensor* booleans, struct Tensor* result) {
    if (!check_size(tensor, booleans, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&booleans, sizeof(booleans));
    cudaMemcpy(booleans, booleans, sizeof(booleans), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);

    if (tensor->size % 256 == 0) {
        gate_tensor<<<1, 256>>>(tensor, booleans, result);
    } else if (tensor->size % 256 < 8) {
        gate_tensor<<<(tensor->size % 256), 256>>>(tensor, booleans, result);
    } else {
        gate_tensor<<<7, 256>>>(tensor, booleans, result);
    }
    cudaDeviceSynchronize();
    cudaFree(tensor);
    cudaFree(booleans);
    cudaFree(result);
}

