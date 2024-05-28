#include "cuda.h"
#include "cuda_runtime.h"
#include <stdio.h>
#include <math.h>

struct Tensor {
    int size;
    short* data;

};

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
        void dot_tensor(struct Tensor* tensor, struct Tensor* other, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            if (tensor->size != other->size) {
                printf("Error: size mismatch\n");
                return;
            }
            for (int i = index; i < tensor->size; i+= stride) {
                result->data[i] = std::sqrt((tensor->data[i] ^ 2) * (other->data[i] ^ 2));
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
        void abs_tensor(struct Tensor* tensor, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size; i+= stride) {
                result->data[i] = std::abs(tensor->data[i]);
            }
        }

        __global__
        void exp_tensor(struct Tensor* tensor, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size; i+= stride) {
                result->data[i] = std::exp(tensor->data[i]);
            }
        }

        __global__
        void log_tensor(struct Tensor* tensor, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size; i+= stride) {
                result->data[i] = std::log(tensor->data[i]);
            }
        }

        __global__
        void sqrt_tensor(struct Tensor* tensor, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size; i+= stride) {
                result->data[i] = std::sqrt(tensor->data[i]);
            }
        }

        __global__
        void pow_tensor(struct Tensor* tensor, struct Tensor* result, int exponent) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size; i+= stride) {
                result->data[i] = std::pow(tensor->data[i], exponent);
            }
        }

        __global__
        void sin_tensor(struct Tensor* tensor, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size; i+= stride) {
                result->data[i] = std::sin(tensor->data[i]);
            }
        }

        __global__
        void cos_tensor(struct Tensor* tensor, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size; i+= stride) {
                result->data[i] = std::cos(tensor->data[i]);
            }
        }

        __global__
        void tan_tensor(struct Tensor* tensor, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size; i+= stride) {
                result->data[i] = std::tan(tensor->data[i]);
            }
        }

        __global__
        void asin_tensor(struct Tensor* tensor, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size; i+= stride) {
                result->data[i] = std::asin(tensor->data[i]);
            }
        }

        __global__
        void acos_tensor(struct Tensor* tensor, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size; i+= stride) {
                result->data[i] = std::acos(tensor->data[i]);
            }
        }

        __global__
        void atan_tensor(struct Tensor* tensor, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size; i+= stride) {
                result->data[i] = std::atan(tensor->data[i]);
            }
        }

        __global__
        void sinh_tensor(struct Tensor* tensor, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size; i+= stride) {
                result->data[i] = std::sinh(tensor->data[i]);
            }
        }

        __global__
        void cosh_tensor(struct Tensor* tensor, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size; i+= stride) {
                result->data[i] = std::cosh(tensor->data[i]);
            }
        }

        __global__
        void tanh_tensor(struct Tensor* tensor, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size; i+= stride) {
                result->data[i] = std::tanh(tensor->data[i]);
            }
        }

        __global__
        void asinh_tensor(struct Tensor* tensor, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size; i+= stride) {
                result->data[i] = asinh(tensor->data[i]);
            }
        }

        __global__
        void acosh_tensor(struct Tensor* tensor, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size; i+= stride) {
                result->data[i] = acosh(tensor->data[i]);
            }
        }

        __global__
        void atanh_tensor(struct Tensor* tensor, struct Tensor* result) {
            int index = threadIdx.x;
            int stride = blockDim.x;
            for (int i = index; i < tensor->size; i+= stride) {
                result->data[i] = atanh(tensor->data[i]);
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
}

extern "C" void dot_tensor_wrapper(struct Tensor* tensor, struct Tensor* other, struct Tensor* result) {
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
        dot_tensor<<<1, 256>>>(tensor, other, result);
    } else if (tensor->size % 256 < 8) {
        dot_tensor<<<(tensor->size % 256), 256>>>(tensor, other, result);
    } else {
        dot_tensor<<<7, 256>>>(tensor, other, result);
    }
    cudaDeviceSynchronize();
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
}

extern "C" void abs_tensor_wrapper(struct Tensor* tensor, struct Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);

    if (tensor->size % 256 == 0) {
        abs_tensor<<<1, 256>>>(tensor, result);
    } else if (tensor->size % 256 < 8) {
        abs_tensor<<<(tensor->size % 256), 256>>>(tensor, result);
    } else {
        abs_tensor<<<7, 256>>>(tensor, result);
    }
    cudaDeviceSynchronize();
}

extern "C" void exp_tensor_wrapper(struct Tensor* tensor, struct Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);

    if (tensor->size % 256 == 0) {
        exp_tensor<<<1, 256>>>(tensor, result);
    } else if (tensor->size % 256 < 8) {
        exp_tensor<<<(tensor->size % 256), 256>>>(tensor, result);
    } else {
        exp_tensor<<<7, 256>>>(tensor, result);
    }
    cudaDeviceSynchronize();
}

extern "C" void log_tensor_wrapper(struct Tensor* tensor, struct Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);

    if (tensor->size % 256 == 0) {
        log_tensor<<<1, 256>>>(tensor, result);
    } else if (tensor->size % 256 < 8) {
        log_tensor<<<(tensor->size % 256), 256>>>(tensor, result);
    } else {
        log_tensor<<<7, 256>>>(tensor, result);
    }
    cudaDeviceSynchronize();
}

extern "C" void sqrt_tensor_wrapper(struct Tensor* tensor, struct Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);

    if (tensor->size % 256 == 0) {
        sqrt_tensor<<<1, 256>>>(tensor, result);
    } else if (tensor->size % 256 < 8) {
        sqrt_tensor<<<(tensor->size % 256), 256>>>(tensor, result);
    } else {
        sqrt_tensor<<<7, 256>>>(tensor, result);
    }
    cudaDeviceSynchronize();
}

extern "C" void pow_tensor_wrapper(struct Tensor* tensor, int exponent, struct Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);

    if (tensor->size % 256 == 0) {
        pow_tensor<<<1, 256>>>(tensor, result, exponent);
    } else if (tensor->size % 256 < 8) {
        pow_tensor<<<(tensor->size % 256), 256>>>(tensor, result, exponent);
    } else {
        pow_tensor<<<7, 256>>>(tensor, result, exponent);
    }
    cudaDeviceSynchronize();
}

extern "C" void sin_tensor_wrapper(struct Tensor* tensor, struct Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);

    if (tensor->size % 256 == 0) {
        sin_tensor<<<1, 256>>>(tensor, result);
    } else if (tensor->size % 256 < 8) {
        sin_tensor<<<(tensor->size % 256), 256>>>(tensor, result);
    } else {
        sin_tensor<<<7, 256>>>(tensor, result);
    }
    cudaDeviceSynchronize();
}

extern "C" void cos_tensor_wrapper(struct Tensor* tensor, struct Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);

    if (tensor->size % 256 == 0) {
        cos_tensor<<<1, 256>>>(tensor, result);
    } else if (tensor->size % 256 < 8) {
        cos_tensor<<<(tensor->size % 256), 256>>>(tensor, result);
    } else {
        cos_tensor<<<7, 256>>>(tensor, result);
    }
    cudaDeviceSynchronize();
}

extern "C" void tan_tensor_wrapper(struct Tensor* tensor, struct Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);

    if (tensor->size % 256 == 0) {
        tan_tensor<<<1, 256>>>(tensor, result);
    } else if (tensor->size % 256 < 8) {
        tan_tensor<<<(tensor->size % 256), 256>>>(tensor, result);
    } else {
        tan_tensor<<<7, 256>>>(tensor, result);
    }
    cudaDeviceSynchronize();
}

extern "C" void asin_tensor_wrapper(struct Tensor* tensor, struct Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);

    if (tensor->size % 256 == 0) {
        asin_tensor<<<1, 256>>>(tensor, result);
    } else if (tensor->size % 256 < 8) {
        asin_tensor<<<(tensor->size % 256), 256>>>(tensor, result);
    } else {
        asin_tensor<<<7, 256>>>(tensor, result);
    }
    cudaDeviceSynchronize();
}

extern "C" void acos_tensor_wrapper(struct Tensor* tensor, struct Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);

    if (tensor->size % 256 == 0) {
        acos_tensor<<<1, 256>>>(tensor, result);
    } else if (tensor->size % 256 < 8) {
        acos_tensor<<<(tensor->size % 256), 256>>>(tensor, result);
    } else {
        acos_tensor<<<7, 256>>>(tensor, result);
    }
    cudaDeviceSynchronize();
}

extern "C" void atan_tensor_wrapper(struct Tensor* tensor, struct Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);

    if (tensor->size % 256 == 0) {
        atan_tensor<<<1, 256>>>(tensor, result);
    } else if (tensor->size % 256 < 8) {
        atan_tensor<<<(tensor->size % 256), 256>>>(tensor, result);
    } else {
        atan_tensor<<<7, 256>>>(tensor, result);
    }
    cudaDeviceSynchronize();
}

extern "C" void sinh_tensor_wrapper(struct Tensor* tensor, struct Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);

    if (tensor->size % 256 == 0) {
        sinh_tensor<<<1, 256>>>(tensor, result);
    } else if (tensor->size % 256 < 8) {
        sinh_tensor<<<(tensor->size % 256), 256>>>(tensor, result);
    } else {
        sinh_tensor<<<7, 256>>>(tensor, result);
    }
    cudaDeviceSynchronize();
}

extern "C" void cosh_tensor_wrapper(struct Tensor* tensor, struct Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);

    if (tensor->size % 256 == 0) {
        cosh_tensor<<<1, 256>>>(tensor, result);
    } else if (tensor->size % 256 < 8) {
        cosh_tensor<<<(tensor->size % 256), 256>>>(tensor, result);
    } else {
        cosh_tensor<<<7, 256>>>(tensor, result);
    }
    cudaDeviceSynchronize();
}

extern "C" void tanh_tensor_wrapper(struct Tensor* tensor, struct Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);

    if (tensor->size % 256 == 0) {
        tanh_tensor<<<1, 256>>>(tensor, result);
    } else if (tensor->size % 256 < 8) {
        tanh_tensor<<<(tensor->size % 256), 256>>>(tensor, result);
    } else {
        tanh_tensor<<<7, 256>>>(tensor, result);
    }
    cudaDeviceSynchronize();
}

extern "C" void asinh_tensor_wrapper(struct Tensor* tensor, struct Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);

    if (tensor->size % 256 == 0) {
        asinh_tensor<<<1, 256>>>(tensor, result);
    } else if (tensor->size % 256 < 8) {
        asinh_tensor<<<(tensor->size % 256), 256>>>(tensor, result);
    } else {
        asinh_tensor<<<7, 256>>>(tensor, result);
    }
    cudaDeviceSynchronize();
}

extern "C" void acosh_tensor_wrapper(struct Tensor* tensor, struct Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);

    if (tensor->size % 256 == 0) {
        acosh_tensor<<<1, 256>>>(tensor, result);
    } else if (tensor->size % 256 < 8) {
        acosh_tensor<<<(tensor->size % 256), 256>>>(tensor, result);
    } else {
        acosh_tensor<<<7, 256>>>(tensor, result);
    }
    cudaDeviceSynchronize();
}

extern "C" void atanh_tensor_wrapper(struct Tensor* tensor, struct Tensor* result) {
    if (!check_size(tensor, result)) {
        return;
    }
    cudaMalloc(&tensor, sizeof(&tensor));
    cudaMemcpy(tensor, tensor, sizeof(tensor), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(result));
    cudaMemcpy(result, result, sizeof(result), cudaMemcpyHostToDevice);

    if (tensor->size % 256 == 0) {
        atanh_tensor<<<1, 256>>>(tensor, result);
    } else if (tensor->size % 256 < 8) {
        atanh_tensor<<<(tensor->size % 256), 256>>>(tensor, result);
    } else {
        atanh_tensor<<<7, 256>>>(tensor, result);
    }
    cudaDeviceSynchronize();
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
}

