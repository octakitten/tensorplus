#pragma once
#include "tensor.cu"

#include <stdio.h>




    void Tensor_multi(struct Tensor_multi* tensor, int n, int m, int k) {
        // Allocate memory for the tensor
        if (n <= 0 || m <= 0 || k <= 0) {
            printf("Invalid dimensions for tensor");
            exit(1);
        }
        tensor->dimx = n;
        tensor->dimy = m;
        tensor->dimz = k;
        tensor->size = n * m * k;
        cudaMallocManaged(&tensor->data, n * m * k * sizeof(short));
    }


    void rm_Tensor_multi(struct Tensor_multi* tensor) {
        // Free memory
        cudaFree(&tensor->data);
    }

    short& tensor_at(struct Tensor_multi* tensor, int i, int j, int k) {
        if (i < 0 || i >= tensor->dimx || j < 0 || j >= tensor->dimy || k < 0 || k >= tensor->dimz) {
            cerr << "Index out of bounds" << endl;
            exit(1);
        }
        return tensor->data[i * tensor->dimy * tensor->dimz + j * tensor->dimz + k];
    }
/**
void fill_multi(struct Tensor_multi* tensor, short value, struct Tensor_multi* result) {};
void print_multi(struct Tensor_multi* tensor);
void add_multi(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2, struct Tensor_multi* result);
void sub_multi(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2, struct Tensor_multi* result);
void mul_multi(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2, struct Tensor_multi* result);
void div_multi(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2, struct Tensor_multi* result);
void add_scalar_multi(struct Tensor_multi* tensor, short scalar, struct Tensor_multi* result);
void sub_scalar_multi(struct Tensor_multi* tensor, short scalar, struct Tensor_multi* result);
void mul_scalar_multi(struct Tensor_multi* tensor, short scalar, struct Tensor_multi* result);
void div_scalar_multi(struct Tensor_multi* tensor, short scalar, struct Tensor_multi* result);
void add_multi_inplace(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2);
void sub_multi_inplace(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2);
void mul_multi_inplace(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2);
void div_multi_inplace(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2);
void add_scalar_multi_inplace(struct Tensor_multi* tensor, short scalar);
void sub_scalar_multi_inplace(struct Tensor_multi* tensor, short scalar);
void mul_scalar_multi_inplace(struct Tensor_multi* tensor, short scalar);
void div_scalar_multi_inplace(struct Tensor_multi* tensor, short scalar);
void fill_multi_inplace(struct Tensor_multi* tensor, short value);
void print_multi_inplace(struct Tensor_multi* tensor);
void copy_multi(struct Tensor_multi* tensor, struct Tensor_multi* result);
void copy_multi_inplace(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2);
void transpose_multi(struct Tensor_multi* tensor, struct Tensor_multi* result);
void transpose_multi_inplace(struct Tensor_multi* tensor);
void reshape_multi(struct Tensor_multi* tensor, int n, int m, int k, struct Tensor_multi* result);
void reshape_multi_inplace(struct Tensor_multi* tensor, int n, int m, int k);
void slice_multi(struct Tensor_multi* tensor, int startx, int endx, int starty, int endy, int startz, int endz, struct Tensor_multi* result);
void slice_multi_inplace(struct Tensor_multi* tensor, int startx, int endx, int starty, int endy, int startz, int endz);
void concat_multi(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2, int axis, struct Tensor_multi* result);
void concat_multi_inplace(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2, int axis);
void split_multi(struct Tensor_multi* tensor, int axis, struct Tensor_multi* result1, struct Tensor_multi* result2);
void split_multi_inplace(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2, int axis);
void max_multi(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2, struct Tensor_multi* result);
void min_multi(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2, struct Tensor_multi* result);
void max_multi_inplace(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2);
void min_multi_inplace(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2);
void abs_multi(struct Tensor_multi* tensor, struct Tensor_multi* result);
void abs_multi_inplace(struct Tensor_multi* tensor);
void sum_multi(struct Tensor_multi* tensor, int axis, struct Tensor_multi* result);
void sum_multi_inplace(struct Tensor_multi* tensor, int axis);
void mean_multi(struct Tensor_multi* tensor, int axis, struct Tensor_multi* result);
void mean_multi_inplace(struct Tensor_multi* tensor, int axis);
void dot_multi(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2, struct Tensor_multi* result);
void dot_multi_inplace(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2);
void conv2d_multi(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2, struct Tensor_multi* result);
void conv2d_multi_inplace(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2);
void relu_multi(struct Tensor_multi* tensor, struct Tensor_multi* result);
void relu_multi_inplace(struct Tensor_multi* tensor);
void sigmoid_multi(struct Tensor_multi* tensor, struct Tensor_multi* result);
void sigmoid_multi_inplace(struct Tensor_multi* tensor);
void softmax_multi(struct Tensor_multi* tensor, struct Tensor_multi* result);
void softmax_multi_inplace(struct Tensor_multi* tensor);
void sin_multi(struct Tensor_multi* tensor, struct Tensor_multi* result);
void sin_multi_inplace(struct Tensor_multi* tensor);
void cos_multi(struct Tensor_multi* tensor, struct Tensor_multi* result);
void cos_multi_inplace(struct Tensor_multi* tensor);
void tan_multi(struct Tensor_multi* tensor, struct Tensor_multi* result);
void tan_multi_inplace(struct Tensor_multi* tensor);
void sinh_multi(struct Tensor_multi* tensor, struct Tensor_multi* result);
void sinh_multi_inplace(struct Tensor_multi* tensor);
void cosh_multi(struct Tensor_multi* tensor, struct Tensor_multi* result);
void cosh_multi_inplace(struct Tensor_multi* tensor);
void tanh_multi(struct Tensor_multi* tensor, struct Tensor_multi* result);
void tanh_multi_inplace(struct Tensor_multi* tensor)
void asin_multi(struct Tensor_multi* tensor, struct Tensor_multi* result);
void asin_multi_inplace(struct Tensor_multi* tensor);
void acos_multi(struct Tensor_multi* tensor, struct Tensor_multi* result);
void acos_multi_inplace(struct Tensor_multi* tensor);
void atan_multi(struct Tensor_multi* tensor, struct Tensor_multi* result);
void atan_multi_inplace(struct Tensor_multi* tensor);
void asinh_multi(struct Tensor_multi* tensor, struct Tensor_multi* result);
void asinh_multi_inplace(struct Tensor_multi* tensor);
void acosh_multi(struct Tensor_multi* tensor, struct Tensor_multi* result);
void acosh_multi_inplace(struct Tensor_multi* tensor);
void atanh_multi(struct Tensor_multi* tensor, struct Tensor_multi* result);
void atanh_multi_inplace(struct Tensor_multi* tensor);
void exp_multi(struct Tensor_multi* tensor, struct Tensor_multi* result);
void exp_multi_inplace(struct Tensor_multi* tensor);
void log_multi(struct Tensor_multi* tensor, struct Tensor_multi* result);
void log_multi_inplace(struct Tensor_multi* tensor);
void sqrt_multi(struct Tensor_multi* tensor, struct Tensor_multi* result);
void sqrt_multi_inplace(struct Tensor_multi* tensor);
void pow_multi(struct Tensor_multi* tensor, short exp, struct Tensor_multi* result);
void pow_multi_inplace(struct Tensor_multi* tensor, short exp);
void gate_multi(struct Tensor_multi* tensor, struct Tensor_multi* booleans, struct Tensor_multi* result);
void gate_multi_inplace(struct Tensor_multi* tensor, struct Tensor_multi* booleans);
**/


void fill_multi(struct Tensor_multi* tensor, short value, struct Tensor_multi* result) {
    for (int i = 0; i < tensor->dimx; i++) {
        for (int j = 0; j < tensor->dimy; j++) {
            for (int k = 0; k < tensor->dimz; k++) {
                tensor_at(tensor, i, j, k) = value;
            }
        }
    }
}

void print_multi(struct Tensor_multi* tensor) {
    for (int i = 0; i < tensor->dimx; i++) {
        for (int j = 0; j < tensor->dimy; j++) {
            for (int k = 0; k < tensor->dimz; k++) {
                cout << tensor_at(tensor, i, j, k) << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
}

void add_multi(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2, struct Tensor_multi* result) {
    for (int i = 0; i < tensor1->dimx; i++) {
        for (int j = 0; j < tensor1->dimy; j++) {
            for (int k = 0; k < tensor1->dimz; k++) {
                tensor_at(result, i, j, k) = tensor_at(tensor1, i, j, k) + tensor_at(tensor2, i, j, k);
            }
        }
    }
}

void sub_multi(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2, struct Tensor_multi* result) {
    for (int i = 0; i < tensor1->dimx; i++) {
        for (int j = 0; j < tensor1->dimy; j++) {
            for (int k = 0; k < tensor1->dimz; k++) {
                tensor_at(result, i, j, k) = tensor_at(tensor1, i, j, k) - tensor_at(tensor2, i, j, k);
            }
        }
    }
}

void mul_multi(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2, struct Tensor_multi* result) {
    for (int i = 0; i < tensor1->dimx; i++) {
        for (int j = 0; j < tensor1->dimy; j++) {
            for (int k = 0; k < tensor1->dimz; k++) {
                tensor_at(result, i, j, k) = tensor_at(tensor1, i, j, k) * tensor_at(tensor2, i, j, k);
            }
        }
    }
}

void div_multi(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2, struct Tensor_multi* result) {
    for (int i = 0; i < tensor1->dimx; i++) {
        for (int j = 0; j < tensor1->dimy; j++) {
            for (int k = 0; k < tensor1->dimz; k++) {
                tensor_at(result, i, j, k) = tensor_at(tensor1, i, j, k) / tensor_at(tensor2, i, j, k);
            }
        }
    }
}

void add_scalar_multi(struct Tensor_multi* tensor, short scalar, struct Tensor_multi* result) {
    for (int i = 0; i < tensor->dimx; i++) {
        for (int j = 0; j < tensor->dimy; j++) {
            for (int k = 0; k < tensor->dimz; k++) {
                tensor_at(result, i, j, k) = tensor_at(tensor, i, j, k) + scalar;
            }
        }
    }
}

void sub_scalar_multi(struct Tensor_multi* tensor, short scalar, struct Tensor_multi* result) {
    for (int i = 0; i < tensor->dimx; i++) {
        for (int j = 0; j < tensor->dimy; j++) {
            for (int k = 0; k < tensor->dimz; k++) {
                tensor_at(result, i, j, k) = tensor_at(tensor, i, j, k) - scalar;
            }
        }
    }
}

void mul_scalar_multi(struct Tensor_multi* tensor, short scalar, struct Tensor_multi* result) {
    for (int i = 0; i < tensor->dimx; i++) {
        for (int j = 0; j < tensor->dimy; j++) {
            for (int k = 0; k < tensor->dimz; k++) {
                tensor_at(result, i, j, k) = tensor_at(tensor, i, j, k) * scalar;
            }
        }
    }
}

void div_scalar_multi(struct Tensor_multi* tensor, short scalar, struct Tensor_multi* result) {
    for (int i = 0; i < tensor->dimx; i++) {
        for (int j = 0; j < tensor->dimy; j++) {
            for (int k = 0; k < tensor->dimz; k++) {
                tensor_at(result, i, j, k) = tensor_at(tensor, i, j, k) / scalar;
            }
        }
    }
}

void add_multi_inplace(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2) {
    for (int i = 0; i < tensor1->dimx; i++) {
        for (int j = 0; j < tensor1->dimy; j++) {
            for (int k = 0; k < tensor1->dimz; k++) {
                tensor_at(tensor1, i, j, k) += tensor_at(tensor2, i, j, k);
            }
        }
    }
}

void sub_multi_inplace(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2) {
    for (int i = 0; i < tensor1->dimx; i++) {
        for (int j = 0; j < tensor1->dimy; j++) {
            for (int k = 0; k < tensor1->dimz; k++) {
                tensor_at(tensor1, i, j, k) -= tensor_at(tensor2, i, j, k);
            }
        }
    }
}

void mul_multi_inplace(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2) {
    for (int i = 0; i < tensor1->dimx; i++) {
        for (int j = 0; j < tensor1->dimy; j++) {
            for (int k = 0; k < tensor1->dimz; k++) {
                tensor_at(tensor1, i, j, k) *= tensor_at(tensor2, i, j, k);
            }
        }
    }
}

void div_multi_inplace(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2) {
    for (int i = 0; i < tensor1->dimx; i++) {
        for (int j = 0; j < tensor1->dimy; j++) {
            for (int k = 0; k < tensor1->dimz; k++) {
                tensor_at(tensor1, i, j, k) /= tensor_at(tensor2, i, j, k);
            }
        }
    }
}

void add_scalar_multi_inplace(struct Tensor_multi* tensor, short scalar) {
    for (int i = 0; i < tensor->dimx; i++) {
        for (int j = 0; j < tensor->dimy; j++) {
            for (int k = 0; k < tensor->dimz; k++) {
                tensor_at(tensor, i, j, k) += scalar;
            }
        }
    }
}

void sub_scalar_multi_inplace(struct Tensor_multi* tensor, short scalar) {
    for (int i = 0; i < tensor->dimx; i++) {
        for (int j = 0; j < tensor->dimy; j++) {
            for (int k = 0; k < tensor->dimz; k++) {
                tensor_at(tensor, i, j, k) -= scalar;
            }
        }
    }
}

void mul_scalar_multi_inplace(struct Tensor_multi* tensor, short scalar) {
    for (int i = 0; i < tensor->dimx; i++) {
        for (int j = 0; j < tensor->dimy; j++) {
            for (int k = 0; k < tensor->dimz; k++) {
                tensor_at(tensor, i, j, k) *= scalar;
            }
        }
    }
}

void div_scalar_multi_inplace(struct Tensor_multi* tensor, short scalar) {
    for (int i = 0; i < tensor->dimx; i++) {
        for (int j = 0; j < tensor->dimy; j++) {
            for (int k = 0; k < tensor->dimz; k++) {
                tensor_at(tensor, i, j, k) /= scalar;
            }
        }
    }
}

void fill_multi_inplace(struct Tensor_multi* tensor, short value) {
    for (int i = 0; i < tensor->dimx; i++) {
        for (int j = 0; j < tensor->dimy; j++) {
            for (int k = 0; k < tensor->dimz; k++) {
                tensor_at(tensor, i, j, k) = value;
            }
        }
    }
}

void print_multi_inplace(struct Tensor_multi* tensor) {
    for (int i = 0; i < tensor->dimx; i++) {
        for (int j = 0; j < tensor->dimy; j++) {
            for (int k = 0; k < tensor->dimz; k++) {
                cout << tensor_at(tensor, i, j, k) << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
}

void copy_multi(struct Tensor_multi* tensor, struct Tensor_multi* result) {
    for (int i = 0; i < tensor->dimx; i++) {
        for (int j = 0; j < tensor->dimy; j++) {
            for (int k = 0; k < tensor->dimz; k++) {
                tensor_at(result, i, j, k) = tensor_at(tensor, i, j, k);
            }
        }
    }
}

void copy_multi_inplace(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2) {
    for (int i = 0; i < tensor1->dimx; i++) {
        for (int j = 0; j < tensor1->dimy; j++) {
            for (int k = 0; k < tensor1->dimz; k++) {
                tensor_at(tensor1, i, j, k) = tensor_at(tensor2, i, j, k);
            }
        }
    }
}

void transpose_multi(struct Tensor_multi* tensor, struct Tensor_multi* result) {
    for (int i = 0; i < tensor->dimx; i++) {
        for (int j = 0; j < tensor->dimy; j++) {
            for (int k = 0; k < tensor->dimz; k++) {
                tensor_at(result, j, i, k) = tensor_at(tensor, i, j, k);
            }
        }
    }
}

void transpose_multi_inplace(struct Tensor_multi* tensor) {
    short* temp = new short[tensor->dimx * tensor->dimy * tensor->dimz];
    for (int i = 0; i < tensor->dimx; i++) {
        for (int j = 0; j < tensor->dimy; j++) {
            for (int k = 0; k < tensor->dimz; k++) {
                temp[j * tensor->dimx * tensor->dimz + i * tensor->dimz + k] = tensor_at(tensor, i, j, k);
            }
        }
    }
    delete[] tensor->data;
    tensor->data = temp;
    int temp_dim = tensor->dimx;
    tensor->dimx = tensor->dimy;
    tensor->dimy = temp_dim;
}

void reshape_multi(struct Tensor_multi* tensor, int n, int m, int k, struct Tensor_multi* result) {
    if (n * m * k != tensor->dimx * tensor->dimy * tensor->dimz) {
        cerr << "Invalid dimensions for reshape" << endl;
        exit(1);
    }
    result->dimx = n;
    result->dimy = m;
    result->dimz = k;
    result->data = tensor->data;
}

void reshape_multi_inplace(struct Tensor_multi* tensor, int n, int m, int k) {
    if (n * m * k != tensor->dimx * tensor->dimy * tensor->dimz) {
        cerr << "Invalid dimensions for reshape" << endl;
        exit(1);
    }
    tensor->dimx = n;
    tensor->dimy = m;
    tensor->dimz = k;
}

void slice_multi(struct Tensor_multi* tensor, int startx, int endx, int starty, int endy, int startz, int endz, struct Tensor_multi* result) {
    if (startx < 0 || startx >= tensor->dimx || endx < 0 || endx >= tensor->dimx || startx > endx ||
        starty < 0 || starty >= tensor->dimy || endy < 0 || endy >= tensor->dimy || starty > endy ||
        startz < 0 || startz >= tensor->dimz || endz < 0 || endz >= tensor->dimz || startz > endz) {
        cerr << "Invalid slice dimensions" << endl;
        exit(1);
    }
    result->dimx = endx - startx + 1;
    result->dimy = endy - starty + 1;
    result->dimz = endz - startz + 1;
    result->data = tensor->data + startx * tensor->dimy * tensor->dimz + starty * tensor->dimz + startz;
}

void slice_multi_inplace(struct Tensor_multi* tensor, int startx, int endx, int starty, int endy, int startz, int endz) {
    if (startx < 0 || startx >= tensor->dimx || endx < 0 || endx >= tensor->dimx || startx > endx ||
        starty < 0 || starty >= tensor->dimy || endy < 0 || endy >= tensor->dimy || starty > endy ||
        startz < 0 || startz >= tensor->dimz || endz < 0 || endz >= tensor->dimz || startz > endz) {
        cerr << "Invalid slice dimensions" << endl;
        exit(1);
    }
    tensor->dimx = endx - startx + 1;
    tensor->dimy = endy - starty + 1;
    tensor->dimz = endz - startz + 1;
    tensor->data = tensor->data + startx * tensor->dimy * tensor->dimz + starty * tensor->dimz + startz;
}

void concat_multi(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2, int axis, struct Tensor_multi* result) {
    if (axis == 0) {
        if (tensor1->dimy != tensor2->dimy || tensor1->dimz != tensor2->dimz) {
            cerr << "Invalid dimensions for concat" << endl;
            exit(1);
        }
        result->dimx = tensor1->dimx + tensor2->dimx;
        result->dimy = tensor1->dimy;
        result->dimz = tensor1->dimz;
        result->data = new short[result->dimx * result->dimy * result->dimz];
        for (int i = 0; i < tensor1->dimx; i++) {
            for (int j = 0; j < tensor1->dimy; j++) {
                for (int k = 0; k < tensor1->dimz; k++) {
                    tensor_at(result, i, j, k) = tensor_at(tensor1, i, j, k);
                }
            }
        }
        for (int i = 0; i < tensor2->dimx; i++) {
            for (int j = 0; j < tensor2->dimy; j++) {
                for (int k = 0; k < tensor2->dimz; k++) {
                    tensor_at(result, i + tensor1->dimx, j, k) = tensor_at(tensor2, i, j, k);
                }
            }
        }
    } else if (axis == 1) {
        if (tensor1->dimx != tensor2->dimx || tensor1->dimz != tensor2->dimz) {
            cerr << "Invalid dimensions for concat" << endl;
            exit(1);
        }
        result->dimx = tensor1->dimx;
        result->dimy = tensor1->dimy + tensor2->dimy;
        result->dimz = tensor1->dimz;
        result->data = new short[result->dimx * result->dimy * result->dimz];
        for (int i = 0; i < tensor1->dimx; i++) {
            for (int j = 0; j < tensor1->dimy; j++) {
                for (int k =
                        0; k < tensor1->dimz; k++) {
                        tensor_at(result, i, j, k) = tensor_at(tensor1, i, j, k);
                    }
                }
            }
            for (int i = 0; i < tensor2->dimx; i++) {
                for (int j = 0; j < tensor2->dimy; j++) {
                    for (int k = 0; k < tensor2->dimz; k++) {
                        tensor_at(result, i, j + tensor1->dimy, k) = tensor_at(tensor2, i, j, k);
                    }
                }
            }
        } else if (axis == 2) {
            if (tensor1->dimx != tensor2->dimx || tensor1->dimy != tensor2->dimy) {
                cerr << "Invalid dimensions for concat" << endl;
                exit(1);
            }
            result->dimx = tensor1->dimx;
            result->dimy = tensor1->dimy;
            result->dimz = tensor1->dimz + tensor2->dimz;
            result->data = new short[result->dimx * result->dimy * result->dimz];
            for (int i = 0; i < tensor1->dimx; i++) {
                for (int j = 0; j < tensor1->dimy; j++) {
                    for (int k = 0; k < tensor1->dimz; k++) {
                        tensor_at(result, i, j, k) = tensor_at(tensor1, i, j, k);
                    }
                }
            }
            for (int i = 0; i < tensor2->dimx; i++) {
                for (int j = 0; j < tensor2->dimy; j++) {
                    for (int k = 0; k < tensor2->dimz; k++) {
                        tensor_at(result, i, j, k + tensor1->dimz) = tensor_at(tensor2, i, j, k);
                    }
                }
            }
        } else {
            cerr << "Invalid axis for concat" << endl;
            exit(1);
        }
    }

    void concat_multi_inplace(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2, int axis) {
        if (axis == 0) {
            if (tensor1->dimy != tensor2->dimy || tensor1->dimz != tensor2->dimz) {
                cerr << "Invalid dimensions for concat" << endl;
                exit(1);
            }
            short* temp = new short[(tensor1->dimx + tensor2->dimx) * tensor1->dimy * tensor1->dimz];
            for (int i = 0; i < tensor1->dimx; i++) {
                for (int j = 0; j < tensor1->dimy; j++) {
                    for (int k = 0; k < tensor1->dimz; k++) {
                        temp[i * tensor1->dimy * tensor1->dimz + j * tensor1->dimz + k] = tensor_at(tensor1, i, j, k);
                    }
                }
            }
            for (int i = 0; i < tensor2->dimx; i++) {
                for (int j = 0; j < tensor2->dimy; j++) {
                    for (int k = 0; k < tensor2->dimz; k++) {
                        temp[(i + tensor1->dimx) * tensor1->dimy * tensor1->dimz + j * tensor1->dimz + k] = tensor_at(tensor2, i, j, k);
                    }
                }
            }
            delete[] tensor1->data;
            tensor1->data = temp;
            tensor1->dimx += tensor2->dimx;
        } else if (axis == 1) {
            if (tensor1->dimx != tensor2->dimx || tensor1->dimz != tensor2->dimz) {
                cerr << "Invalid dimensions for concat" << endl;
                exit(1);
            }
            short* temp = new short[tensor1->dimx * (tensor1->dimy + tensor2->dimy) * tensor1->dimz];
            for (int i = 0; i < tensor1->dimx; i++) {
                for (int j = 0; j < tensor1->dimy; j++) {
                    for (int k = 0; k < tensor1->dimz; k++) {
                        temp[i * (tensor1->dimy + tensor2->dimy) * tensor1->dimz + j * tensor1->dimz + k] = tensor_at(tensor1, i, j, k);
                    }
                }
            }
            for (int i = 0; i < tensor2->dimx; i++) {
                for (int j = 0; j < tensor2->dimy; j++) {
                    for (int k = 0; k < tensor2->dimz; k++) {
                        temp[i * (tensor1->dimy + tensor2->dimy) * tensor1->dimz + (j + tensor1->dimy) * tensor1->dimz + k] = tensor_at(tensor2, i, j, k);
                    }
                }
            }
            delete[] tensor1->data;
            tensor1->data = temp;
            tensor1->dimy += tensor2->dimy;
        } else if (axis == 2) {
            if (tensor1->dimx != tensor2->dimx || tensor1->dimy != tensor2->dimy) {
                cerr << "Invalid dimensions for concat" << endl;
                exit(1);
            }
            short* temp = new short[tensor1->dimx * tensor1->dimy * (tensor1->dimz + tensor2->dimz)];
            for (int i = 0; i < tensor1->dimx; i++) {
                for (int j = 0; j < tensor1->dimy; j++) {
                    for (int k = 0; k < tensor1->dimz; k++) {
                        temp[i * tensor1->dimy * (tensor1->dimz + tensor2->dimz) + j * (tensor1->dimz + tensor2->dimz) + k] = tensor_at(tensor1, i, j, k);
                    }
                }
            }
            for (int i = 0; i < tensor2->dimx; i++) {
                for (int j = 0; j < tensor2->dimy; j++) {
                    for (int k = 0; k < tensor2->dimz; k++) {
                        temp[i * tensor1->dimy * (tensor1->dimz + tensor2->dimz) + j * (tensor1->dimz + tensor2->dimz) + (k + tensor1->dimz)] = tensor_at(tensor2, i, j, k);
                    }
                }
            }
            delete[] tensor1->data;
            tensor1->data = temp;
            tensor1->dimz += tensor2->dimz;
        } else {
            cerr << "Invalid axis for concat" << endl;
            exit(1);
        }
    }

    void split_multi(struct Tensor_multi* tensor, int axis, struct Tensor_multi* result1, struct Tensor_multi* result2) {
        if (axis == 0) {
            result1->dimx = tensor->dimx / 2;
            result1->dimy = tensor->dimy;
            result1->dimz = tensor->dimz;
            result1->data = tensor->data;
            result2->dimx = tensor->dimx - tensor->dimx / 2;
            result2->dimy = tensor->dimy;
            result2->dimz = tensor->dimz;
            result2->data = tensor->data + tensor->dimx / 2 * tensor->dimy * tensor->dimz;
        } else if (axis == 1) {
            result1->dimx = tensor->dimx;
            result1->dimy = tensor->dimy / 2;
            result1->dimz = tensor->dimz;
            result1->data = tensor->data;
            result2->dimx = tensor->dimx;
            result2->dimy = tensor->dimy - tensor->dimy / 2;
            result2->dimz = tensor->dimz;
            result2->data = tensor->data + tensor->dimx * tensor->dimy / 2 * tensor->dimz;
        } else if (axis == 2) {
            result1->dimx = tensor->dimx;
            result1->dimy = tensor->dimy;
            result1->dimz = tensor->dimz / 2;
            result1->data = tensor->data;
            result2->dimx = tensor->dimx;
            result2->dimy = tensor->dimy;
            result2->dimz = tensor->dimz - tensor->dimz / 2;
            result2->data = tensor->data + tensor->dimx * tensor->dimy * tensor->dimz / 2;
        } else {
            cerr << "Invalid axis for split" << endl;
            exit(1);
        }
    }

    void split_multi_inplace(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2, int axis) {
        if (axis == 0) {
            tensor1->dimx = tensor1->dimx / 2;
            tensor2->dimx = tensor2->dimx - tensor1->dimx;
            tensor2->data = tensor1->data + tensor1->dimx * tensor1->dimy * tensor1->dimz;
        } else if (axis == 1) {
            tensor1->dimy = tensor1->dimy / 2;
            tensor2->dimy = tensor2->dimy - tensor1->dimy;
            tensor2->data = tensor1->data + tensor1->dimx * tensor1->dimy * tensor1->dimz;
        } else if (axis == 2) {
            tensor1->dimz = tensor1->dimz / 2;
            tensor2->dimz = tensor2->dimz - tensor1->dimz;
            tensor2->data = tensor1->data + tensor1->dimx * tensor1->dimy * tensor1->dimz;
        } else {
            cerr << "Invalid axis for split" << endl;
            exit(1);
        }
    }

    void max_multi(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2, struct Tensor_multi* result) {
        for (int i = 0; i < tensor1->dimx; i++) {
            for (int j = 0; j < tensor1->dimy; j++) {
                for (int k = 0; k < tensor1->dimz; k++) {
                    tensor_at(result, i, j, k) = tensor_at(tensor1, i, j, k) > tensor_at(tensor2, i, j, k) ? tensor_at(tensor1, i, j, k) : tensor_at(tensor2, i, j, k);
                }
            }
        }
    }

    void min_multi(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2, struct Tensor_multi* result) {
        for (int i = 0; i < tensor1->dimx; i++) {
            for (int j = 0; j < tensor1->dimy; j++) {
                for (int k = 0; k < tensor1->dimz; k++) {
                    tensor_at(result, i, j, k) = tensor_at(tensor1, i, j, k) < tensor_at(tensor2, i, j, k) ? tensor_at(tensor1, i, j, k) : tensor_at(tensor2, i, j, k);
                }
            }
        }
    }

    void max_multi_inplace(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2) {
        for (int i = 0; i < tensor1->dimx; i++) {
            for (int j = 0; j < tensor1->dimy; j++) {
                for (int k = 0; k < tensor1->dimz; k++) {
                    tensor_at(tensor1, i, j, k) = tensor_at(tensor1, i, j, k) > tensor_at(tensor2, i, j, k) ? tensor_at(tensor1, i, j, k) : tensor_at(tensor2, i, j, k);
                }
            }
        }
    }

    void min_multi_inplace(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2) {
        for (int i = 0; i < tensor1->dimx; i++) {
            for (int j = 0; j < tensor1->dimy; j++) {
                for (int k = 0; k < tensor1->dimz; k++) {
                    tensor_at(tensor1, i, j, k) = tensor_at(tensor1, i, j, k) < tensor_at(tensor2, i, j, k) ? tensor_at(tensor1, i, j, k) : tensor_at(tensor2, i, j, k);
                }
            }
        }
    }

    void abs_multi(struct Tensor_multi* tensor, struct Tensor_multi* result) {
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    tensor_at(result, i, j, k) = tensor_at(tensor, i, j, k) < 0 ? -tensor_at(tensor, i, j, k) : tensor_at(tensor, i, j, k);
                }
            }
        }
    }

    void abs_multi_inplace(struct Tensor_multi* tensor) {
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    tensor_at(tensor, i, j, k) = tensor_at(tensor, i, j, k) < 0 ? -tensor_at(tensor, i, j, k) : tensor_at(tensor, i, j, k);
                }
            }
        }
    }

    void sum_multi(struct Tensor_multi* tensor, int axis, struct Tensor_multi* result) {
        if (axis == 0) {
            result->dimx = 1;
            result->dimy = tensor->dimy;
            result->dimz = tensor->dimz;
            result->data = new short[result->dimy * result->dimz];
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    tensor_at(result, 0, j, k) = 0;
                    for (int i = 0; i < tensor->dimx; i++) {
                        tensor_at(result, 0, j, k) += tensor_at(tensor, i, j, k);
                    }
                }
            }
        } else if (axis == 1) {
            result->dimx = tensor->dimx;
            result->dimy = 1;
            result->dimz = tensor->dimz;
            result->data = new short[result->dimx * result->dimz];
            for (int i = 0; i < tensor->dimx; i++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    tensor_at(result, i, 0, k) = 0;
                    for (int j = 0; j < tensor->dimy; j++) {
                        tensor_at(result, i, 0, k) += tensor_at(tensor, i, j, k);
                    }
                }
            }
        } else if (axis == 2) {
            result->dimx = tensor->dimx;
            result->dimy = tensor->dimy;
            result->dimz = 1;
            result->data = new short[result->dimx * result->dimy];
            for (int i = 0; i < tensor->dimx; i++) {
                for (int j = 0; j < tensor->dimy; j++) {
                    tensor_at(result, i, j, 0) = 0;
                    for (int k = 0; k < tensor->dimz; k++) {
                        tensor_at(result, i, j, 0) += tensor_at(tensor, i, j, k);
                    }
                }
            }
        } else {
            cerr << "Invalid axis for sum" << endl;
            exit(1);
        }
    }

    void sum_multi_inplace(struct Tensor_multi* tensor, int axis) {
        if (axis == 0) {
            short* temp = new short[tensor->dimy * tensor->dimz];
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    temp[j * tensor->dimz + k] = 0;
                    for (int i = 0; i < tensor->dimx; i++) {
                        temp[j * tensor->dimz + k] += tensor_at(tensor, i, j, k);
                    }
                }
            }
            delete[] tensor->data;
            tensor->data = temp;
            tensor->dimx = 1;
        } else if (axis == 1) {
            short* temp = new short[tensor->dimx * tensor->dimz];
            for (int i = 0; i < tensor->dimx; i++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    temp[i * tensor->dimz + k] = 0;
                    for (int j = 0; j < tensor->dimy; j++) {
                        temp[i * tensor->dimz + k] += tensor_at(tensor, i, j, k);
                    }
                }
            }
            delete[] tensor->data;
            tensor->data = temp;
            tensor->dimy = 1;
        } else if (axis == 2) {
            short* temp = new short[tensor->dimx * tensor->dimy];
            for (int i = 0; i < tensor->dimx; i++) {
                for (int j = 0; j < tensor->dimy; j++) {
                    temp[i * tensor->dimy + j] = 0;
                    for (int k = 0; k < tensor->dimz; k++) {
                        temp[i * tensor->dimy + j] += tensor_at(tensor, i, j, k);
                    }
                }
            }
            delete[] tensor->data;
            tensor->data = temp;
            tensor->dimz = 1;
        } else {
            cerr << "Invalid axis for sum" << endl;
            exit(1);
        }
    }

    void mean_multi(struct Tensor_multi* tensor, int axis, struct Tensor_multi* result) {
        sum_multi(tensor, axis, result);
        int size = result->dimx * result->dimy * result->dimz;
        for (int i = 0; i < size; i++) {
            result->data[i] /= tensor->dimx * tensor->dimy * tensor->dimz;
        }
    }

    void mean_multi_inplace(struct Tensor_multi* tensor, int axis) {
        sum_multi_inplace(tensor, axis);
        int size = tensor->dimx * tensor->dimy * tensor->dimz;
        for (int i = 0; i < size; i++) {
            tensor->data[i] /= tensor->dimx * tensor->dimy * tensor->dimz;
        }
    }

    void dot_multi(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2, struct Tensor_multi* result) {
        if (tensor1->dimy != tensor2->dimx) {
            cerr << "Invalid dimensions for dot" << endl;
            exit(1);
        }
        result->dimx = tensor1->dimx;
        result->dimy = tensor2->dimy;
        result->dimz = 1;
        result->data = new short[result->dimx * result->dimy];
        for (int i = 0; i < tensor1->dimx; i++) {
            for (int j = 0; j < tensor2->dimy; j++) {
                tensor_at(result, i, j, 0) = 0;
                for (int k = 0; k < tensor1->dimy; k++) {
                    tensor_at(result, i, j, 0) += tensor_at(tensor1, i, k, 0) * tensor_at(tensor2, k, j, 0);
                }
            }
        }
    }

    void dot_multi_inplace(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2) {
        if (tensor1->dimy != tensor2->dimx) {
            cerr << "Invalid dimensions for dot" << endl;
            exit(1);
        }
        short* temp = new short[tensor1->dimx * tensor2->dimy];
        for (int i = 0; i < tensor1->dimx; i++) {
            for (int j = 0; j < tensor2->dimy; j++) {
                temp[i * tensor2->dimy + j] = 0;
                for (int k = 0; k < tensor1->dimy; k++) {
                    temp[i * tensor2->dimy + j] += tensor_at(tensor1, i, k, 0) * tensor_at(tensor2, k, j, 0);
                }
            }
        }
        delete[] tensor1->data;
        tensor1->data = temp;
        tensor1->dimy = tensor2->dimy;
    }

    void conv2d_multi(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2, struct Tensor_multi* result, int stride, int padding) {
        if (tensor2->dimz != tensor1->dimz) {
            cerr << "Invalid dimensions for conv2d" << endl;
            exit(1);
        }
        int outx = (tensor1->dimx - tensor2->dimx + 2 * padding) / stride + 1;
        int outy = (tensor1->dimy - tensor2->dimy + 2 * padding) / stride + 1;
        result->dimx = outx;
        result->dimy = outy;
        result->dimz = tensor2->dimy;
        result->data = new short[outx * outy * tensor2->dimy];
        for (int i = 0; i < outx; i++) {
            for (int j = 0; j < outy; j++) {
                for (int k = 0; k < tensor2->dimy; k++) {
                    tensor_at(result, i, j, k) = 0;
                    for (int l = 0; l < tensor2->dimx; l++) {
                        for (int m = 0; m < tensor2->dimy; m++) {
                            for (int n = 0; n < tensor2->dimz; n++) {
                                if (i * stride + l - padding >= 0 && i * stride + l - padding < tensor2->dimx &&
                                    j * stride + m - padding >= 0 && j * stride + m - padding < tensor2->dimy) {
                                    tensor_at(result, i, j, k) += tensor_at(tensor1, i * stride + l - padding, j * stride + m - padding, n) *
                                                          tensor_at(tensor2, l, m, n);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void conv2d_multi_inplace(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2, int stride, int padding) {
        if (tensor2->dimz != tensor1->dimz) {
            cerr << "Invalid dimensions for conv2d" << endl;
            exit(1);
        }
        int outx = (tensor1->dimx - tensor2->dimx + 2 * padding) / stride + 1;
        int outy = (tensor1->dimy - tensor2->dimy + 2 * padding) / stride + 1;
        short* temp = new short[outx * outy * tensor2->dimy];
        for (int i = 0; i < outx; i++) {
            for (int j = 0; j < outy; j++) {
                for (int k = 0; k < tensor2->dimy; k++) {
                    temp[i * outy * tensor2->dimy + j * tensor2->dimy + k] = 0;
                    for (int l = 0; l < tensor2->dimx; l++) {
                        for (int m = 0; m < tensor2->dimy; m++) {
                            for (int n = 0; n < tensor2->dimz; n++) {
                                if (i * stride + l - padding >= 0 && i * stride + l - padding < tensor2->dimx &&
                                    j * stride + m - padding >= 0 && j * stride + m - padding < tensor2->dimy) {
                                    temp[i * outy * tensor2->dimy + j * tensor2->dimy + k] += tensor_at(tensor1, i * stride + l - padding, j * stride + m - padding, n) *
                                                                                           tensor_at(tensor2, l, m, n);
                                }
                            }
                        }
                    }
                }
            }
        }
        delete[] tensor1->data;
        tensor1->data = temp;
        tensor1->dimx = outx;
        tensor1->dimy = outy;
        tensor1->dimz = tensor2->dimy;
    }

    void relu_multi(struct Tensor_multi* tensor, struct Tensor_multi* result) {
        copy_multi(tensor, result);
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    tensor_at(result, i, j, k) = tensor_at(result, i, j, k) < 0 ? 0 : tensor_at(result, i, j, k);
                }
            }
        }
    }

    void relu_multi_inplace(struct Tensor_multi* tensor) {
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    tensor_at(tensor, i, j, k) = tensor_at(tensor, i, j, k) < 0 ? 0 : tensor_at(tensor, i, j, k);
                }
            }
        }
    }

    void softmax_multi(struct Tensor_multi* tensor, struct Tensor_multi* result) {
        copy_multi(tensor, result);
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                short max_val = tensor_at(result, i, j, 0);
                for (int k = 1; k < tensor->dimz; k++) {
                    max_val = tensor_at(result, i, j, k) > max_val ? tensor_at(result, i, j, k) : max_val;
                }
                short sum = 0;
                for (int k = 0; k < tensor->dimz; k++) {
                    tensor_at(result, i, j, k) = exp(tensor_at(result, i, j, k) - max_val);
                    sum += tensor_at(result, i, j, k);
                }
                for (int k = 0; k < tensor->dimz; k++) {
                    tensor_at(result, i, j, k) /= sum;
                }
            }
        }
    }

    void softmax_multi_inplace(struct Tensor_multi* tensor) {
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                short max_val = tensor_at(tensor, i, j, 0);
                for (int k = 1; k < tensor->dimz; k++) {
                    max_val = tensor_at(tensor, i, j, k) > max_val ? tensor_at(tensor, i, j, k) : max_val;
                }
                short sum = 0;
                for (int k = 0; k < tensor->dimz; k++) {
                    tensor_at(tensor, i, j, k) = exp(tensor_at(tensor, i, j, k) - max_val);
                    sum += tensor_at(tensor, i, j, k);
                }
                for (int k = 0; k < tensor->dimz; k++) {
                    tensor_at(tensor, i, j, k) /= sum;
                }
            }
        }
    }

    void sigmoid_multi(struct Tensor_multi* tensor, struct Tensor_multi* result) {
        copy_multi(tensor, result);
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    tensor_at(result, i, j, k) = 1 / (1 + exp(-tensor_at(result, i, j, k)));
                }
            }
        }
    }

    void sigmoid_multi_inplace(struct Tensor_multi* tensor) {
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    tensor_at(tensor, i, j, k) = 1 / (1 + exp(-tensor_at(tensor, i, j, k)));
                }
            }
        }
    }

    void sin_multi(struct Tensor_multi* tensor, struct Tensor_multi* result) {
        copy_multi(tensor, result);
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    tensor_at(result, i, j, k) = sin(tensor_at(result, i, j, k));
                }
            }
        }
    }

    void sin_multi_inplace(struct Tensor_multi* tensor) {
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    tensor_at(tensor, i, j, k) = sin(tensor_at(tensor, i, j, k));
                }
            }
        }
    }

    void cos_multi(struct Tensor_multi* tensor, struct Tensor_multi* result) {
        copy_multi(tensor, result);
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    tensor_at(result, i, j, k) = cos(tensor_at(result, i, j, k));
                }
            }
        }
    }

    void cos_multi_inplace(struct Tensor_multi* tensor) {
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    tensor_at(tensor, i, j, k) = cos(tensor_at(tensor, i, j, k));
                }
            }
        }
    }

    void tan_multi(struct Tensor_multi* tensor, struct Tensor_multi* result) {
        copy_multi(tensor, result);
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    tensor_at(result, i, j, k) = tan(tensor_at(result, i, j, k));
                }
            }
        }
    }

    void tan_multi_inplace(struct Tensor_multi* tensor) {
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    tensor_at(tensor, i, j, k) = tan(tensor_at(tensor, i, j, k));
                }
            }
        }
    }

    void sinh_multi(struct Tensor_multi* tensor, struct Tensor_multi* result) {
        copy_multi(tensor, result);
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    tensor_at(result, i, j, k) = sinh(tensor_at(result, i, j, k));
                }
            }
        }
    }

    void sinh_multi_inplace(struct Tensor_multi* tensor) {
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    tensor_at(tensor, i, j, k) = sinh(tensor_at(tensor, i, j, k));
                }
            }
        }
    }

    void cosh_multi(struct Tensor_multi* tensor, struct Tensor_multi* result) {
        copy_multi(tensor, result);
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    tensor_at(result, i, j, k) = cosh(tensor_at(result, i, j, k));
                }
            }
        }
    }

    void cosh_multi_inplace(struct Tensor_multi* tensor) {
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    tensor_at(tensor, i, j, k) = cosh(tensor_at(tensor, i, j, k));
                }
            }
        }
    }

    void asin_multi(struct Tensor_multi* tensor, struct Tensor_multi* result) {
        copy_multi(tensor, result);
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    if (tensor_at(result, i, j, k) < -1 || tensor_at(result, i, j, k) > 1) {
                        cerr << "Invalid input for asin" << endl;
                        exit(1);
                    }
                    tensor_at(result, i, j, k) = asin(tensor_at(result, i, j, k));
                }
            }
        }
    }

    void asin_multi_inplace(struct Tensor_multi* tensor) {
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    if (tensor_at(tensor, i, j, k) < -1 || tensor_at(tensor, i, j, k) > 1) {
                        cerr << "Invalid input for asin" << endl;
                        exit(1);
                    }
                    tensor_at(tensor, i, j, k) = asin(tensor_at(tensor, i, j, k));
                }
            }
        }
    }

    void acos_multi(struct Tensor_multi* tensor, struct Tensor_multi* result) {
        copy_multi(tensor, result);
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    if (tensor_at(result, i, j, k) < -1 || tensor_at(result, i, j, k) > 1) {
                        cerr << "Invalid input for acos" << endl;
                        exit(1);
                    }
                    tensor_at(result, i, j, k) = acos(tensor_at(result, i, j, k));
                }
            }
        }
    }

    void acos_multi_inplace(struct Tensor_multi* tensor) {
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    if (tensor_at(tensor, i, j, k) < -1 || tensor_at(tensor, i, j, k) > 1) {
                        cerr << "Invalid input for acos" << endl;
                        exit(1);
                    }
                    tensor_at(tensor, i, j, k) = acos(tensor_at(tensor, i, j, k));
                }
            }
        }
    }

    void atan_multi(struct Tensor_multi* tensor, struct Tensor_multi* result) {
        copy_multi(tensor, result);
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    tensor_at(result, i, j, k) = atan(tensor_at(result, i, j, k));
                }
            }
        }
    }

    void atan_multi_inplace(struct Tensor_multi* tensor) {
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    tensor_at(tensor, i, j, k) = atan(tensor_at(tensor, i, j, k));
                }
            }
        }
    }

    void tanh_multi(struct Tensor_multi* tensor, struct Tensor_multi* result) {
        copy_multi(tensor, result);
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    tensor_at(result, i, j, k) = tanh(tensor_at(result, i, j, k));
                }
            }
        }
    }

    void tanh_multi_inplace(struct Tensor_multi* tensor) {
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    tensor_at(tensor, i, j, k) = tanh(tensor_at(tensor, i, j, k));
                }
            }
        }
    }

    void asinh_multi(struct Tensor_multi* tensor, struct Tensor_multi* result) {
        copy_multi(tensor, result);
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    tensor_at(result, i, j, k) = asinh(tensor_at(result, i, j, k));
                }
            }
        }
    }

    void asinh_multi_inplace(struct Tensor_multi* tensor) {
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    tensor_at(tensor, i, j, k) = asinh(tensor_at(tensor, i, j, k));
                }
            }
        }
    }

    void acosh_multi(struct Tensor_multi* tensor, struct Tensor_multi* result) {
        copy_multi(tensor, result);
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    if (tensor_at(result, i, j, k) < 1) {
                        cerr << "Invalid input for acosh" << endl;
                        exit(1);
                    }
                    tensor_at(result, i, j, k) = acosh(tensor_at(result, i, j, k));
                }
            }
        }
    }

    void acosh_multi_inplace(struct Tensor_multi* tensor) {
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    if (tensor_at(tensor, i, j, k) < 1) {
                        cerr << "Invalid input for acosh" << endl;
                        exit(1);
                    }
                    tensor_at(tensor, i, j, k) = acosh(tensor_at(tensor, i, j, k));
                }
            }
        }
    }

    void atanh_multi(struct Tensor_multi* tensor, struct Tensor_multi* result) {
        copy_multi(tensor, result);
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    if (tensor_at(result, i, j, k) <= -1 || tensor_at(result, i, j, k) >= 1) {
                        cerr << "Invalid input for atanh" << endl;
                        exit(1);
                    }
                    tensor_at(result, i, j, k) = atanh(tensor_at(result, i, j, k));
                }
            }
        }
    }

    void atanh_multi_inplace(struct Tensor_multi* tensor) {
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    if (tensor_at(tensor, i, j, k) <= -1 || tensor_at(tensor, i, j, k) >= 1) {
                        cerr << "Invalid input for atanh" << endl;
                        exit(1);
                    }
                    tensor_at(tensor, i, j, k) = atanh(tensor_at(tensor, i, j, k));
                }
            }
        }
    }



    void sqrt_multi(struct Tensor_multi* tensor, struct Tensor_multi* result) {
        copy_multi(tensor, result);
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    if (tensor_at(result, i, j, k) < 0) {
                        cerr << "Invalid input for sqrt" << endl;
                        exit(1);
                    }
                    tensor_at(result, i, j, k) = sqrt(tensor_at(result, i, j, k));
                }
            }
        }
    }

    void sqrt_multi_inplace(struct Tensor_multi* tensor) {
        for (int i = 0; i < tensor->dimx; i++) {
            for (int j = 0; j < tensor->dimy; j++) {
                for (int k = 0; k < tensor->dimz; k++) {
                    if (tensor_at(tensor, i, j, k) < 0) {
                        cerr << "Invalid input for sqrt" << endl;
                        exit(1);
                    }
                    tensor_at(tensor, i, j, k) = sqrt(tensor_at(tensor, i, j, k));
                }
            }
        }
    }

    void pow_multi(struct Tensor_multi* tensor, short exponent, struct Tensor_multi* result) {
        copy_multi(tensor, result);
        if (exponent == 0) {
            for (int i = 0; i < tensor->dimx * tensor->dimy * tensor->dimz; i++) {
                result->data[i] = 1;
            }
        } else if (exponent == 1) {
            return;
        } else if (exponent == 2) {
            for (int i = 0; i < tensor->dimx * tensor->dimy * tensor->dimz; i++) {
                result->data[i] *= result->data[i];
            }
        } else {
            for (int i = 0; i < tensor->dimx * tensor->dimy * tensor->dimz; i++) {
                result->data[i] = pow(result->data[i], exponent);
            }
        }
    }

    void pow_multi_inplace(struct Tensor_multi* tensor, short exponent) {
        if (exponent == 0) {
            for (int i = 0; i < tensor->dimx * tensor->dimy * tensor->dimz; i++) {
                tensor->data[i] = 1;
            }
        } else if (exponent == 1) {
            return;
        } else if (exponent == 2) {
            for (int i = 0; i < tensor->dimx * tensor->dimy * tensor->dimz; i++) {
                tensor->data[i] *= tensor->data[i];
            }
        } else {
            for (int i = 0; i < tensor->dimx * tensor->dimy * tensor->dimz; i++) {
                tensor->data[i] = pow(tensor->data[i], exponent);
            }
        }
    }

    void log_multi(struct Tensor_multi* tensor, struct Tensor_multi* result) {
        copy_multi(tensor, result);
        for (int i = 0; i < tensor->dimx * tensor->dimy * tensor->dimz; i++) {
            if (result->data[i] <= 0) {
                cerr << "Invalid input for log" << endl;
                exit(1);
            }
            result->data[i] = log(result->data[i]);
        }
    }

    void log_multi_inplace(struct Tensor_multi* tensor) {
        for (int i = 0; i < tensor->dimx * tensor->dimy * tensor->dimz; i++) {
            if (tensor->data[i] <= 0) {
                cerr << "Invalid input for log" << endl;
                exit(1);
            }
            tensor->data[i] = log(tensor->data[i]);
        }
    }

    void exp_multi(struct Tensor_multi* tensor, struct Tensor_multi* result) {
        copy_multi(tensor, result);
        for (int i = 0; i < tensor->dimx * tensor->dimy * tensor->dimz; i++) {
            result->data[i] = exp(result->data[i]);
        }
    }

    void exp_multi_inplace(struct Tensor_multi* tensor) {
        for (int i = 0; i < tensor->dimx * tensor->dimy * tensor->dimz; i++) {
            tensor->data[i] = exp(tensor->data[i]);
        }
    }

    void gate_multi(struct Tensor_multi* tensor, struct Tensor_multi* booleans, struct Tensor_multi* result) {
        copy_multi(tensor, result);
        for (int i = 0; i < tensor->dimx * tensor->dimy * tensor->dimz; i++) {
            result->data[i] = booleans->data[i] ? result->data[i] : 0;
        }
    }

    void gate_multi_inplace(struct Tensor_multi* tensor, struct Tensor_multi* booleans) {
        for (int i = 0; i < tensor->dimx * tensor->dimy * tensor->dimz; i++) {
            tensor->data[i] = booleans->data[i] ? tensor->data[i] : 0;
        }
    }

    extern "C" void gate_multi_wrapper(struct Tensor_multi* tensor, struct Tensor_multi* booleans, struct Tensor_multi* result) {
        gate_multi(tensor, booleans, result);
    }

    extern "C" void gate_multi_inplace_wrapper(struct Tensor_multi* tensor, struct Tensor_multi* booleans) {
        gate_multi_inplace(tensor, booleans);
    }

    extern "C" void add_multi_wrapper(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2, struct Tensor_multi* result) {
        add_multi(tensor1, tensor2, result);
    }

    extern "C" void add_multi_inplace_wrapper(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2) {
        add_multi_inplace(tensor1, tensor2);
    }

    extern "C" void sub_multi_wrapper(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2, struct Tensor_multi* result) {
        sub_multi(tensor1, tensor2, result);
    }

    extern "C" void sub_multi_inplace_wrapper(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2) {
        sub_multi_inplace(tensor1, tensor2);
    }

    extern "C" void mul_multi_wrapper(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2, struct Tensor_multi* result) {
        mul_multi(tensor1, tensor2, result);
    }

    extern "C" void mul_multi_inplace_wrapper(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2) {
        mul_multi_inplace(tensor1, tensor2);
    }

    extern "C" void div_multi_wrapper(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2, struct Tensor_multi* result) {
        div_multi(tensor1, tensor2, result);
    }

    extern "C" void div_multi_inplace_wrapper(struct Tensor_multi* tensor1, struct Tensor_multi* tensor2) {
        div_multi_inplace(tensor1, tensor2);
    }

    extern "C" void sum_multi_wrapper(struct Tensor_multi* tensor, int axis, struct Tensor_multi* result) {
        sum_multi(tensor, axis, result);
    }

    extern "C" void sum_multi_inplace_wrapper(struct Tensor_multi* tensor, int axis) {
        sum_multi_inplace(tensor, axis);
    }

    
