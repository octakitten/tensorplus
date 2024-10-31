#pragma once
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "tensor.h"

/*
   Here, we're going to define some functions to handle dealing with multidimensional tensors.
   The idea here is that handling the actual data will be done by the cuda kernel functions we already defined in
   tensor.cu, while the functions here will simple arrange the data-holding tensors and match them up to
   where they need to go.

   This means that we'll be defining higher level wrapper functions around the basic functions provided 
   in tensor.cu that adapt based on the dimensions that the overall tensor is supposed to have. If the user
   calls a function that, for instance, adds two 3 dimensional tensors together, we'll first follow the 
   TensorMulti to one of its elements, follow the TensorMulti thats held in that element to one of its elements,
   which is a Tensor of one type or another, and then do the same for the other 3D tensor. Then we'll call the 
   appropriate function from tensor.cu using these two data tensors we picked out, and repeat that process
   element-wise through the whole 3D tensor.

   And of course, we'll need to create functions to handle the creation and destruction of these TensorMulti,
   as well as appropriate functions for every operation supported by the underlying Tensor type.

   These TensorMulti structures are also able to adapt to whatever datatype the underlying data tensor has, so
   since we're using a void pointer to hold the actual data of the tensor, we need to cast that data to the 
   appropriate struct type before we use it. We'll make a function to handle that in this file.
*/

// function to create a TensorMulti
TensorMulti* create_tensor_short_multi(int numDims, int* dims, int dataType) {
    // allocate memory for the tensor_short_multi
    TensorMulti* tensormu = (TensorMulti*)malloc(sizeof(TensorMulti));
    // allocate memory for the dims array
    tensormu->dims = (int*)malloc((1 + numDims) * sizeof(int));
    //set the first element of the dims array to be the number of dims
    tensormu->dims[0] = numDims;
    //set the rest of the dims array
    for (int i = 0; i < numDims; i++) {
        tensormu->dims[i + 1] = dims[i];
    }
    // set the data type
    tensormu->dataType = &dataType;
    // here we make a distinction between each of the 6 potential data types
    // and create the appropriate tensor for each one
    if (numDims == 1) {
        if (dataType == 0) {
            tensormu->data = create_tensor_short(dims[0]);
        }
        if (dataType == 1) {
            tensormu->data = create_tensor_int(dims[0]);
        }
        if (dataType == 2) {
            tensormu->data = create_tensor_float(dims[0]);
        }
        if (dataType == 3) {
            tensormu->data = create_tensor_double(dims[0]);
        }
        if (dataType == 4) {
            tensormu->data = create_tensor_long(dims[0]);
        }
    if (numDims > 1) {
        tensormu->data = create_tensor_short_multi(numDims - 1, dims[1], dataType);
    }
    return tensormu;
}

// function to destroy a TensorMulti
void destroy_tensor_short_multi(TensorMulti* tensormu) {
    // if the tensormu has dimensions greater than 1, we need to 
    // iterate through the data and destroy each tensor in it
    if (*(tensormu->dims[0]) > 1) {
        for (int i = 0; i < tensormu->dims[1]; i++) {
            destroy_tensor_short_multi((TensorMulti*)tensormu->data[i]);
        }
    }
    // otherwise if the remaining dims is just 1, we destroy the data tensor
    else {
        if (*(tensormu->dataType) == 0) {
            destroy_tensor_short((Tensor*)tensormu->data);
        }
        if (*(tensormu->dataType) == 1) {
            destroy_tensor_int((TensorInt*)tensormu->data);
        }
        if (*(tensormu->dataType) == 2) {
            destroy_tensor_float((TensorFloat*)tensormu->data);
        }
        if (*(tensormu->dataType) == 3) {
            destroy_tensor_double((TensorDouble*)tensormu->data);
        }
        if (*(tensormu->dataType) == 4) {
            destroy_tensor_long((TensorLong*)tensormu->data);
        }    
    }
    // free the dims array
    free(tensormu->dims);
    // free the tensor_short_multi
    free(tensormu);
}


// function to create a TensorMulti which puts its data on the device
TensorMulti* create_tensor_short_multi_device(int numDims, int* dims, int dataType) {
    // allocate memory for the tensor_short_multi
    TensorMulti* tensormu = (TensorMulti*)malloc(sizeof(TensorMulti));
    // allocate memory for the dims array
    tensormu->dims = (int*)malloc((1 + numDims) * sizeof(int));
    //set the first element of the dims array to be the number of dims
    tensormu->dims[0] = numDims;
    //set the rest of the dims array
    for (int i = 0; i < numDims; i++) {
        tensormu->dims[i + 1] = dims[i];
    }
    // set the data type
    tensormu->dataType = &dataType;
    // here we make a distinction between each of the 6 potential data types
    // and create the appropriate tensor for each one
    if (numDims == 1) {
        if (dataType == 0) {
            tensormu->data = create_device_tensor_short(dims[0]);
        }
        if (dataType == 1) {
            tensormu->data = create_device_tensor_int(dims[0]);
        }
        if (dataType == 2) {
            tensormu->data = create_device_tensor_float(dims[0]);
        }
        if (dataType == 3) {
            tensormu->data = create_device_tensor_double(dims[0]);
        }
        if (dataType == 4) {
            tensormu->data = create_device_tensor_long(dims[0]);
        }
    if (numDims > 1) {
        tensormu->data = create_tensor_short_multi_device(numDims - 1, dims[1], dataType);
    }
    return tensormu;
}

// function to destroy a TensorMulti which puts its data on the device
void destroy_tensor_short_multi_device(TensorMulti* tensormu) {
    // if the tensormu has dimensions greater than 1, we need to
    // iterate through the data and destroy each tensor in it
    if (*(tensormu->dims[0]) > 1) {
        for (int i = 0; i < tensormu->dims[1]; i++) {
            destroy_tensor_short_multi_device((TensorMulti*)tensormu->data[i]);
        }
    }
    // otherwise if the remaining dims is just 1, we destroy the data tensor
    else {
        if (*(tensormu->dataType) == 0) {
            destroy_device_tensor_short((Tensor*)tensormu->data);
        }
        if (*(tensormu->dataType) == 1) {
            destroy_device_tensor_int((TensorInt*)tensormu->data);
        }
        if (*(tensormu->dataType) == 2) {
            destroy_device_tensor_float((TensorFloat*)tensormu->data);
        }
        if (*(tensormu->dataType) == 3) {
            destroy_device_tensor_double((TensorDouble*)tensormu->data);
        }
        if (*(tensormu->dataType) == 4) {
            destroy_device_tensor_long((TensorLong*)tensormu->data);
        }
    }
    // free the dims array
    free(tensormu->dims);
    // free the tensor_short_multi
    free(tensormu);
}


// function to cast the data of a TensorMulti to the appropriate struct type
void* cast_data(TensorMulti* tensormu) {
    // if the tensor_short_multi has multiple dimensions, its holding tensor_multi's, 
    // and we need to cast the data tensor as such
    if (*(tensormu->dims[0]) > 1) {
        return cast_data((TensorMulti*)tensormu->data);
    }
    // otherwise, we can cast the data tensor directly
    else {
        if (*(tensormu->dataType) == 0) {
            return (TensorShort*)tensormu->data;
        }
        if (*(tensormu->dataType) == 1) {
            return (TensorInt*)tensormu->data;
        }
        if (*(tensormu->dataType) == 2) {
            return (TensorFloat*)tensormu->data;
        }
        if (*(tensormu->dataType) == 3) {
            return (TensorDouble*)tensormu->data;
        }
        if (*(tensormu->dataType) == 4) {
            return (TensorLong*)tensormu->data;
        }
    }
}

// function to set a tensormulti's values to 0
void set_tensor_short_multi_zero(TensorMulti* tensormu) {
    // if the tensor_short_multi has multiple dimensions, its holding tensor_multi's, 
    // and we need to set the data tensor as such
    if (*(tensormu->dims[0]) > 1) {
        set_tensor_short_multi_zero((TensorMulti*)tensormu->data);
    }
    // otherwise, we can set the data tensor directly
    else {
        if (*(tensormu->dataType) == 0) {
            set_tensor_zero_short((Tensor*)tensormu->data);
        }
        if (*(tensormu->dataType) == 1) {
            set_tensor_int_zero((TensorInt*)tensormu->data);
        }
        if (*(tensormu->dataType) == 2) {
            set_tensor_float_zero((TensorFloat*)tensormu->data);
        }
        if (*(tensormu->dataType) == 3) {
            set_tensor_double_zero((TensorDouble*)tensormu->data);
        }
        if (*(tensormu->dataType) == 4) {
            set_tensor_long_zero((TensorLong*)tensormu->data);
        }
    }
}

// function to set a tensormulti's values to 1
void set_tensor_short_multi_one(TensorMulti* tensormu) {
    // if the tensor_short_multi has multiple dimensions, its holding tensor_multi's, 
    // and we need to set the data tensor as such
    if (*(tensormu->dims[0]) > 1) {
        set_tensor_short_multi_one((TensorMulti*)tensormu->data);
    }
    // otherwise, we can set the data tensor directly
    else {
        if (*(tensormu->dataType) == 0) {
            set_tensor_short_one((Tensor*)tensormu->data);
        }
        if (*(tensormu->dataType) == 1) {
            set_tensor_int_one((TensorInt*)tensormu->data);
        }
        if (*(tensormu->dataType) == 2) {
            set_tensor_float_one((TensorFloat*)tensormu->data);
        }
        if (*(tensormu->dataType) == 3) {
            set_tensor_double_one((TensorDouble*)tensormu->data);
        }
        if (*(tensormu->dataType) == 4) {
            set_tensor_long_one((TensorLong*)tensormu->data);
        }
    }
}

// function to set a tensormulti's values to a constant
void fill_tensor_short_multi(TensorMulti* tensormu, void* constant) {
    // if the tensor_short_multi has multiple dimensions, its holding tensor_multi's, 
    // and we need to set the data tensor as such
    if (*(tensormu->dims[0]) > 1) {
        set_tensor_short_multi_constant((TensorMulti*)tensormu->data, constant);
    }
    // otherwise, we can set the data tensor directly
    else {
        if (*(tensormu->dataType) == 0) {
            fill_tensor_short((Tensor*)tensormu->data, constant);
        }
        if (*(tensormu->dataType) == 1) {
            fill_tensor_int((TensorInt*)tensormu->data, constant);
        }
        if (*(tensormu->dataType) == 2) {
            fill_tensor_float((TensorFloat*)tensormu->data, constant);
        }
        if (*(tensormu->dataType) == 3) {
            fill_tensor_double((TensorDouble*)tensormu->data, constant);
        }
        if (*(tensormu->dataType) == 4) {
            fill_tensor_long((TensorLong*)tensormu->data, constant);
        }
    }
}

// function to set a particular element in a tensor multi to a value
void set_tensor_short_multi_element(TensorMulti* tensormu, int* indices, void* value) {
    // if the tensor_short_multi has multiple dimensions, its holding tensor_multi's, 
    // and we need to set the data tensor as such
    if (*(tensormu->dims[0]) > 1) {
        set_tensor_short_multi_element((TensorMulti*)tensormu->data, indices + 1, value);
    }
    // otherwise, we can set the data tensor directly
    else {
        if (*(tensormu->dataType) == 0) {
            set_tensor_short_element((Tensor*)tensormu->data, indices[0], value);
        }
        if (*(tensormu->dataType) == 1) {
            set_tensor_int_element((TensorInt*)tensormu->data, indices[0], value);
        }
        if (*(tensormu->dataType) == 2) {
            set_tensor_float_element((TensorFloat*)tensormu->data, indices[0], value);
        }
        if (*(tensormu->dataType) == 3) {
            set_tensor_double_element((TensorDouble*)tensormu->data, indices[0], value);
        }
        if (*(tensormu->dataType) == 4) {
            set_tensor_long_element((TensorLong*)tensormu->data, indices[0], value);
        }
    }
}

// function to get a particular element in a tensor multi
void* get_tensor_short_multi_element(TensorMulti* tensormu, int* indices) {
    // if the tensor_short_multi has multiple dimensions, its holding tensor_multi's, 
    // and we need to set the data tensor as such
    if (*(tensormu->dims[0]) > 1) {
        return get_tensor_short_multi_element((TensorMulti*)tensormu->data, indices + 1);
    }
    // otherwise, we can set the data tensor directly
    else {
        if (*(tensormu->dataType) == 0) {
            return get_tensor_short_element((Tensor*)tensormu->data, indices[0]);
        }
        if (*(tensormu->dataType) == 1) {
            return get_tensor_int_element((TensorInt*)tensormu->data, indices[0]);
        }
        if (*(tensormu->dataType) == 2) {
            return get_tensor_float_element((TensorFloat*)tensormu->data, indices[0]);
        }
        if (*(tensormu->dataType) == 3) {
            return get_tensor_double_element((TensorDouble*)tensormu->data, indices[0]);
        }
        if (*(tensormu->dataType) == 4) {
            return get_tensor_long_element((TensorLong*)tensormu->data, indices[0]);
        }
    }
}

// function to add two tensor multi's together
void add_tensor_short_multi(TensorMulti* tensormu1, TensorMulti* tensormu2, TensorMulti* result) {
    // if the tensor_short_multi has multiple dimensions, its holding tensor_multi's, 
    // and we need to set the data tensor as such
    if (*(tensormu1->dims[0]) > 1) {
        add_tensor_short_multi((TensorMulti*)tensormu1->data, (TensorMulti*)tensormu2->data, (TensorMulti*)result->data);
    }
    // otherwise, we can set the data tensor directly
    else {
        if (*(tensormu1->dataType) == 0) {
            add_tensor_short((Tensor*)tensormu1->data, (Tensor*)tensormu2->data, (Tensor*)result->data);
        }
        if (*(tensormu1->dataType) == 1) {
            add_tensor_int((TensorInt*)tensormu1->data, (TensorInt*)tensormu2->data, (TensorInt*)result->data);
        }
        if (*(tensormu1->dataType) == 2) {
            add_tensor_float((TensorFloat*)tensormu1->data, (TensorFloat*)tensormu2->data, (TensorFloat*)result->data);
        }
        if (*(tensormu1->dataType) == 3) {
            add_tensor_double((TensorDouble*)tensormu1->data, (TensorDouble*)tensormu2->data, (TensorDouble*)result->data);
        }
        if (*(tensormu1->dataType) == 4) {
            add_tensor_long((TensorLong*)tensormu1->data, (TensorLong*)tensormu2->data, (TensorLong*)result->data);
        }
    }
}

// function to subtract two tensor multi's
void subtract_tensor_short_multi(TensorMulti* tensormu1, TensorMulti* tensormu2, TensorMulti* result) {
    // if the tensor_short_multi has multiple dimensions, its holding tensor_multi's, 
    // and we need to set the data tensor as such
    if (*(tensormu1->dims[0]) > 1) {
        subtract_tensor_short_multi((TensorMulti*)tensormu1->data, (TensorMulti*)tensormu2->data, (TensorMulti*)result->data);
    }
    // otherwise, we can set the data tensor directly
    else {
        if (*(tensormu1->dataType) == 0) {
            subtract_tensor_short((Tensor*)tensormu1->data, (Tensor*)tensormu2->data, (Tensor*)result->data);
        }
        if (*(tensormu1->dataType) == 1) {
            subtract_tensor_int((TensorInt*)tensormu1->data, (TensorInt*)tensormu2->data, (TensorInt*)result->data);
        }
        if (*(tensormu1->dataType) == 2) {
            subtract_tensor_float((TensorFloat*)tensormu1->data, (TensorFloat*)tensormu2->data, (TensorFloat*)result->data);
        }
        if (*(tensormu1->dataType) == 3) {
            subtract_tensor_double((TensorDouble*)tensormu1->data, (TensorDouble*)tensormu2->data, (TensorDouble*)result->data);
        }
        if (*(tensormu1->dataType) == 4) {
            subtract_tensor_long((TensorLong*)tensormu1->data, (TensorLong*)tensormu2->data, (TensorLong*)result->data);
        }
    }
}

// function to multiply two tensor multi's
void multiply_tensor_short_multi(TensorMulti* tensormu1, TensorMulti* tensormu2, TensorMulti* result) {
    // if the tensor_short_multi has multiple dimensions, its holding tensor_multi's, 
    // and we need to set the data tensor as such
    if (*(tensormu1->dims[0]) > 1) {
        multiply_tensor_short_multi((TensorMulti*)tensormu1->data, (TensorMulti*)tensormu2->data, (TensorMulti*)result->data);
    }
    // otherwise, we can set the data tensor directly
    else {
        if (*(tensormu1->dataType) == 0) {
            multiply_tensor_short((Tensor*)tensormu1->data, (Tensor*)tensormu2->data, (Tensor*)result->data);
        }
        if (*(tensormu1->dataType) == 1) {
            multiply_tensor_int((TensorInt*)tensormu1->data, (TensorInt*)tensormu2->data, (TensorInt*)result->data);
        }
        if (*(tensormu1->dataType) == 2) {
            multiply_tensor_float((TensorFloat*)tensormu1->data, (TensorFloat*)tensormu2->data, (TensorFloat*)result->data);
        }
        if (*(tensormu1->dataType) == 3) {
            multiply_tensor_double((TensorDouble*)tensormu1->data, (TensorDouble*)tensormu2->data, (TensorDouble*)result->data);
        }
        if (*(tensormu1->dataType) == 4) {
            multiply_tensor_long((TensorLong*)tensormu1->data, (TensorLong*)tensormu2->data, (TensorLong*)result->data);
        }
    }
}

// function to divide two tensor multi's
void divide_tensor_short_multi(TensorMulti* tensormu1, TensorMulti* tensormu2, TensorMulti* result) {
    // if the tensor_short_multi has multiple dimensions, its holding tensor_multi's, 
    // and we need to set the data tensor as such
    if (*(tensormu1->dims[0]) > 1) {
        divide_tensor_short_multi((TensorMulti*)tensormu1->data, (TensorMulti*)tensormu2->data, (TensorMulti*)result->data);
    }
    // otherwise, we can set the data tensor directly
    else {
        if (*(tensormu1->dataType) == 0) {
            divide_tensor_short((Tensor*)tensormu1->data, (Tensor*)tensormu2->data, (Tensor*)result->data);
        }
        if (*(tensormu1->dataType) == 1) {
            divide_tensor_int((TensorInt*)tensormu1->data, (TensorInt*)tensormu2->data, (TensorInt*)result->data);
        }
        if (*(tensormu1->dataType) == 2) {
            divide_tensor_float((TensorFloat*)tensormu1->data, (TensorFloat*)tensormu2->data, (TensorFloat*)result->data);
        }
        if (*(tensormu1->dataType) == 3) {
            divide_tensor_double((TensorDouble*)tensormu1->data, (TensorDouble*)tensormu2->data, (TensorDouble*)result->data);
        }
        if (*(tensormu1->dataType) == 4) {
            divide_tensor_long((TensorLong*)tensormu1->data, (TensorLong*)tensormu2->data, (TensorLong*)result->data);
        }
    }
}

// function to add a scalar to a tensor multi
void add_scalar_tensor_short_multi(TensorMulti* tensormu, void* scalar, TensorMulti* result) {
    // if the tensor_short_multi has multiple dimensions, its holding tensor_multi's, 
    // and we need to set the data tensor as such
    if (*(tensormu->dims[0]) > 1) {
        add_scalar_tensor_short_multi((TensorMulti*)tensormu->data, scalar, (TensorMulti*)result->data);
    }
    // otherwise, we can set the data tensor directly
    else {
        if (*(tensormu->dataType) == 0) {
            add_scalar_tensor_short((Tensor*)tensormu->data, scalar, (Tensor*)result->data);
        }
        if (*(tensormu->dataType) == 1) {
            add_scalar_tensor_int((TensorInt*)tensormu->data, scalar, (TensorInt*)result->data);
        }
        if (*(tensormu->dataType) == 2) {
            add_scalar_tensor_float((TensorFloat*)tensormu->data, scalar, (TensorFloat*)result->data);
        }
        if (*(tensormu->dataType) == 3) {
            add_scalar_tensor_double((TensorDouble*)tensormu->data, scalar, (TensorDouble*)result->data);
        }
        if (*(tensormu->dataType) == 4) {
            add_scalar_tensor_long((TensorLong*)tensormu->data, scalar, (TensorLong*)result->data);
        }
    }
}

// function to subtract a scalar from a tensor multi
void subtract_scalar_tensor_short_multi(TensorMulti* tensormu, void* scalar, TensorMulti* result) {
    // if the tensor_short_multi has multiple dimensions, its holding tensor_multi's, 
    // and we need to set the data tensor as such
    if (*(tensormu->dims[0]) > 1) {
        subtract_scalar_tensor_short_multi((TensorMulti*)tensormu->data, scalar, (TensorMulti*)result->data);
    }
    // otherwise, we can set the data tensor directly
    else {
        if (*(tensormu->dataType) == 0) {
            subtract_scalar_tensor_short((Tensor*)tensormu->data, scalar, (Tensor*)result->data);
        }
        if (*(tensormu->dataType) == 1) {
            subtract_scalar_tensor_int((TensorInt*)tensormu->data, scalar, (TensorInt*)result->data);
        }
        if (*(tensormu->dataType) == 2) {
            subtract_scalar_tensor_float((TensorFloat*)tensormu->data, scalar, (TensorFloat*)result->data);
        }
        if (*(tensormu->dataType) == 3) {
            subtract_scalar_tensor_double((TensorDouble*)tensormu->data, scalar, (TensorDouble*)result->data);
        }
        if (*(tensormu->dataType) == 4) {
            subtract_scalar_tensor_long((TensorLong*)tensormu->data, scalar, (TensorLong*)result->data);
        }
    }
}

// function to multiply a tensor multi by a scalar
void multiply_scalar_tensor_short_multi(TensorMulti* tensormu, void* scalar, TensorMulti* result) {
    // if the tensor_short_multi has multiple dimensions, its holding tensor_multi's, 
    // and we need to set the data tensor as such
    if (*(tensormu->dims[0]) > 1) {
        multiply_scalar_tensor_short_multi((TensorMulti*)tensormu->data, scalar, (TensorMulti*)result->data);
    }
    // otherwise, we can set the data tensor directly
    else {
        if (*(tensormu->dataType) == 0) {
            multiply_scalar_tensor_short((Tensor*)tensormu->data, scalar, (Tensor*)result->data);
        }
        if (*(tensormu->dataType) == 1) {
            multiply_scalar_tensor_int((TensorInt*)tensormu->data, scalar, (TensorInt*)result->data);
        }
        if (*(tensormu->dataType) == 2) {
            multiply_scalar_tensor_float((TensorFloat*)tensormu->data, scalar, (TensorFloat*)result->data);
        }
        if (*(tensormu->dataType) == 3) {
            multiply_scalar_tensor_double((TensorDouble*)tensormu->data, scalar, (TensorDouble*)result->data);
        }
        if (*(tensormu->dataType) == 4) {
            multiply_scalar_tensor_long((TensorLong*)tensormu->data, scalar, (TensorLong*)result->data);
        }
    }
}

// function to divide a tensor multi by a scalar
void divide_scalar_tensor_short_multi(TensorMulti* tensormu, void* scalar, TensorMulti* result) {
    // if the tensor_short_multi has multiple dimensions, its holding tensor_multi's, 
    // and we need to set the data tensor as such
    if (*(tensormu->dims[0]) > 1) {
        divide_scalar_tensor_short_multi((TensorMulti*)tensormu->data, scalar, (TensorMulti*)result->data);
    }
    // otherwise, we can set the data tensor directly
    else {
        if (*(tensormu->dataType) == 0) {
            divide_scalar_tensor_short((Tensor*)tensormu->data, scalar, (Tensor*)result->data);
        }
        if (*(tensormu->dataType) == 1) {
            divide_scalar_tensor_int((TensorInt*)tensormu->data, scalar, (TensorInt*)result->data);
        }
        if (*(tensormu->dataType) == 2) {
            divide_scalar_tensor_float((TensorFloat*)tensormu->data, scalar, (TensorFloat*)result->data);
        }
        if (*(tensormu->dataType) == 3) {
            divide_scalar_tensor_double((TensorDouble*)tensormu->data, scalar, (TensorDouble*)result->data);
        }
        if (*(tensormu->dataType) == 4) {
            divide_scalar_tensor_long((TensorLong*)tensormu->data, scalar, (TensorLong*)result->data);
        }
    }
}

// function that takes a tensor multi and a set of booleans in a tensor multi and returns the logical AND of both tensors element-wise
void logical_and_tensor_short_multi(TensorMulti* tensormu1, TensorMulti* tensormu2, TensorMulti* result) {
    // if the tensor_short_multi has multiple dimensions, its holding tensor_multi's, 
    // and we need to set the data tensor as such
    if (*(tensormu1->dims[0]) > 1) {
        logical_and_tensor_short_multi((TensorMulti*)tensormu1->data, (TensorMulti*)tensormu2->data, (TensorMulti*)result->data);
    }
    // otherwise, we can set the data tensor directly
    else {
        if (*(tensormu1->dataType) == 0) {
            logical_and_tensor_short((Tensor*)tensormu1->data, (Tensor*)tensormu2->data, (Tensor*)result->data);
        }
        if (*(tensormu1->dataType) == 1) {
            logical_and_tensor_int((TensorInt*)tensormu1->data, (TensorInt*)tensormu2->data, (TensorInt*)result->data);
        }
        if (*(tensormu1->dataType) == 2) {
            logical_and_tensor_float((TensorFloat*)tensormu1->data, (TensorFloat*)tensormu2->data, (TensorFloat*)result->data);
        }
        if (*(tensormu1->dataType) == 3) {
            logical_and_tensor_double((TensorDouble*)tensormu1->data, (TensorDouble*)tensormu2->data, (TensorDouble*)result->data);
        }
        if (*(tensormu1->dataType) == 4) {
            logical_and_tensor_long((TensorLong*)tensormu1->data, (TensorLong*)tensormu2->data, (TensorLong*)result->data);
        }
    }
}

// function that takes a tensor multi and a set of booleans in a tensor multi and returns the logical OR of both tensors element-wise
void logical_or_tensor_short_multi(TensorMulti* tensormu1, TensorMulti* tensormu2, TensorMulti* result) {
    // if the tensor_short_multi has multiple dimensions, its holding tensor_multi's, 
    // and we need to set the data tensor as such
    if (*(tensormu1->dims[0]) > 1) {
        logical_or_tensor_short_multi((TensorMulti*)tensormu1->data, (TensorMulti*)tensormu2->data, (TensorMulti*)result->data);
    }
    // otherwise, we can set the data tensor directly
    else {
        if (*(tensormu1->dataType) == 0) {
            logical_or_tensor_short((Tensor*)tensormu1->data, (Tensor*)tensormu2->data, (Tensor*)result->data);
        }
        if (*(tensormu1->dataType) == 1) {
            logical_or_tensor_int((TensorInt*)tensormu1->data, (TensorInt*)tensormu2->data, (TensorInt*)result->data);
        }
        if (*(tensormu1->dataType) == 2) {
            logical_or_tensor_float((TensorFloat*)tensormu1->data, (TensorFloat*)tensormu2->data, (TensorFloat*)result->data);
        }
        if (*(tensormu1->dataType) == 3) {
            logical_or_tensor_double((TensorDouble*)tensormu1->data, (TensorDouble*)tensormu2->data, (TensorDouble*)result->data);
        }
        if (*(tensormu1->dataType) == 4) {
            logical_or_tensor_long((TensorLong*)tensormu1->data, (TensorLong*)tensormu2->data, (TensorLong*)result->data);
        }
    }
}

// function that takes a tensor multi and a set of booleans in a tensor multi and returns the logical NOT of the tensor
void logical_not_tensor_short_multi(TensorMulti* tensormu, TensorMulti* result) {
    // if the tensor_short_multi has multiple dimensions, its holding tensor_multi's, 
    // and we need to set the data tensor as such
    if (*(tensormu->dims[0]) > 1) {
        logical_not_tensor_short_multi((TensorMulti*)tensormu->data, (TensorMulti*)result->data);
    }
    // otherwise, we can set the data tensor directly
    else {
        if (*(tensormu->dataType) == 0) {
            logical_not_tensor_short((Tensor*)tensormu->data, (Tensor*)result->data);
        }
        if (*(tensormu->dataType) == 1) {
            logical_not_tensor_int((TensorInt*)tensormu->data, (TensorInt*)result->data);
        }
        if (*(tensormu->dataType) == 2) {
            logical_not_tensor_float((TensorFloat*)tensormu->data, (TensorFloat*)result->data);
        }
        if (*(tensormu->dataType) == 3) {
            logical_not_tensor_double((TensorDouble*)tensormu->data, (TensorDouble*)result->data);
        }
        if (*(tensormu->dataType) == 4) {
            logical_not_tensor_long((TensorLong*)tensormu->data, (TensorLong*)result->data);
        }
    }
}

// function that takes a tensor multi and a set of booleans in a tensor multi and returns the logical XOR of both tensors element-wise
void logical_xor_tensor_short_multi(TensorMulti* tensormu1, TensorMulti* tensormu2, TensorMulti* result) {
    // if the tensor_short_multi has multiple dimensions, its holding tensor_multi's, 
    // and we need to set the data tensor as such
    if (*(tensormu1->dims[0]) > 1) {
        logical_xor_tensor_short_multi((TensorMulti*)tensormu1->data, (TensorMulti*)tensormu2->data, (TensorMulti*)result->data);
    }
    // otherwise, we can set the data tensor directly
    else {
        if (*(tensormu1->dataType) == 0) {
            logical_xor_tensor_short((Tensor*)tensormu1->data, (Tensor*)tensormu2->data, (Tensor*)result->data);
        }
        if (*(tensormu1->dataType) == 1) {
            logical_xor_tensor_int((TensorInt*)tensormu1->data, (TensorInt*)tensormu2->data, (TensorInt*)result->data);
        }
        if (*(tensormu1->dataType) == 2) {
            logical_xor_tensor_float((TensorFloat*)tensormu1->data, (TensorFloat*)tensormu2->data, (TensorFloat*)result->data);
        }
        if (*(tensormu1->dataType) == 3) {
            logical_xor_tensor_double((TensorDouble*)tensormu1->data, (TensorDouble*)tensormu2->data, (TensorDouble*)result->data);
        }
        if (*(tensormu1->dataType) == 4) {
            logical_xor_tensor_long((TensorLong*)tensormu1->data, (TensorLong*)tensormu2->data, (TensorLong*)result->data);
        }
    }
}

// transposes a tensor multi
void transpose_tensor_short_multi(TensorMulti* tensormu, TensorMulti* result) {
    // we need to get elements from the actual data and swap them around
    // so we'll need to first reverse the order of the data in the data tensors
    // and then reverse the order of the data tensors in the tensor multi overall
    tmp = (TensorMulti*)malloc(sizeof(TensorMulti));
    tmp->dims = tensormu->dims;
    tmp->dataType = tensormu->dataType;

    if (*(tensormu->dims[0]) > 1) {
        tmp->data = (TensorMulti*)malloc(tensormu->dims[1] * sizeof(TensorMulti));
        // reverse the order of the multi tensors in the tensor multi
        for (int i = 0, i < tensormu->dims[1], i++) {
            tmp->data[i] = tensormu->data[tensormu->dims[1] - i - 1];
            // now iterate across the data tensor and transpose each one
            transpose_tensor_short_multi(tmp->data[i], tmp->data[i]);
        }
    }
    // now, if we get a data tensor, we just reverse the order of it and return it
    if (*(tensormu->dims[0]) == 1) {
        if (*(tensormu->dataType) == 0) {
            transpose_tensor_short((TensorShort*)tensormu->data, (TensorShort*)tmp->data);
        }
        if (*(tensormu->dataType) == 1) {
            transpose_tensor_int((TensorInt*)tensormu->data, (TensorInt*)tmp->data);
        }
        if (*(tensormu->dataType) == 2) {
            transpose_tensor_float((TensorFloat*)tensormu->data, (TensorFloat*)tmp->data);
        }
        if (*(tensormu->dataType) == 3) {
            transpose_tensor_double((TensorDouble*)tensormu->data, (TensorDouble*)tmp->data);
        }
        if (*(tensormu->dataType) == 4) {
            transpose_tensor_long((TensorLong*)tensormu->data, (TensorLong*)tmp->data);
        }
    }

    result = tmp;
}

// function to get the shape of a tensor multi
int* get_shape_tensor_short_multi(TensorMulti* tensormu) {
    return tensormu->dims;
}

// function to get the data type of a tensor multi
int get_data_type_tensor_short_multi(TensorMulti* tensormu) {
    return *(tensormu->dataType);
}

// function to get the number of dimensions of a tensor multi
int get_num_dims_tensor_short_multi(TensorMulti* tensormu) {
    return tensormu->dims[0];
}

// function to get the number of elements in a tensor multi
int get_num_elements_tensor_short_multi(TensorMulti* tensormu) {
    int numElements = 1;
    for (int i = 1; i < tensormu->dims[0] + 1; i++) {
        numElements *= tensormu->dims[i];
    }
    return numElements;
}

// function that adds one tensor multi with another by following the vectors of a third tensor multi
extern "C" void vector_add_tensor_short_multi(TensorMulti* tensormu, TensorMulti* other, TensorMulti* vectors, TensorMulti* result) {
    




