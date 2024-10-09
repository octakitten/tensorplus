#define PY_SSIZE_T_CLEAN
#include "python3.12/Python.h"
#include "includes.h"

static PyObject * create(PyObject * self, PyObject * args) {
    unsigned int size;
    if (!PyArg_ParseTuple(args, "I", &size)) {
        return NULL;
    }
    Tensor* tensor = create_device_tensor(size);
    return PyCapsule_New(tensor, "Tensor", NULL);
}

static PyObject * destroy(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    destroy_device_tensor(tensor);
    return Py_BuildValue("");
}

static PyObject * copy(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    PyObject * result_obj;
    if (!PyArg_ParseTuple(args, "OO", &tensor_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    Tensor* result = (Tensor*) PyCapsule_GetPointer(result_obj, "Tensor");
    copy_tensor(tensor, result);
    return Py_BuildValue("");
}

static PyObject * clone_t(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    PyObject * result_obj;
    if (!PyArg_ParseTuple(args, "OO", &tensor_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    Tensor* result = (Tensor*) PyCapsule_GetPointer(result_obj, "Tensor");
    clone_tensor(tensor, result);
    return Py_BuildValue("");
}

static PyObject * create_ct(PyObject * self, PyObject * args) {
    unsigned int size;
    if (!PyArg_ParseTuple(args, "I", &size)) {
        return NULL;
    }
    Tensor* tensor = create_tensor(size);
    return PyCapsule_New(tensor, "Tensor", NULL);
}

static PyObject * destroy_ct(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    destroy_tensor(tensor);
    return Py_BuildValue("");
}

static PyObject * zeros(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    zeros_tensor_wrapper(tensor);
    return Py_BuildValue("");
}

static PyObject * ones(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    ones_tensor_wrapper(tensor);
    return Py_BuildValue("");
}

static PyObject * printtn(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    print_tensor_wrapper(tensor);
    return Py_BuildValue("");
}

static PyObject * fill(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    short value;
    if (!PyArg_ParseTuple(args, "Oh", &tensor_obj, &value)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    fill_tensor_wrapper(tensor, value);
    return Py_BuildValue("");
}

static PyObject * set(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    unsigned int index;
    short value;
    if (!PyArg_ParseTuple(args, "OIh", &tensor_obj, &value)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    set_tensor_wrapper(tensor, index, value);
    return Py_BuildValue("");
}

static PyObject * get(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    unsigned int index;
    short value;
    if (!PyArg_ParseTuple(args, "OIh", &tensor_obj, &value)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    value = get_tensor_value_wrapper(tensor, index);
    return Py_BuildValue("");
}

static PyObject * add(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    PyObject * other_obj;
    PyObject * result_obj;
    if (!PyArg_ParseTuple(args, "OOO", &tensor_obj, &other_obj, &result_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    Tensor* other = (Tensor*) PyCapsule_GetPointer(other_obj, "Tensor");
    Tensor* result = (Tensor*) PyCapsule_GetPointer(result_obj , "Tensor");
    add_tensor_wrapper(tensor, other, result);
    return Py_BuildValue("");
}

static PyObject * sub(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    PyObject * other_obj;
    PyObject * result_obj;
    if (!PyArg_ParseTuple(args, "OOO", &tensor_obj, &other_obj, &result_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    Tensor* other = (Tensor*) PyCapsule_GetPointer(other_obj, "Tensor");
    Tensor* result = (Tensor*) PyCapsule_GetPointer(result_obj, "Tensor");
    sub_tensor_wrapper(tensor, other, result);
    return Py_BuildValue("");
}

static PyObject * mul(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    PyObject * other_obj;
    PyObject * result_obj;
    if (!PyArg_ParseTuple(args, "OOO", &tensor_obj, &other_obj, &result_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    Tensor* other = (Tensor*) PyCapsule_GetPointer(other_obj, "Tensor");
    Tensor* result = (Tensor*) PyCapsule_GetPointer(result_obj, "Tensor");
    mul_tensor_wrapper(tensor, other, result);
    return Py_BuildValue("");
}

static PyObject * div_tn(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    PyObject * other_obj;
    PyObject * result_obj;
    if (!PyArg_ParseTuple(args, "OOO", &tensor_obj, &other_obj, &result_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    Tensor* other = (Tensor*) PyCapsule_GetPointer(other_obj, "Tensor");
    Tensor* result = (Tensor*) PyCapsule_GetPointer(result_obj, "Tensor");
    div_tensor_wrapper(tensor, other, result);
    return Py_BuildValue("");
}

static PyObject * add_scalar(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    short scalar;
    PyObject * result_obj;
    if (!PyArg_ParseTuple(args, "OhO", &tensor_obj, &scalar, &result_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    Tensor* result = (Tensor*) malloc(sizeof(Tensor*));
    add_scalar_tensor_wrapper(tensor, scalar, result);
    return Py_BuildValue("");
}

static PyObject * sub_scalar(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    short scalar;
    PyObject * result_obj;
    if (!PyArg_ParseTuple(args, "OhO", &tensor_obj, &scalar, &result_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    Tensor* result = (Tensor*) PyCapsule_GetPointer(result_obj, "Tensor");
    sub_scalar_tensor_wrapper(tensor, scalar, result);
    return Py_BuildValue("");
}

static PyObject * mul_scalar(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    short scalar;
    PyObject * result_obj;
    if (!PyArg_ParseTuple(args, "OhO", &tensor_obj, &scalar, &result_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    Tensor* result = (Tensor*) PyCapsule_GetPointer(result_obj, "Tensor");
    mul_scalar_tensor_wrapper(tensor, scalar, result);
    return Py_BuildValue("");
}

static PyObject * div_scalar(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    short scalar;
    PyObject * result_obj;
    if (!PyArg_ParseTuple(args, "OhO", &tensor_obj, &scalar, &result_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    Tensor* result = (Tensor*) PyCapsule_GetPointer(result_obj, "Tensor");
    div_scalar_tensor_wrapper(tensor, scalar, result);
    return Py_BuildValue("");
}

static PyObject * transpose(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    PyObject * result_obj;
    if (!PyArg_ParseTuple(args, "OO", &tensor_obj, &result_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    Tensor* result = (Tensor*) malloc(sizeof(Tensor*));
    transpose_tensor_wrapper(tensor, result);
    return Py_BuildValue("");
}

static PyObject * sum(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    PyObject * result_obj;
    if (!PyArg_ParseTuple(args, "OO", &tensor_obj, &result_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    Tensor* result = (Tensor*) PyCapsule_GetPointer(result_obj, "Tensor");
    sum_tensor_wrapper(tensor, result);
    return Py_BuildValue("");
}

static PyObject * max(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    PyObject * result_obj;
    if (!PyArg_ParseTuple(args, "OO", &tensor_obj, &result_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    Tensor* result = (Tensor*) PyCapsule_GetPointer(result_obj, "Tensor");
    max_tensor_wrapper(tensor, result);
    return Py_BuildValue("");
}

static PyObject * min(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    PyObject * result_obj;
    if (!PyArg_ParseTuple(args, "OO", &tensor_obj, &result_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    Tensor* result = (Tensor*) PyCapsule_GetPointer(result_obj, "Tensor");
    min_tensor_wrapper(tensor, result);
    return Py_BuildValue("");
}

static PyObject * gradient(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    PyObject * result_obj;
    if (!PyArg_ParseTuple(args, "OO", &tensor_obj, &result_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    Tensor* result = (Tensor*) PyCapsule_GetPointer(result_obj, "Tensor");
    gradient_tensor_wrapper(tensor, result);
    return Py_BuildValue("");
}

static PyObject * gate(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    PyObject * booleans_obj;
    PyObject * result_obj;
    if (!PyArg_ParseTuple(args, "OOO", &tensor_obj, &booleans_obj, &result_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    Tensor* booleans = (Tensor*) PyCapsule_GetPointer(booleans_obj, "Tensor");
    Tensor* result = (Tensor*) PyCapsule_GetPointer(result_obj, "Tensor");
    gate_tensor_wrapper(tensor, booleans, result);
    return Py_BuildValue("");
}

static PyObject * vector_sort(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    PyObject * vectors_obj;
    PyObject * result_obj;
    if (!PyArg_ParseTuple(args, "OOO", &tensor_obj, &vectors_obj, &result_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    Tensor* vectors = (Tensor*) PyCapsule_GetPointer(vectors_obj, "Tensor");
    Tensor* result = (Tensor*) PyCapsule_GetPointer(result_obj, "Tensor");
    vector_sort_tensor_wrapper(tensor, vectors, result);
    return Py_BuildValue("");
}

static PyObject * vector_add(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    PyObject * other_obj;
    PyObject * vectors_obj;
    PyObject * result_obj;
    if (!PyArg_ParseTuple(args, "OOOO", &tensor_obj, &other_obj, &vectors_obj, &result_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    Tensor* other = (Tensor*) PyCapsule_GetPointer(other_obj, "Tensor");
    Tensor* vectors = (Tensor*) PyCapsule_GetPointer(vectors_obj, "Tensor");
    Tensor* result = (Tensor*) PyCapsule_GetPointer(result_obj, "Tensor");
    vector_add_wrapper(tensor, other, vectors, result);
    return Py_BuildValue("");
}

static PyObject * vector_sub(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    PyObject * other_obj;
    PyObject * vectors_obj;
    PyObject * result_obj;
    if (!PyArg_ParseTuple(args, "OOOO", &tensor_obj, &other_obj, &vectors_obj, &result_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    Tensor* other = (Tensor*) PyCapsule_GetPointer(other_obj, "Tensor");
    Tensor* vectors = (Tensor*) PyCapsule_GetPointer(vectors_obj, "Tensor");
    Tensor* result = (Tensor*) PyCapsule_GetPointer(result_obj, "Tensor");
    vector_sub_wrapper(tensor, other, vectors, result);
    return Py_BuildValue("");
}

static PyObject * vector_mul(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    PyObject * other_obj;
    PyObject * vectors_obj;
    PyObject * result_obj;
    if (!PyArg_ParseTuple(args, "OOOO", &tensor_obj, &other_obj, &vectors_obj, &result_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    Tensor* other = (Tensor*) PyCapsule_GetPointer(other_obj, "Tensor");
    Tensor* vectors = (Tensor*) PyCapsule_GetPointer(vectors_obj, "Tensor");
    Tensor* result = (Tensor*) PyCapsule_GetPointer(result_obj, "Tensor");
    vector_mul_wrapper(tensor, other, vectors, result);
    return Py_BuildValue("");
}

static PyObject * vector_div(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    PyObject * other_obj;
    PyObject * vectors_obj;
    PyObject * result_obj;
    if (!PyArg_ParseTuple(args, "OOOO", &tensor_obj, &other_obj, &vectors_obj, &result_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    Tensor* other = (Tensor*) PyCapsule_GetPointer(other_obj, "Tensor");
    Tensor* vectors = (Tensor*) PyCapsule_GetPointer(vectors_obj, "Tensor");
    Tensor* result = (Tensor*) PyCapsule_GetPointer(result_obj, "Tensor");
    vector_div_wrapper(tensor, other, vectors, result);
    return Py_BuildValue("");
}

static PyObject * vector_gate(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    PyObject * booleans_obj;
    PyObject * vectors_obj;
    PyObject * result_obj;
    if (!PyArg_ParseTuple(args, "OOOO", &tensor_obj, &booleans_obj, &vectors_obj, &result_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    Tensor* booleans = (Tensor*) PyCapsule_GetPointer(booleans_obj, "Tensor");
    Tensor* vectors = (Tensor*) PyCapsule_GetPointer(vectors_obj, "Tensor");
    Tensor* result = (Tensor*) PyCapsule_GetPointer(result_obj, "Tensor");
    vector_gate_wrapper(tensor, booleans, vectors, result);
    return Py_BuildValue("");
}

static PyObject * vector_resize(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    PyObject * vectors_obj;
    PyObject * result_obj;
    if (!PyArg_ParseTuple(args, "OOO", &tensor_obj, &vectors_obj, &result_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    Tensor* vectors = (Tensor*) PyCapsule_GetPointer(vectors_obj, "Tensor");
    Tensor* result = (Tensor*) PyCapsule_GetPointer(result_obj, "Tensor");
    vector_resize_tensor_wrapper(tensor, vectors, result);
    return Py_BuildValue("");
}

static PyObject * negate(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    unsigned int scale_factor;
    PyObject * result_obj;
    if (!PyArg_ParseTuple(args, "OIO", &tensor_obj, &scale_factor, &result_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    Tensor* result = (Tensor*) PyCapsule_GetPointer(result_obj, "Tensor");
    negate_tensor_wrapper(tensor, result);
    return Py_BuildValue("");
}



static PyObject * enlarge(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    unsigned int scale_factor;
    PyObject * result_obj;
    if (!PyArg_ParseTuple(args, "OIO", &tensor_obj, &scale_factor, &result_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    Tensor* result = (Tensor*) PyCapsule_GetPointer(result_obj, "Tensor");
    tensor_enlarge_wrapper(tensor, scale_factor, result);
    return Py_BuildValue("");
}

static PyObject * lesser(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    PyObject * other_obj;
    PyObject * result_obj;
    if (!PyArg_ParseTuple(args, "OOO", &tensor_obj, &other_obj, &result_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    Tensor* other = (Tensor*) PyCapsule_GetPointer(other_obj, "Tensor");
    Tensor* result = (Tensor*) PyCapsule_GetPointer(result_obj, "Tensor");
    lesser_tensor_wrapper(tensor, other, result);
    return Py_BuildValue("");
}

static PyObject * greater(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    PyObject * other_obj;
    PyObject * result_obj;
    if (!PyArg_ParseTuple(args, "OOO", &tensor_obj, &other_obj, &result_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    Tensor* other = (Tensor*) PyCapsule_GetPointer(other_obj, "Tensor");
    Tensor* result = (Tensor*) PyCapsule_GetPointer(result_obj, "Tensor");
    greater_tensor_wrapper(tensor, other, result);
    return Py_BuildValue("");
}

static PyObject * lesser_equals(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    PyObject * other_obj;
    PyObject * result_obj;
    if (!PyArg_ParseTuple(args, "OOO", &tensor_obj, &other_obj, &result_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    Tensor* other = (Tensor*) PyCapsule_GetPointer(other_obj, "Tensor");
    Tensor* result = (Tensor*) PyCapsule_GetPointer(result_obj, "Tensor");
    lesser_equals_tensor_wrapper(tensor, other, result);
    return Py_BuildValue("");
}

static PyObject * greater_equals(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    PyObject * other_obj;
    PyObject * result_obj;
    if (!PyArg_ParseTuple(args, "OOO", &tensor_obj, &other_obj, &result_obj)) {
        return NULL;
    }
    Tensor* tensor = (Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    Tensor* other = (Tensor*) PyCapsule_GetPointer(other_obj, "Tensor");
    Tensor* result = (Tensor*) PyCapsule_GetPointer(result_obj, "Tensor");
    greater_equals_tensor_wrapper(tensor, other, result);
    return Py_BuildValue("");
}

static PyMethodDef tensorplus_methods[] = {
    {"create", create, METH_VARARGS, "Creates a tensor"},
    {"destroy", destroy, METH_VARARGS, "Destroys a tensor"},
    {"copy", copy, METH_VARARGS, "Copies a tensor by reference"},
    {"clone_t", clone_t, METH_VARARGS, "Copies a tensor by value"},
    {"create_ct", create_ct, METH_VARARGS, "Creates a cpu tensor"},
    {"destroy_ct", destroy_ct, METH_VARARGS, "Destroys a cpu tensor"},
    {"zeros", zeros, METH_VARARGS, "Creates a tensor of zeros"},
    {"ones", ones, METH_VARARGS, "Creates a tensor of ones"},
    {"print", printtn, METH_VARARGS, "Prints the tensor"},
    {"fill", fill, METH_VARARGS, "Fills the tensor with a value"},
    {"set", set, METH_VARARGS, "Sets an element of a tensor to a value"},
    {"add", add, METH_VARARGS, "Adds two tensors"},
    {"sub", sub, METH_VARARGS, "Subtracts two tensors"},
    {"mul", mul, METH_VARARGS, "Multiplies two tensors"},
    {"div", div_tn, METH_VARARGS, "Divides two tensors"},
    {"add_scalar", add_scalar, METH_VARARGS, "Adds a scalar to a tensor"},
    {"sub_scalar", sub_scalar, METH_VARARGS, "Subtracts a scalar from a tensor"},
    {"mul_scalar", mul_scalar, METH_VARARGS, "Multiplies a tensor by a scalar"},
    {"div_scalar", div_scalar, METH_VARARGS, "Divides a tensor by a scalar"},
    {"transpose", transpose, METH_VARARGS, "Transposes a tensor"},
    {"sum", sum, METH_VARARGS, "Computes the sum of a tensor"},
    {"max", max, METH_VARARGS, "Finds the maximum value in a tensor"},
    {"min", min, METH_VARARGS, "Finds the minimum value in a tensor"},
    {"gradient", gradient, METH_VARARGS, "Computes the gradient of a tensor"},
    {"gate", gate, METH_VARARGS, "Gates a tensor based on a boolean tensor"},
    {"vector_sort", vector_sort, METH_VARARGS, "Sorts a tensor based on a vector tensor"},
    {"vector_add", vector_add, METH_VARARGS, "Adds two tensors based on a vector tensor"},
    {"vector_sub", vector_sub, METH_VARARGS, "Subtracts two tensors based on a vector tensor"},
    {"vector_mul", vector_mul, METH_VARARGS, "Multiplies two tensors based on a vector tensor"},
    {"vector_div", vector_div, METH_VARARGS, "Divides two tensors based on a vector tensor"},
    {"vector_gate", vector_gate, METH_VARARGS, "Gates a tensor based on a boolean vector tensor"},
    {"vector_resize", vector_resize, METH_VARARGS, "Enlarges a tensor and sets its values according to the vector."},
    {"negate", negate, METH_VARARGS, "Returns true for zero values and false for nonzero values"},
    {"enlarge", enlarge, METH_VARARGS, "Enlarges a tensor by a whole number scale factor"},
    {"lesser", lesser, METH_VARARGS, "Returns true if source is less than other"},
    {"greater", greater, METH_VARARGS, "Returns true if source is greater than other"},
    {"lesser_equals", lesser_equals, METH_VARARGS, "Returns true if source is less than or equal to other"},
    {"greater_equals", greater_equals, METH_VARARGS, "Returns true if source is greater than or equal to other"},
    {NULL, NULL, 0, NULL}
};

static PyModuleDef tensorplus = {
    PyModuleDef_HEAD_INIT,
    "tensorplus",
    PyDoc_STR("A module for tensor operations"),
    -1,
    tensorplus_methods
};

PyMODINIT_FUNC PyInit_tensorplus(void) {
    return PyModule_Create(&tensorplus);
}
