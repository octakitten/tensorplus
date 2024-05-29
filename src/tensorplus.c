#define PY_SSIZE_T_CLEAN
#include "python3.12/Python.h"
#include "includes.h"


static PyObject * printtn(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    print_tensor_wrapper(tensor);
    return Py_BuildValue("");
}

static PyObject * fill(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    int value;
    if (!PyArg_ParseTuple(args, "Oi", &tensor_obj, &value)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    fill_tensor_wrapper(tensor, value);
    return Py_BuildValue("");
}

static PyObject * add(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    PyObject * other_obj;
    if (!PyArg_ParseTuple(args, "OO", &tensor_obj, &other_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* other = (struct Tensor*) PyCapsule_GetPointer(other_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor*));
    add_tensor_wrapper(tensor, other, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject * sub(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    PyObject * other_obj;
    if (!PyArg_ParseTuple(args, "OO", &tensor_obj, &other_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* other = (struct Tensor*) PyCapsule_GetPointer(other_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor*));
    sub_tensor_wrapper(tensor, other, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject * mul(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    PyObject * other_obj;
    if (!PyArg_ParseTuple(args, "OO", &tensor_obj, &other_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* other = (struct Tensor*) PyCapsule_GetPointer(other_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor*));
    mul_tensor_wrapper(tensor, other, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject * div_tn(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    PyObject * other_obj;
    if (!PyArg_ParseTuple(args, "OO", &tensor_obj, &other_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* other = (struct Tensor*) PyCapsule_GetPointer(other_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor*));
    div_tensor_wrapper(tensor, other, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject * add_scalar(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    int scalar;
    if (!PyArg_ParseTuple(args, "Oi", &tensor_obj, &scalar)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor*));
    add_scalar_tensor_wrapper(tensor, scalar, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject * sub_scalar(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    int scalar;
    if (!PyArg_ParseTuple(args, "Oi", &tensor_obj, &scalar)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor*));
    sub_scalar_tensor_wrapper(tensor, scalar, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject * mul_scalar(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    int scalar;
    if (!PyArg_ParseTuple(args, "Oi", &tensor_obj, &scalar)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor*));
    mul_scalar_tensor_wrapper(tensor, scalar, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject * div_scalar(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    int scalar;
    if (!PyArg_ParseTuple(args, "Oi", &tensor_obj, &scalar)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor*));
    div_scalar_tensor_wrapper(tensor, scalar, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject * transpose(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor*));
    transpose_tensor_wrapper(tensor, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject * sum(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor*));
    sum_tensor_wrapper(tensor, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject * max(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor*));
    max_tensor_wrapper(tensor, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject * min(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor*));
    min_tensor_wrapper(tensor, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject * gradient(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor*));
    gradient_tensor_wrapper(tensor, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject * gate(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    PyObject * booleans_obj;
    if (!PyArg_ParseTuple(args, "OO", &tensor_obj, &booleans_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* booleans = (struct Tensor*) PyCapsule_GetPointer(booleans_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor*));
    gate_tensor_wrapper(tensor, booleans, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject * vector_sort(PyObject * self, PyObject * args) {
    PyObject * tensor_obj;
    PyObject * vectors_obj;
    if (!PyArg_ParseTuple(args, "OO", &tensor_obj, &vectors_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* vectors = (struct Tensor*) PyCapsule_GetPointer(vectors_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor*));
    vector_sort_tensor_wrapper(tensor, vectors, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyMethodDef tensorplus_methods[] = {
    {"print", printtn, METH_VARARGS, "Prints the tensor"},
    {"fill", fill, METH_VARARGS, "Fills the tensor with a value"},
    {"add", add, METH_VARARGS, "Adds two tensors"},
    {"sub", sub, METH_VARARGS, "Subtracts two tensors"},
    {"mul", mul, METH_VARARGS, "Multiplies two tensors"},
    {"div", div, METH_VARARGS, "Divides two tensors"},
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
    {NULL, NULL, 0, NULL}
};

static PyModuleDef tensorplus = {
    PyModuleDef_HEAD_INIT,
    "tensorplus",
    PyDoc_STR("A module for tensor operations"),
    -1,
    tensorplus_methods
};