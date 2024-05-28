#define PY_SSIZE_T_CLEAN
#include "python3.12/Python.h"
#include "includes.h"


void print_tensor_wrapper(struct Tensor* tensor);
void fill_tensor_wrapper(struct Tensor* tensor, int value);
void add_tensor_wrapper(struct Tensor* tensor, struct Tensor* other, struct Tensor* result);
void sub_tensor_wrapper(struct Tensor* tensor, struct Tensor* other, struct Tensor* result);
void mul_tensor_wrapper(struct Tensor* tensor, struct Tensor* other, struct Tensor* result);
void div_tensor_wrapper(struct Tensor* tensor, struct Tensor* other, struct Tensor* result);
void add_scalar_tensor_wrapper(struct Tensor* tensor, int scalar, struct Tensor* result);
void sub_scalar_tensor_wrapper(struct Tensor* tensor, int scalar, struct Tensor* result);
void mul_scalar_tensor_wrapper(struct Tensor* tensor, int scalar, struct Tensor* result);
void div_scalar_tensor_wrapper(struct Tensor* tensor, int scalar, struct Tensor* result);
void dot_tensor_wrapper(struct Tensor* tensor, struct Tensor* other, struct Tensor* result);
void transpose_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void sum_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void max_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void min_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void abs_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void exp_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void log_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void sqrt_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void pow_tensor_wrapper(struct Tensor* tensor, int power, struct Tensor* result);
void sin_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void cos_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void tan_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void asin_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void acos_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void atan_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void sinh_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void cosh_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void tanh_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void asinh_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void acosh_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void atanh_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void gradient_tensor_wrapper(struct Tensor* tensor, struct Tensor* result);
void gate_tensor_wrapper(struct Tensor* tensor, struct Tensor* booleans, struct Tensor* result);
void vector_sort_tensor_wrapper(struct Tensor* tensor, struct Tensor* vectors, struct Tensor* result);


static PyObject* print_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    print_tensor_wrapper(tensor);
    return Py_BuildValue("");
}

static PyObject* fill_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    int value;
    if (!PyArg_ParseTuple(args, "Oi", &tensor_obj, &value)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    fill_tensor_wrapper(tensor, value);
    return Py_BuildValue("");
}

static PyObject* add_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    PyObject* other_obj;
    if (!PyArg_ParseTuple(args, "OO", &tensor_obj, &other_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* other = (struct Tensor*) PyCapsule_GetPointer(other_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    add_tensor_wrapper(tensor, other, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* sub_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    PyObject* other_obj;
    if (!PyArg_ParseTuple(args, "OO", &tensor_obj, &other_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* other = (struct Tensor*) PyCapsule_GetPointer(other_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    sub_tensor_wrapper(tensor, other, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* mul_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    PyObject* other_obj;
    if (!PyArg_ParseTuple(args, "OO", &tensor_obj, &other_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* other = (struct Tensor*) PyCapsule_GetPointer(other_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    mul_tensor_wrapper(tensor, other, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* div_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    PyObject* other_obj;
    if (!PyArg_ParseTuple(args, "OO", &tensor_obj, &other_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* other = (struct Tensor*) PyCapsule_GetPointer(other_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    div_tensor_wrapper(tensor, other, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* add_scalar_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    int scalar;
    if (!PyArg_ParseTuple(args, "Oi", &tensor_obj, &scalar)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    add_scalar_tensor_wrapper(tensor, scalar, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* sub_scalar_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    int scalar;
    if (!PyArg_ParseTuple(args, "Oi", &tensor_obj, &scalar)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    sub_scalar_tensor_wrapper(tensor, scalar, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* mul_scalar_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    int scalar;
    if (!PyArg_ParseTuple(args, "Oi", &tensor_obj, &scalar)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    mul_scalar_tensor_wrapper(tensor, scalar, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* div_scalar_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    int scalar;
    if (!PyArg_ParseTuple(args, "Oi", &tensor_obj, &scalar)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    div_scalar_tensor_wrapper(tensor, scalar, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* dot_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    PyObject* other_obj;
    if (!PyArg_ParseTuple(args, "OO", &tensor_obj, &other_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* other = (struct Tensor*) PyCapsule_GetPointer(other_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    dot_tensor_wrapper(tensor, other, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* transpose_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    transpose_tensor_wrapper(tensor, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* sum_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    sum_tensor_wrapper(tensor, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* max_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    max_tensor_wrapper(tensor, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* min_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    min_tensor_wrapper(tensor, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* abs_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    abs_tensor_wrapper(tensor, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* exp_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    exp_tensor_wrapper(tensor, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* log_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    log_tensor_wrapper(tensor, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* sqrt_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    sqrt_tensor_wrapper(tensor, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* pow_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    int power;
    if (!PyArg_ParseTuple(args, "Oi", &tensor_obj, &power)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    pow_tensor_wrapper(tensor, power, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* sin_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    sin_tensor_wrapper(tensor, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* cos_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    cos_tensor_wrapper(tensor, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* tan_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    tan_tensor_wrapper(tensor, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* asin_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    asin_tensor_wrapper(tensor, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* acos_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    acos_tensor_wrapper(tensor, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* atan_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    atan_tensor_wrapper(tensor, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* sinh_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    sinh_tensor_wrapper(tensor, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* cosh_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    cosh_tensor_wrapper(tensor, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* tanh_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    tanh_tensor_wrapper(tensor, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* asinh_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    asinh_tensor_wrapper(tensor, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* acosh_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    acosh_tensor_wrapper(tensor, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* atanh_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    atanh_tensor_wrapper(tensor, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* gradient_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    gradient_tensor_wrapper(tensor, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* gate_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    PyObject* booleans_obj;
    if (!PyArg_ParseTuple(args, "OO", &tensor_obj, &booleans_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* booleans = (struct Tensor*) PyCapsule_GetPointer(booleans_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    gate_tensor_wrapper(tensor, booleans, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyObject* vector_sort_tensor(PyObject* self, PyObject* args) {
    PyObject* tensor_obj;
    PyObject* vectors_obj;
    if (!PyArg_ParseTuple(args, "OO", &tensor_obj, &vectors_obj)) {
        return NULL;
    }
    struct Tensor* tensor = (struct Tensor*) PyCapsule_GetPointer(tensor_obj, "Tensor");
    struct Tensor* vectors = (struct Tensor*) PyCapsule_GetPointer(vectors_obj, "Tensor");
    struct Tensor* result = (struct Tensor*) malloc(sizeof(struct Tensor));
    vector_sort_tensor_wrapper(tensor, vectors, result);
    return PyCapsule_New(result, "Tensor", NULL);
}

static PyMethodDef tensorplus_methods[] = {
    {"print_tensor", print_tensor, METH_VARARGS, "Prints the tensor"},
    {"fill_tensor", fill_tensor, METH_VARARGS, "Fills the tensor with a value"},
    {"add_tensor", add_tensor, METH_VARARGS, "Adds two tensors"},
    {"sub_tensor", sub_tensor, METH_VARARGS, "Subtracts two tensors"},
    {"mul_tensor", mul_tensor, METH_VARARGS, "Multiplies two tensors"},
    {"div_tensor", div_tensor, METH_VARARGS, "Divides two tensors"},
    {"add_scalar_tensor", add_scalar_tensor, METH_VARARGS, "Adds a scalar to a tensor"},
    {"sub_scalar_tensor", sub_scalar_tensor, METH_VARARGS, "Subtracts a scalar from a tensor"},
    {"mul_scalar_tensor", mul_scalar_tensor, METH_VARARGS, "Multiplies a tensor by a scalar"},
    {"div_scalar_tensor", div_scalar_tensor, METH_VARARGS, "Divides a tensor by a scalar"},
    {"dot_tensor", dot_tensor, METH_VARARGS, "Computes the dot product of two tensors"},
    {"transpose_tensor", transpose_tensor, METH_VARARGS, "Transposes a tensor"},
    {"sum_tensor", sum_tensor, METH_VARARGS, "Computes the sum of a tensor"},
    {"max_tensor", max_tensor, METH_VARARGS, "Finds the maximum value in a tensor"},
    {"min_tensor", min_tensor, METH_VARARGS, "Finds the minimum value in a tensor"},
    {"abs_tensor", abs_tensor, METH_VARARGS, "Computes the absolute value of a tensor"},
    {"exp_tensor", exp_tensor, METH_VARARGS, "Computes the exponential of a tensor"},
    {"log_tensor", log_tensor, METH_VARARGS, "Computes the natural logarithm of a tensor"},
    {"sqrt_tensor", sqrt_tensor, METH_VARARGS, "Computes the square root of a tensor"},
    {"pow_tensor", pow_tensor, METH_VARARGS, "Raises a tensor to a power"},
    {"sin_tensor", sin_tensor, METH_VARARGS, "Computes the sine of a tensor"},
    {"cos_tensor", cos_tensor, METH_VARARGS, "Computes the cosine of a tensor"},
    {"tan_tensor", tan_tensor, METH_VARARGS, "Computes the tangent of a tensor"},
    {"asin_tensor", asin_tensor, METH_VARARGS, "Computes the arcsine of a tensor"},
    {"acos_tensor", acos_tensor, METH_VARARGS, "Computes the arccosine of a tensor"},
    {"atan_tensor", atan_tensor, METH_VARARGS, "Computes the arctangent of a tensor"},
    {"sinh_tensor", sinh_tensor, METH_VARARGS, "Computes the hyperbolic sine of a tensor"},
    {"cosh_tensor", cosh_tensor, METH_VARARGS, "Computes the hyperbolic cosine of a tensor"},
    {"tanh_tensor", tanh_tensor, METH_VARARGS, "Computes the hyperbolic tangent of a tensor"},
    {"asinh_tensor", asinh_tensor, METH_VARARGS, "Computes the hyperbolic arcsine of a tensor"},
    {"acosh_tensor", acosh_tensor, METH_VARARGS, "Computes the hyperbolic arccosine of a tensor"},
    {"atanh_tensor", atanh_tensor, METH_VARARGS, "Computes the hyperbolic arctangent of a tensor"},
    {"gradient_tensor", gradient_tensor, METH_VARARGS, "Computes the gradient of a tensor"},
    {"gate_tensor", gate_tensor, METH_VARARGS, "Gates a tensor based on a boolean tensor"},
    {"vector_sort_tensor", vector_sort_tensor, METH_VARARGS, "Sorts a tensor based on a vector tensor"},
    {NULL, NULL, 0, NULL}
};

static PyModuleDef tensorplus = {
    PyModuleDef_HEAD_INIT,
    "tensorplus",
    PyDoc_STR("A module for tensor operations"),
    -1,
    tensorplus_methods
};