'''from . import tps as _tp

def create(size):
    return _tp.create(size)

def destroy(tensor):
    _tp.destroy(tensor)

def size(tensor, size):
    _tp.size(tensor, size)
    
def create_ct(size):
    return _tp.create_ct(size)

def destroy_ct(tensor):
    _tp.destroy_ct(tensor)

def set(tensor, index, value):
    _tp.set(tensor, index, value)
    
def get(tensor, index, value):
    _tp.get(tensor, index, value)

def copy(tensor, result):
    _tp.copy(tensor, result)

def clone_t(tensor, result):
    _tp.clone_t(tensor, result)

def zeros(tensor):
    _tp.zeros(tensor)
    
def ones(tensor):
    _tp.ones(tensor)
    
def fill(tensor, value):
    _tp.fill(tensor, value)
    
def add(tensor, value, result):
    _tp.add(tensor, value, result)
    
def sub(tensor, value, result):
    _tp.sub(tensor, value, result)
    
def mul(tensor, value, result):
    _tp.mul(tensor, value, result)
    
def div(tensor, value, result):
    _tp.div(tensor, value, result)

def add_scalar(tensor, scalar, result):
    _tp.add_scalar(tensor, scalar, result)
    
def sub_scalar(tensor, scalar, result):
    _tp.sub_scalar(tensor, scalar, result)
    
def mul_scalar(tensor, scalar, result):
    _tp.mul_scalar(tensor, scalar, result)

def div_scalar(tensor, scalar, result):
    _tp.div_scalar(tensor, scalar, result)
    
def transpose(tensor, result):
    _tp.transpose(tensor, result)

def sum(tensor, result):
    _tp.sum(tensor, result)

def max(tensor, result):
    _tp.max(tensor, result)
    
def min(tensor, result):
    _tp.min(tensor, result)
    
def gradient(tensor, result):
    _tp.gradient(tensor, result)

def gate(tensor, booleans, result):
    _tp.gate(tensor, booleans, result)

def vector_sort(tensor, vectors, result):
    _tp.vector_sort(tensor, vectors, result)

def vector_add(tensor, other, vectors, result):
    _tp.vector_add(tensor, other, vectors, result)

def vector_sub(tensor, other, vectors, result):
    _tp.vector_sub(tensor, other, vectors, result)

def vector_mul(tensor, other, vectors, result):
    _tp.vector_mul(tensor, other, vectors, result)

def vector_div(tensor, other, vectors, result):
    _tp.vector_div(tensor, other, vectors, result)

def vector_gate(tensor, booleans, vectors, result):
    _tp.vector_gate(tensor, booleans, vectors, result)

def vector_resize(tensor, vectors, result):
    _tp.vector_resize(tensor, vectors, result)

def enlarge(tensor, scale_factor, result):
    _tp.enlarge(tensor, scale_factor, result)

def negate(tensor, result):
    _tp.negate(tensor, result)

def lesser(tensor, other, result):
    _tp.lesser(tensor, other, result)

def greater(tensor, other, result):
    _tp.greater(tensor, other, result)

def lesser_equals(tensor, other, result):
    _tp.lesser_equals(tensor, other, result)

def greater_equals(tensor, other, result):
    _tp.greater_equals(tensor, other, result)

def equals(tensor, other, result):
    _tp.equals(tensor, other, result)

'''
