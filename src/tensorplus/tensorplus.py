import tensorplus

def create(size):
    return tensorplus.create(size)

def destroy(tensor):
    tensorplus.destroy(tensor)

def size(tensor, size):
    tensorplus.size(tensor, size)
    
def create_ct(size):
    return tensorplus.create_ct(size)

def destroy_ct(tensor):
    tensorplus.destroy_ct(tensor)

def set(tensor, index, value):
    tensorplus.set(tensor, index, value)
    
def get(tensor, index, value):
    tensorplus.get(tensor, index, value)

def copy(tensor, result):
    tensorplus.copy(tensor, result)

def clone_t(tensor, result):
    tensorplus.clone_t(tensor, result)

def zeros(tensor):
    tensorplus.zeros(tensor)
    
def ones(tensor):
    tensorplus.ones(tensor)
    
def fill(tensor, value):
    tensorplus.fill(tensor, value)
    
def add(tensor, value, result):
    tensorplus.add(tensor, value, result)
    
def sub(tensor, value, result):
    tensorplus.sub(tensor, value, result)
    
def mul(tensor, value, result):
    tensorplus.mul(tensor, value, result)
    
def div(tensor, value, result):
    tensorplus.div(tensor, value, result)

def add_scalar(tensor, scalar, result):
    tensorplus.add_scalar(tensor, scalar, result)
    
def sub_scalar(tensor, scalar, result):
    tensorplus.sub_scalar(tensor, scalar, result)
    
def mul_scalar(tensor, scalar, result):
    tensorplus.mul_scalar(tensor, scalar, result)

def div_scalar(tensor, scalar, result):
    tensorplus.div_scalar(tensor, scalar, result)
    
def transpose(tensor, result):
    tensorplus.transpose(tensor, result)

def sum(tensor, result):
    tensorplus.sum(tensor, result)

def max(tensor, result):
    tensorplus.max(tensor, result)
    
def min(tensor, result):
    tensorplus.min(tensor, result)
    
def gradient(tensor, result):
    tensorplus.gradient(tensor, result)

def gate(tensor, booleans, result):
    tensorplus.gate(tensor, booleans, result)

def vector_sort(tensor, vectors, result):
    tensorplus.vector_sort(tensor, vectors, result)

def vector_add(tensor, other, vectors, result):
    tensorplus.vector_add(tensor, other, vectors, result)

def vector_sub(tensor, other, vectors, result):
    tensorplus.vector_sub(tensor, other, vectors, result)

def vector_mul(tensor, other, vectors, result):
    tensorplus.vector_mul(tensor, other, vectors, result)

def vector_div(tensor, other, vectors, result):
    tensorplus.vector_div(tensor, other, vectors, result)

def vector_gate(tensor, booleans, vectors, result):
    tensorplus.vector_gate(tensor, booleans, vectors, result)

def vector_resize(tensor, vectors, result):
    tensorplus.vector_resize(tensor, vectors, result)

def enlarge(tensor, scale_factor, result):
    tensorplus.enlarge(tensor, scale_factor, result)

def negate(tensor, result):
    tensorplus.negate(tensor, result)

def lesser(tensor, other, result):
    tensorplus.lesser(tensor, other, result)

def greater(tensor, other, result):
    tensorplus.greater(tensor, other, result)

def lesser_equals(tensor, other, result):
    tensorplus.lesser_equals(tensor, other, result)

def greater_equals(tensor, other, result):
    tensorplus.greater_equals(tensor, other, result)

def equals(tensor, other, result):
    tensorplus.equals(tensor, other, result)
