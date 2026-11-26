from tensorplus import tps as tp
import pytest

tensor_size = 4
tensor = tp.create(tensor_size)
other = tp.create(tensor_size)
vectors = tp.create(tensor_size)
result = tp.create(tensor_size)

def test_zeros():
    print("Testing zeros()...")
    tp.zeros(tensor)
    num = -1
    print("Testing get()...")
    num = tp.get(tensor, 1)
    assert num == 0
    return

def test_ones():
    print("Testing ones()...")
    tp.ones(other)
    num = -1
    num = tp.get(other, 0)
    assert num == 1
    return

def test_fill():
    print("Testing fill()...")
    tp.fill(tensor, 2)
    num = -1
    num = tp.get(tensor, 0)
    assert num == 2
    return

def test_add():
    print("Testing add()...")
    tp.add(tensor, other, result)
    num = -1
    num = tp.get(result, 0)
    assert num == 3
    return

def test_print():
    print("Testing print()...")
    tp.print(result)
    tp.print(tensor)
    tp.print(other)
    return

def test_size():
    print("Testing size()...")
    num = -1
    num = tp.size(tensor)
    assert num == 4
    return

def test_set():
    print("Testing set...")
    num = 5
    ind = 1
    tp.set(tensor, ind, num)
    assert tp.get(tensor, ind) == num
    return

def test_sub():
    print("Testing sub()...")
    tp.fill(tensor, 2)
    tp.fill(other, 1)
    tp.sub(tensor, other, result)
    num = -1
    num = tp.get(result, 0)
    assert num == 0
    return

def test_mul():
    print("Testing mul()...")
    tp.fill(tensor, 2)
    tp.fill(other, 3)
    tp.mul(tensor, other, result)
    num = -1
    num = tp.get(result, 0)
    assert num == 6
    return

def test_div():
    print("Testing div()...")
    tp.fill(tensor, 4)
    tp.fill(other, 2)
    tp.mul(tensor, other, result)
    num = -1
    num = tp.get(result, 0)
    assert num == 2
    return

def test_add_scalar():
    print("Testing add_scalar()...")
    tp.fill(tensor, 1)
    scalar = 2
    tp.add_scalar(tensor, scalar, result)
    num = -1
    num = tp.get(result, 0)
    assert num == 3
    return

def test_sub_scalar():
    print("Testing sub_scalar()...")
    tp.fill(tensor, 2)
    scalar = 1
    tp.sub_scalar(tensor, scalar, result)
    num = -1
    num = tp.get(result, 0)
    assert num == 1
    return

def test_mul_scalar():
    print("Testing mul_scalar()...")
    tp.fill(tensor, 2)
    scalar = 3
    tp.mul_scalar(tensor, scalar, result)
    num = -1
    num = tp.get(result, 0)
    assert num == 6
    return

def test_div_scalar():
    print("Testing mul_scalar()...")
    tp.fill(tensor, 6)
    scalar = 3
    tp.div_scalar(tensor, scalar, result)
    num = -1
    num = tp.get(result, 0)
    assert num == 2
    return

def test_div_scalar():
    print("Testing mul_scalar()...")
    tp.fill(tensor, 6)
    scalar = 2
    tp.div_scalar(tensor, scalar, result)
    num = -1
    num = tp.get(result, 0)
    assert num == 6
    return

def test_transpose():
    print("Testing transpose...")
    tp.set(tensor, 0, 0)
    tp.set(tensor, 1, 1)
    tp.set(tensor, 2, 2)
    tp.set(tensor, 3, 3)
    tp.transpose(tensor)
    num = -1
    num = tp.get(tensor, 2)
    assert num == 1
    return

def test_sum():
    print("Testing sum...")
    tp.fill(tensor, 1)
    num = -1
    num = tp.sum(tensor)
    assert num == 4
    return

def test_max():
    print("Testing max...")
    tp.set(tensor, 0, 0)
    tp.set(tensor, 1, 1)
    tp.set(tensor, 2, 2)
    tp.set(tensor, 3, 3)
    num = -1
    num = tp.max(tensor)
    assert num == 3
    return
    
def test_min():
    print("Testing min...")
    tp.set(tensor, 0, 0)
    tp.set(tensor, 1, 1)
    tp.set(tensor, 2, 2)
    tp.set(tensor, 3, 3)
    num = -1
    num = tp.max(tensor)
    assert num == 0
    return

def test_gradient():
    print("Testing gradient...")
    tp.set(tensor, 0, 0)
    tp.set(tensor, 1, 1)
    tp.set(tensor, 2, 2)
    tp.set(tensor, 3, 3)
    num = -1
    num2 = -1
    num3 = -1
    num4 = -1
    tp.gradient(tensor)
    num = tp.get(tensor, 0)
    num2 = tp.get(tensor, 1)
    num3 = tp.get(tensor, 2)
    num4 = tp.get(tensor, 3)
    assert num == 1
    assert num2 == 1
    assert num3 == 1
    assert num4 == -3
    return

def test_gate():
    print("Testing gate...")
    tp.fill(tensor, 1)
    tp.fill(other, 0)
    tp.set(other, 0, 1)
    tp.set(other, 1, 2)
    tp.gate(tensor, other, result)
    num1 = -1
    num2 = -1
    num3 = -1
    num4 = -1
    num1 = tp.get(result, 0)
    num2 = tp.get(result, 1)
    num3 = tp.get(result, 2)
    num4 = tp.get(result, 3)
    assert num1 == 1
    assert num2 == 1
    assert num3 == 0
    assert num4 == 0
    return

def test_vector_sort():
    print("Testing vector_sort...")
    tp.set(tensor, 0, 0)
    tp.set(tensor, 1, 1)
    tp.set(tensor, 2, 2)
    tp.set(tensor, 3, 3)
    tp.set(other, 0, 3)
    tp.set(other, 1, 2)
    tp.set(other, 2, 1)
    tp.set(other, 3, 0)
    num = -1
    num2 = -1
    num3 = -1
    num4 = -1
    num = tp.get(result, 0)
    num2 = tp.get(result, 1)
    num3 = tp.get(result, 2)
    num4 = tp.get(result, 3)
    tp.vector_sort(tensor, other, result)
    assert num == 3
    assert num2 == 2
    assert num3 == 1
    assert num4 == 0
    return

def test_vector_add():
    print("Testing vector_add...")
    tp.set(tensor, 0, 0)
    tp.set(tensor, 1, 1)
    tp.set(tensor, 2, 2)
    tp.set(tensor, 3, 3)
    tp.set(other, 0, 4)
    tp.set(other, 1, 5)
    tp.set(other, 2, 6)
    tp.set(other, 3, 7)
    tp.set(vectors, 0, 1)
    tp.set(vectors, 1, 2)
    tp.set(vectors, 2, 3)
    tp.set(vectors, 3, 4)
    num = -1
    num2 = -1
    num3 = -1
    num4 = -1
    tp.vector_add(tensor, other, vectors, result)
    num = tp.get(result, 0)
    num2 = tp.get(result, 1)
    num3 = tp.get(result, 2)
    num4 = tp.get(result, 3)
    assert num == 5
    assert num2 == 7
    assert num3 == 9
    assert num4 == 7
    return

def test_vector_sub():
    print("Testing vector_sub...")
    tp.set(tensor, 0, 0)
    tp.set(tensor, 1, 1)
    tp.set(tensor, 2, 2)
    tp.set(tensor, 3, 3)
    tp.set(other, 0, 4)
    tp.set(other, 1, 5)
    tp.set(other, 2, 6)
    tp.set(other, 3, 7)
    tp.set(vectors, 0, 1)
    tp.set(vectors, 1, 2)
    tp.set(vectors, 2, 3)
    tp.set(vectors, 3, 4)
    num = -2
    num2 = -2
    num3 = -2
    num4 = -2
    tp.vector_sub(tensor, other, vectors, result)
    num = tp.get(result, 0)
    num2 = tp.get(result, 1)
    num3 = tp.get(result, 2)
    num4 = tp.get(result, 3)
    assert num == -5
    assert num2 == -5
    assert num3 == -5
    assert num4 == -1
    return

def test_vector_mul():
    print("Testing vector_mul...")
    tp.set(tensor, 0, 0)
    tp.set(tensor, 1, 1)
    tp.set(tensor, 2, 2)
    tp.set(tensor, 3, 3)
    tp.set(other, 0, 4)
    tp.set(other, 1, 5)
    tp.set(other, 2, 6)
    tp.set(other, 3, 7)
    tp.set(vectors, 0, 1)
    tp.set(vectors, 1, 2)
    tp.set(vectors, 2, 3)
    tp.set(vectors, 3, 4)
    num = -1
    num2 = -1
    num3 = -1
    num4 = -1
    tp.vector_mul(tensor, other, vectors, result)
    num = tp.get(result, 0)
    num2 = tp.get(result, 1)
    num3 = tp.get(result, 2)
    num4 = tp.get(result, 3)
    assert num == 0
    assert num2 == 6
    assert num3 == 14
    assert num4 == 12
    return

def test_vector_div():
    print("Testing vector_div...")
    tp.set(tensor, 0, 4)
    tp.set(tensor, 1, 6)
    tp.set(tensor, 2, 8)
    tp.set(tensor, 3, 10)
    tp.set(other, 0, 5)
    tp.set(other, 1, 2)
    tp.set(other, 2, 3)
    tp.set(other, 3, 4)
    tp.set(vectors, 0, 1)
    tp.set(vectors, 1, 2)
    tp.set(vectors, 2, 3)
    tp.set(vectors, 3, 4)
    num = -1
    num2 = -1
    num3 = -1
    num4 = -1
    tp.vector_div(tensor, other, vectors, result)
    num = tp.get(result, 0)
    num2 = tp.get(result, 1)
    num3 = tp.get(result, 2)
    num4 = tp.get(result, 3)
    assert num == 2
    assert num2 == 2
    assert num3 == 2
    assert num4 == 2
    return

def test_vector_gate():
    print("Testing vector_gate...")
    tp.set(tensor, 0, 1)
    tp.set(tensor, 1, 1)
    tp.set(tensor, 2, 1)
    tp.set(tensor, 3, 1)
    tp.set(other, 0, 0)
    tp.set(other, 1, 5)
    tp.set(other, 2, 0)
    tp.set(other, 3, 1)
    tp.set(vectors, 0, 1)
    tp.set(vectors, 1, 2)
    tp.set(vectors, 2, 3)
    tp.set(vectors, 3, 4)
    num = -1
    num2 = -1
    num3 = -1
    num4 = -1
    tp.vector_gate(tensor, other, vectors, result)
    num = tp.get(result, 0)
    num2 = tp.get(result, 1)
    num3 = tp.get(result, 2)
    num4 = tp.get(result, 3)
    assert num == 1
    assert num2 == 0
    assert num3 == 1
    assert num4 == 0
    return

def test_vector_resize():
    print("Skipping vector_resize for now...")
    return

def test_enlarge():
    print("Skipping enlarge for now...")
    return

def test_negate():
    print("Testing negate...")
    tp.set(tensor, 0, 1)
    tp.set(tensor, 1, 0)
    tp.set(tensor, 2, 11)
    tp.set(tensor, 3, 0)
    num = -1
    num2 = -1
    num3 = -1
    num4 = -1
    tp.negate(tensor, result)
    num = tp.get(result, 0)
    num2 = tp.get(result, 1)
    num3 = tp.get(result, 2)
    num4 = tp.get(result, 3)
    assert num == 0
    assert num2 == 1
    assert num3 == 0
    assert num4 == 1
    return

def test_lesser():
    print("Testing lesser...")
    tp.set(tensor, 0, 1)
    tp.set(tensor, 1, 1)
    tp.set(tensor, 2, 1)
    tp.set(tensor, 3, 1)
    tp.set(other, 0, 0)
    tp.set(other, 1, 5)
    tp.set(other, 2, 0)
    tp.set(other, 3, 1)
    num = -1
    num2 = -1
    num3 = -1
    num4 = -1
    tp.lesser(tensor, other, result)
    num = tp.get(result, 0)
    num2 = tp.get(result, 1)
    num3 = tp.get(result, 2)
    num4 = tp.get(result, 3)
    assert num == 0
    assert num2 == 1
    assert num3 == 0
    assert num4 == 1
    return

def test_greater():
    print("Testing greater...")
    tp.set(tensor, 0, 1)
    tp.set(tensor, 1, 1)
    tp.set(tensor, 2, 1)
    tp.set(tensor, 3, 1)
    tp.set(other, 0, 0)
    tp.set(other, 1, 5)
    tp.set(other, 2, 0)
    tp.set(other, 3, 1)
    num = -1
    num2 = -1
    num3 = -1
    num4 = -1
    tp.greater(tensor, other, result)
    num = tp.get(result, 0)
    num2 = tp.get(result, 1)
    num3 = tp.get(result, 2)
    num4 = tp.get(result, 3)
    assert num == 1
    assert num2 == 5
    assert num3 == 1
    assert num4 == 1
    return

def test_lesser_equals():
    print("Testing lesser_equals...")
    tp.set(tensor, 0, 1)
    tp.set(tensor, 1, 1)
    tp.set(tensor, 2, 1)
    tp.set(tensor, 3, 1)
    tp.set(other, 0, 0)
    tp.set(other, 1, 5)
    tp.set(other, 2, 0)
    tp.set(other, 3, 1)
    num = -1
    num2 = -1
    num3 = -1
    num4 = -1
    tp.lesser_equals(tensor, other, result)
    num = tp.get(result, 0)
    num2 = tp.get(result, 1)
    num3 = tp.get(result, 2)
    num4 = tp.get(result, 3)
    assert num == 0
    assert num2 == 1
    assert num3 == 0
    assert num4 == 1
    return

def test_greater_equals():
    print("Testing greater_equals...")
    tp.set(tensor, 0, 1)
    tp.set(tensor, 1, 1)
    tp.set(tensor, 2, 1)
    tp.set(tensor, 3, 1)
    tp.set(other, 0, 0)
    tp.set(other, 1, 5)
    tp.set(other, 2, 0)
    tp.set(other, 3, 1)
    num = -1
    num2 = -1
    num3 = -1
    num4 = -1
    tp.greater_equals(tensor, other, result)
    num = tp.get(result, 0)
    num2 = tp.get(result, 1)
    num3 = tp.get(result, 2)
    num4 = tp.get(result, 3)
    assert num == 1
    assert num2 == 5
    assert num3 == 1
    assert num4 == 1
    return


tp.destroy(tensor)
tp.destroy(other)
tp.destroy(vectors)
tp.destroy(result)
