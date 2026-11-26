from tensorplus import tps as tp
import pytest

tensor_size = 4
tensor = tp.create(tensor_size)
other = tp.create(tensor_size)
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

#tp.destroy(tensor)
#tp.destroy(other)
#tp.destroy(result)
