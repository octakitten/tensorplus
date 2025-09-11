import tensorplus as tp
import pytest

tensor_size = 4
tensor = tp.create(tensor_size)
other = tp.create(tensor_size)
result = tp.create(tensor_size)

def test_zeros():
    print("Testing zeros()...")
    tp.zeros(tensor)
    num = 1
    print("Testing get()...")
    tp.get(tensor, 1, num)
    assert num == 0
    return

def test_ones():
    print("Testing ones()...")
    tp.ones(other)
    num = -1
    tp.get(tensor, 0, num)
    assert num == 1
    return

def test_fill():
    print("Testing fill()...")
    tp.fill(result, 2)
    num = -1
    tp.get(tensor, 0, num)
    assert num == 1
    return

def test_add():
    print("Testing add()...")
    tp.add(tensor, other, result)
    num = -1
    tp.get(result, 0, num)
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
    tp.size(tensor, num)
    assert num == 4
    return
