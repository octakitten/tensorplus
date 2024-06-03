from tensorplus import tensorplus as tp

tensor = tp.create(4)
other = tp.create(4)
result = tp.create(4)

tp.zeros(tensor)
tp.ones(other)
tp.zeros(result)

tp.add(tensor, other, result)
tp.print(result)