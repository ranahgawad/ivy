import ivy
ivy.set_backend('tensorflow')
x, y = ivy.array([1, 0], dtype=ivy.uint8), ivy.array([1, 1], dtype=ivy.uint8)
print(ivy.intersection(x, y, False, True, False))