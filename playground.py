import ivy
ivy.set_backend('numpy')
x, y = ivy.array([0, 0], dtype=ivy.uint16), ivy.array([1, 1], dtype=ivy.uint16)
print(ivy.intersection(x, y, False, False, False))