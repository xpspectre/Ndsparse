import numpy as np

a = np.array([[[1, 7, 3], [2, 8, 4]], [[3, 9, 5], [4, 0, 6]], [[5, 1, 7], [6, 2, 8]], [[0, 1, 9], [1, 0, 3]]])
b = np.array([[[5, 1], [7, 0], [8, 4], [0, 4]], [[0, 3], [1, 5], [9, 6], [1, 2]], [[4, 9], [3, 8], [6, 7], [2, 0]]])

# a = np.arange(60.).reshape(3,4,5)
# b = np.arange(24.).reshape(4,3,2)
c = np.tensordot(a, b, axes=([0, 1], [1, 2]))
x = c.shape
print('a\n', a)
print('b\n', b)
print('c\n', c)
print('shape', x)

# A slower but equivalent way of computing the same...
d = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        for k in range(4):
            for n in range(2):
                d[i, j] += a[k, n, i] * b[j, k, n]
print(c == d)
