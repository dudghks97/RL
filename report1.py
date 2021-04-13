import numpy as np

a = np.array([[1, -0.45, -0.45, 0, 0, 0, 0, 0, 0],
              [-0.36, 1, -0.27, 0, 0, 0, 0, 0, -0.27],
              [0, 0, 1, -0.72, 0, -0.18, 0, 0, 0],
              [-0.54, 0, 0, 1, -0.36, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, -0.45, -0.45, 0, 0],
              [0, 0, 0, 0, 0, 1, -0.36, -0.27, -0.27],
              [0, 0, 0, 0, -0.18, 0, 1, -0.36, 0],
              [0, 0, 0, 0, 0, 0, -0.18, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, -0.54, 0.64]
              ])
print(a.shape)

inv_a = np.linalg.inv(a)

print('I-A 행렬의 역행렬')
print(inv_a)

b = np.array([[0], [0], [0], [-0.6], [0.5], [-0.5], [4], [1.6], [0]])

v = np.dot(inv_a, b)

print('V 행렬')
print(v.shape)
print(v)