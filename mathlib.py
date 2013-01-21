import math
import random
import time

"""
Find squared length of a vector.
"""
def length2(v):
    length2 = 0.0
    for x_i in v:
        length2 += x_i**2
    return length2

def length(v):
    return math.sqrt(length2(v))

"""
Normalize 3D vector.
A normalized vector points in the same direction of the original
vector but has unit length (length = 1).
"""
def normalize(v):
    length = math.sqrt(length2(v))
    return [v[i] / length for i in range(3)]

"""
Add two 3D vectors.
"""
def add(u, v):
    return [u[i] + v[i] for i in range(3)]

"""
Subtract two 3D vectors.
"""
def sub(u, v):
    return [u[i] - v[i] for i in range(3)]

"""
Multiply a 3D vector by a scalar.
"""
def mul(s, v):
    return [s * v[i] for i in range(3)]

"""
Find cross product of two 3D vectors.
"""
def cross(u, v):
    c = [0, 0, 0]
    c[0] += u[1] * v[2] - v[1] * u[2]
    c[1] -= u[0] * v[2] - v[0] * u[2]
    c[2] += u[0] * v[1] - v[0] * u[1]
    return c

"""
Find dot product of two 3D vectors.
"""
def dot(u, v):
    c = u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
    return c

"""
Multiply two matrices (stored as lists of lists).
"""
def matrix_mul(A, B):
    AB = [[None for x in range( len(B[0]) )] for y in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            AB[i][j] = 0
            for r in range(len(B)):
                AB[i][j] += A[i][r] * B[r][j]
    if len(AB) == 1:
        AB = AB[0]
    return AB

def transpose(A):
    T = list(A)
    for i in range( len(A) ):
        for j in range( len(A[0]) ):
            T[i][j] = A[j][i]
    return T

"""
Create a 3D rotation matrix (3x3) for a given angle
about an arbitrary axis. Meant to multiply a column
vector.
"""
def rotation_matrix_3D(axis, angle):
    x, y, z = normalize(axis)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotation_matrix = [0,0,0]
    rotation_matrix[0] = [
        cos_a +  x**2 * (1 - cos_a),
        x * y * (1 - cos_a) - z * sin_a,
        x * z * (1 - cos_a) + y * sin_a,
    ]
    rotation_matrix[1] = [
        y * x * (1 - cos_a) + z * sin_a,
        cos_a + y**2 * (1 - cos_a),
        y * z * (1 - cos_a) - x * sin_a,
    ]
    rotation_matrix[2] = [
        z * x * (1 - cos_a) - y * sin_a,
        z * y * (1 - cos_a) + x * sin_a,
        cos_a + z**2 * (1 - cos_a),
    ]
    return rotation_matrix

"""
Builds a rotation matrix for rotating a 2D column vector.
"""
def rotation_matrix_2D(angle):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotation_matrix = [
        [cos_a, -sin_a],
        [sin_a, cos_a],
    ]
    return rotation_matrix
