
# created by YU Tiezheng on 28/8/2019

# a quick sort function
# def quicksort(arr):
#     if len(arr) <= 1:
#         return arr
#     pivot = arr[len(arr)//2]
#     left = [x for x in arr if x < pivot]
#     middle = [x for x in arr if x == pivot]
#     right = [x for x in arr if x > pivot]
#     return quicksort(left) + middle + quicksort(right)
# print(quicksort([3,6,8,10,1,2,1]))

# bool 
# t = True
# f = False
# print(type(t)) # Prints "<class 'bool'>"
# print(t and f) # Logical AND; prints "False"
# print(t or f)  # Logical OR; prints "True"
# print(not t)   # Logical NOT; prints "False"
# print(t != f)  # Logical XOR; prints "True"

# string
# hello = 'hello'    # String literals can use single quotes
# world = "world"    # or double quotes; it does not matter.
# print(hello)       # Prints "hello"
# print(len(hello))  # String length; prints "5"
# hw = hello + ' ' + world  # String concatenation
# print(hw)  # prints "hello world"
# hw12 = '%s %s %d' % (hello, world, 12)  # sprintf style string formatting
# print(hw12)  # prints "hello world 12"

# s = "hello"
# print(s.capitalize())  # Capitalize a string; prints "Hello"
# print(s.upper())       # Convert a string to uppercase; prints "HELLO"
# print(s.rjust(7))      # Right-justify a string, padding with spaces; prints "  hello"
# print(s.center(7))     # Center a string, padding with spaces; prints " hello "
# print(s.replace('l', '(ell)'))  # Replace all instances of one substring with another;
#                                 # prints "he(ell)(ell)o"
# print('  world '.strip())  # Strip leading and trailing whitespace; prints "world"

# List
# xs = [3, 1, 2]    # Create a list
# print(xs, xs[2])  # Prints "[3, 1, 2] 2"
# print(xs[-1])     # Negative indices count from the end of the list; prints "2"
# xs[2] = 'foo'     # Lists can contain elements of different types
# print(xs)         # Prints "[3, 1, 'foo']"
# xs.append('bar')  # Add a new element to the end of the list
# print(xs)         # Prints "[3, 1, 'foo', 'bar']"
# x = xs.pop()      # Remove and return the last element of the list
# print(x, xs)      # Prints "bar [3, 1, 'foo']"

# class
# class Greeter(object):

#     # Constructor
#     def __init__(self, name):
#         self.name = name  # Create an instance variable

#     # Instance method
#     def greet(self, loud=False):
#         if loud:
#             print('HELLO, %s!' % self.name.upper())
#         else:
#             print('Hello, %s' % self.name)

# g = Greeter('Fred')  # Construct an instance of the Greeter class
# g.greet()            # Call an instance method; prints "Hello, Fred"
# g.greet(loud=True)   # Call an instance method; prints "HELLO, FRED!"


# Numpy
# import numpy as np
# a = np.array([1, 2, 3])   # Create a rank 1 array
# print(type(a))            # Prints "<class 'numpy.ndarray'>"
# print(a.shape)            # Prints "(3,)"
# print(a[0], a[1], a[2])   # Prints "1 2 3"
# a[0] = 5                  # Change an element of the array
# print(a)                  # Prints "[5, 2, 3]"

# b = np.array([[1,2,3],[4,5,6]])    # Create a rank 2 array
# print(b.shape)                     # Prints "(2, 3)"
# print(b[0, 0], b[0, 1], b[1, 0])   # Prints "1 2 4"

# import numpy as np
# a = np.zeros((2,2))   # Create an array of all zeros
# print(a)              # Prints "[[ 0.  0.]
#                       #          [ 0.  0.]]"
# b = np.ones((1,2))    # Create an array of all ones
# print(b)              # Prints "[[ 1.  1.]]"
# c = np.full((2,2), 7)  # Create a constant array
# print(c)               # Prints "[[ 7.  7.]
#                        #          [ 7.  7.]]"
# d = np.eye(2)         # Create a 2x2 identity matrix
# print(d)              # Prints "[[ 1.  0.]
#                       #          [ 0.  1.]]"
# e = np.random.random((2,2))  # Create an array filled with random values
# print(e)                     # Might print "[[ 0.91940167  0.08143941]

# import numpy as np
# # Create the following rank 2 array with shape (3, 4)
# # [[ 1  2  3  4]
# #  [ 5  6  7  8]
# #  [ 9 10 11 12]]
# a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
# print("a = ",a)
# # Use slicing to pull out the subarray consisting of the first 2 rows
# # and columns 1 and 2; b is the following array of shape (2, 2):
# # [[2 3]
# #  [6 7]]
# b = a[:2, 1:3]
# print("b = ", b)
# # A slice of an array is a view into the same data, so modifying it
# # will modify the original array.
# print(a[0, 1])   # Prints "2"
# b[0, 0] = 77     # b[0, 0] is the same piece of data as a[0, 1]
# print(a[0, 1])   # Prints "77

# import numpy as np

# # Create the following rank 2 array with shape (3, 4)
# # [[ 1  2  3  4]
# #  [ 5  6  7  8]
# #  [ 9 10 11 12]]
# a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# # Two ways of accessing the data in the middle row of the array.
# # Mixing integer indexing with slices yields an array of lower rank,
# # while using only slices yields an array of the same rank as the
# # original array:
# row_r1 = a[1, :]    # Rank 1 view of the second row of a
# row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
# print(row_r1, row_r1.shape)  # Prints "[5 6 7 8] (4,)"
# print(row_r2, row_r2.shape)  # Prints "[[5 6 7 8]] (1, 4)"

# # We can make the same distinction when accessing columns of an array:
# col_r1 = a[:, 1]
# col_r2 = a[:, 1:2]
# print(col_r1, col_r1.shape)  # Prints "[ 2  6 10] (3,)"
# print(col_r2, col_r2.shape)  # Prints "[[ 2]
#                              #          [ 6]
#                              #          [10]] (3, 1)"

# import numpy as np

# a = np.array([[1,2], [3, 4], [5, 6]])

# bool_idx = (a > 2)   # Find the elements of a that are bigger than 2;
#                      # this returns a numpy array of Booleans of the same
#                      # shape as a, where each slot of bool_idx tells
#                      # whether that element of a is > 2.

# print(bool_idx)      # Prints "[[False False]
#                      #          [ True  True]
#                      #          [ True  True]]"

# # We use boolean array indexing to construct a rank 1 array
# # consisting of the elements of a corresponding to the True values
# # of bool_idx
# print(a[bool_idx])  # Prints "[3 4 5 6]"

# # We can do all of the above in a single concise statement:
# print(a[a > 2])     # Prints "[3 4 5 6]"

# import numpy as np

# x = np.array([1, 2])   # Let numpy choose the datatype
# print(x.dtype)         # Prints "int64"

# x = np.array([1.0, 2.0])   # Let numpy choose the datatype
# print(x.dtype)             # Prints "float64"

# x = np.array([1, 2], dtype=np.int64)   # Force a particular datatype
# print(x.dtype)                         # Prints "int64"
# print("the shape of x  is : ", x.shape)

# Gradient calculate
# import math
# x = 3 # example values
# y = -4

# # forward pass
# sigy = 1.0 / (1 + math.exp(-y)) # sigmoid in numerator   #(1)
# num = x + sigy # numerator                               #(2)
# sigx = 1.0 / (1 + math.exp(-x)) # sigmoid in denominator #(3)
# xpy = x + y                                              #(4)
# xpysqr = xpy**2                                          #(5)
# den = sigx + xpysqr # denominator                        #(6)
# invden = 1.0 / den                                       #(7)
# f = num * invden # done!                                 #(8)

# Gradients for vectorized operations
# import numpy as np
# W = np.random.randn(5, 10)
# X = np.random.randn(10, 3)
# D = W.dot(X)
# print("W = ",W)
# print("X = ",X)
# print("D = ",D.shape)

# # now suppose we had the gradient on D from above in the circuit
# dD = np.random.randn(*D.shape) # same shape as D
# dW = dD.dot(X.T) #.T gives the transpose of the matrix
# dX = W.T.dot(dD)
# # print(dX)

X = np.random.randn(1000,1000)
Y = X[X > 5]
print(Y)