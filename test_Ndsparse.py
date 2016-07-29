import unittest
import numpy as np
from random import random

from Ndsparse import *


class TestNdsparse(unittest.TestCase):

    def setUp(self):
        # Some default testing matrices
        Xl = [[[1, 7, 3], [2, 8, 4]], [[3, 9, 5], [4, 0, 6]], [[5, 1, 7], [6, 2, 8]], [[0, 1, 9], [1, 0, 3]]]
        self.X = Ndsparse(Xl)

        Yl = [[[5, 1], [7, 0], [8, 4], [0, 4]], [[0, 3], [1, 5], [9, 6], [1, 2]], [[4, 9], [3, 8], [6, 7], [2, 0]]]
        self.Y = Ndsparse(Yl)

        self.Anp = np.array([[[1, 7, 3], [2, 8, 4]], [[3, 9, 5], [4, 0, 6]], [[5, 1, 7], [6, 2, 8]], [[0, 1, 9], [1, 0, 3]]])
        self.Bnp = np.array([[[5, 1], [7, 0], [8, 4], [0, 4]], [[0, 3], [1, 5], [9, 6], [1, 2]], [[4, 9], [3, 8], [6, 7], [2, 0]]])

    # Test construction
    def test_construct_blank(self):
        X = Ndsparse()
        self.assertEqual(X.d, 0)
        self.assertEqual(X.shape, ())
        self.assertEqual(len(X.entries), 0)

    def test_construct_scalar(self):
        # Different inputs to construct a scalar Ndsparse
        Xl = {(): random()}
        X = Ndsparse(Xl)
        self.assertEqual(X.d, 0)
        self.assertEqual(X.shape, ())
        self.assertEqual(len(X.entries), 1)

        Yl = random()
        Y = Ndsparse(Yl)
        self.assertEqual(Y.d, 0)
        self.assertEqual(Y.shape, ())
        self.assertEqual(len(Y.entries), 1)

    def test_construct_from_dict(self):
        Xd = {(0, 0): 1, (2, 1): 3, (1, 2): 2}
        X = Ndsparse(Xd)
        self.assertEqual(X.d, 2)
        self.assertEqual(X.shape, (3, 3))
        self.assertEqual(len(X.entries), 3)

        Yd = {(1, 0): 1, (2, 0): 3, (0, 1): 1, (0, 2): 2}
        Y = Ndsparse(Yd)
        self.assertEqual(Y.d, 2)
        self.assertEqual(Y.shape, (3, 3))
        self.assertEqual(len(Y.entries), 4)

    def test_construct_from_ndarray(self):
        Anp = self.Anp
        A = Ndsparse(Anp)
        self.assertEqual(A.d, len(Anp.shape))
        self.assertEqual(A.shape, Anp.shape)
        self.assertEqual(len(A.entries), np.count_nonzero(Anp))

        Bnp = self.Bnp
        B = Ndsparse(Bnp)
        self.assertEqual(B.d, len(Bnp.shape))
        self.assertEqual(B.shape, Bnp.shape)
        self.assertEqual(len(B.entries), np.count_nonzero(Bnp))

    def test_construct_from_lists(self):
        X = self.X
        self.assertEqual(X.d, 3)
        self.assertEqual(X.shape, (4, 2, 3))
        self.assertEqual(len(X.entries), 4 * 3 * 2 - 3)  # 3 zeros in Xl

        Y = self.Y
        self.assertEqual(Y.d, 3)
        self.assertEqual(Y.shape, (3, 4, 2))
        self.assertEqual(len(Y.entries), 3 * 4 * 2 - 4)  # 4 zeros in Yl

        # Corner case of single element in a vector (1-d array)
        Zl = [random()]
        Z = Ndsparse(Zl)
        self.assertEqual(Z.d, 1)
        self.assertEqual(Z.shape, (1,))
        self.assertEqual(len(Z.entries), 1)

    def test_construct_from_Ndsparse(self):
        X = self.X
        X2 = Ndsparse(X)
        self.assertEqual(X, X2)  # like '==', which is implemented by overloading __eq__
        self.assertEqual(X.d, X2.d)  # more checks, including testing __eq__ implementation
        self.assertEqual(X.shape, X2.shape)
        self.assertEqual(X.entries, X2.entries)

    # Test utility functions
    def test_set_and_get(self):
        X = self.X
        newpos = (1, 1, 1)
        newval = 1.23
        X[newpos] = newval
        val1 = X[newpos]  # indexing can be with a tuple
        val2 = X[1, 1, 1]  # or several integer args
        self.assertEqual(val1, newval)
        self.assertEqual(val2, newval)

    def test_remove_entry(self):
        X = self.X
        pos = (0, 1, 2)
        self.assertEqual(X[pos], 4)
        X[pos] = 0
        self.assertEqual(X[pos], 0)

    def test_nnz(self):
        X = self.X
        self.assertEqual(X.nnz(), 21)

    def test_to_np(self):
        Xnp = np.random.rand(4, 2, 3)
        X = Ndsparse(Xnp)
        Xnp2 = X.to_np()
        self.assertTrue(np.array_equal(Xnp, Xnp2))

    # Test tensor operations
    def test_add(self):
        Xnp = np.random.rand(4, 2, 3)
        Ynp = np.random.rand(4, 2, 3)
        X = Ndsparse(Xnp)
        Y = Ndsparse(Ynp)

        Znp = Xnp + Ynp
        Z = X + Y
        Znp2 = Z.to_np()
        self.assertTrue(np.allclose(Znp, Znp2))

    def test_subtract(self):
        Xnp = np.random.rand(4, 2, 3)
        Ynp = np.random.rand(4, 2, 3)
        X = Ndsparse(Xnp)
        Y = Ndsparse(Ynp)

        Znp = Xnp - Ynp
        Z = X - Y
        Znp2 = Z.to_np()
        self.assertTrue(np.allclose(Znp, Znp2))

    def test_multiply(self):
        # Element-wise multiply
        Xnp = np.random.rand(4, 2, 3)
        Ynp = np.random.rand(4, 2, 3)
        X = Ndsparse(Xnp)
        Y = Ndsparse(Ynp)

        Znp = Xnp * Ynp
        Z = X * Y
        Znp2 = Z.to_np()
        self.assertTrue(np.allclose(Znp, Znp2))

    def test_matrix_multiply(self):
        Xnp = np.random.rand(4, 3)
        Ynp = np.random.rand(3, 2)
        X = Ndsparse(Xnp)
        Y = Ndsparse(Ynp)

        Znp = np.dot(Xnp, Ynp)
        Z = matrix_product(X, Y)
        self.assertEqual(Z.shape, (4, 2))
        Znp2 = Z.to_np()
        self.assertTrue(np.allclose(Znp, Znp2))

    def test_outer_product(self):
        Xnp = np.random.rand(4)
        Ynp = np.random.rand(3)
        X = Ndsparse(Xnp)
        Y = Ndsparse(Ynp)

        Znp = np.outer(Xnp, Ynp)
        Z = outer_product(X, Y)
        self.assertEqual(Z.shape, (4, 3))
        Znp2 = Z.to_np()
        self.assertTrue(np.allclose(Znp, Znp2))

    def test_kronecker_product(self):
        Xnp = np.random.rand(4, 3)
        Ynp = np.random.rand(3, 2)
        X = Ndsparse(Xnp)
        Y = Ndsparse(Ynp)

        Znp = np.kron(Xnp, Ynp)
        Z = kronecker_product(X, Y)
        self.assertEqual(Z.shape, (12, 6))
        Znp2 = Z.to_np()
        self.assertTrue(np.allclose(Znp, Znp2))

    def test_transpose(self):
        Xnp = np.random.rand(5, 3)
        X = Ndsparse(Xnp)  # in-place

        Xnpt = Xnp.T
        X.transpose()
        self.assertEqual(X.shape, Xnpt.shape)


if __name__ == '__main__':
    unittest.main()