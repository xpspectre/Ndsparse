import numpy as np
from numbers import Number


class Ndsparse:
    """
    N-dimensional sparse matrix.
    entries: dict of positions and values in matrix
        key: N-tuple of positions (i,k,...)
        val: value at position
    d: dimension
    """

    def __init__(self, *args):
        """
        Constructor
        Ndsparse(scalar)
        Ndsparse(dict of (pos):val pairs, optional list of dims for shape)
        Ndsparse(other Ndsparse)
        Ndsparse(numpy.array)
        Ndsparse(nested lists)
        """
        # Blank Ndsparse
        if len(args) == 0:
            self.entries = {}
            self.d = 0
            self.shape = ()

        # From a single scalar
        elif isinstance(args[0], (int, float, complex)):
            self.entries = {(): args[0]}
            self.d = 0
            self.shape = ()

        # From dict of pos,val pairs
        #   Make sure a new dict is produced for the Ndsparse
        elif args[0].__class__.__name__ == 'dict' or args[0].__class__.__name__ == 'Ndsparse':
            if args[0].__class__.__name__ == 'Ndsparse':
                entries = args[0].entries
            else:
                entries = args[0]

            self.entries = {}
            ctr = 0
            for pos, val in entries.items():
                if ctr == 0:
                    d = len(pos)
                if not all(isinstance(x, int) for x in pos):
                    raise IndexError('Position indices must be integers')
                if len(pos) != d:
                    raise IndexError('Position index dimension mismatch')
                if not isinstance(val, Number):
                    raise ValueError('Values must be numbers')
                self.entries[pos] = val
                ctr += 1

            self.d = d

            if len(args) > 1:
                self.shape = args[1]
            else:
                self.shape = get_entries_shape(entries)

        # From numpy array or list of lists (of lists...) dense format
        #   1st dim = rows, 2nd dim = cols, 3rd dim = pages, ...
        #   Uses a numpy array as an intermediate when constructing from lists for convenience
        #   Note that for now, values are converted to floats
        elif args[0].__class__.__name__ == 'ndarray' or args[0].__class__.__name__ == 'list':
            if args[0].__class__.__name__ == 'list':
                array = np.array(args[0])
            else:
                array = args[0]

            self.entries = {}
            it = np.nditer(array, flags=['multi_index'])
            while not it.finished:
                self.entries[it.multi_index] = float(it[0])
                it.iternext()
            self.shape = array.shape
            self.d = len(self.shape)

        # Catch unsupported initialization
        else:
            raise TypeError("Unknown type for Ndsparse construction.")

        # Cleanup
        self.remove_zeros()

    def copy(self):
        """
        Copy "constructor"
        """
        return Ndsparse(self.entries)

    def __repr__(self):
        """
        String representation of Ndsparse class
        """
        rep = [''.join([str(self.d), '-d sparse tensor with ', str(self.nnz()), ' nonzero entries\n'])]
        poss = list(self.entries.keys())
        poss.sort()
        for pos in poss:
            rep.append(''.join([str(pos), '\t', str(self.entries[pos]), '\n']))
        return ''.join(rep)

    def nnz(self):
        """
        Number of nonzero entries. Number of indexed entries if no explicit 0's allowed.
        """
        return len(self.entries)

    def merge_positions(self, other):
        """
        Return (overlap, self_free, other_free)
            overlap: set of tuples of positions where self and other overlap
            self_free: set of tuples of positions where only self is nonzero
            other_free: set of tuples of positions where only other is nonzero
        """
        self_keys = set(self.entries.keys())
        other_keys = set(other.entries.keys())
        overlap = self_keys & other_keys
        self_free = self_keys.difference(other_keys)
        other_free = other_keys.difference(self_keys)
        return overlap, self_free, other_free

    def remove_zeros(self):
        """
        Remove explicit 0 entries in Ndsparse matrix
        """
        new_entries = {}
        for pos, val in self.entries.items():
            if val != 0:
                new_entries[pos] = val
        self.entries = new_entries

    def __getitem__(self, index):
        """Get value at tuple item"""
        if len(index) != self.d:
            raise IndexError('Wrong number of indices specified')
        for i, ind in enumerate(index):
            if ind > self.shape[i] or ind < 0:
                raise IndexError('%i-th index is out of bounds' % (i,))

        try:
            return self.entries[index]
        except KeyError:
            return 0

    def __setitem__(self, index, value):
        """Set value at tuple """
        if len(index) != self.d:
            raise IndexError('Wrong number of indices specified')
        for i, ind in enumerate(index):
            if ind > self.shape[i] or ind < 0:
                raise IndexError('%i-th index is out of bounds' % (i,))

        if value == 0:  # Special case adds structural 0
            del self.entries[index]
        else:
            self.entries[index] = value

    def __eq__(self, other):
        """
        Test equality of 2 Ndsparse objects by value. Must have the same nonzero elements, rank, and dimensions.
        """
        if self.d == other.d and self.shape == other.shape and self.entries == other.entries:
            return True
        else:
            return False

    def __add__(self, other):
        """
        Element-wise addition of self + other.
        """
        # Error handling: make sure Dims are same
        overlap, self_free, other_free = self.merge_positions(other)
        out = {}

        for pos in overlap:
            out[pos] = self.entries[pos] + other.entries[pos]
        for pos in self_free:
            out[pos] = self.entries[pos]
        for pos in other_free:
            out[pos] = other.entries[pos]

        return Ndsparse(out, self.shape)

    def __sub__(self, other):
        """
        Element-wise subtraction of self - other.
        """
        # Error handling: make sure Dims are same
        overlap, self_free, other_free = self.merge_positions(other)
        out = {}

        for pos in overlap:
            out[pos] = self.entries[pos] - other.entries[pos]
        for pos in self_free:
            out[pos] = self.entries[pos]
        for pos in other_free:
            out[pos] = -other.entries[pos]

        return Ndsparse(out, self.shape)

    def __mul__(self, other):
        """
        Element-wise multiplication of self .* other.
        """
        # Error handling: make sure Dims are same
        overlap, self_free, other_free = self.merge_positions(other)
        out = {}

        for pos in overlap:
            out[pos] = self.entries[pos] * other.entries[pos]

        return Ndsparse(out, self.shape)

    def matrix_product(self, other):
        """
        Standard 2-d matrix multiply using ttt
        """
        assert self.d == 2 and other.d == 2, 'Matrices must both be 2-d for usual multiplication.'
        return self.ttt(other, [1], [0])

    def outer_product(self, other):
        """
        Outer product using ttt
        """
        return self.ttt(other)

    def kronecker_product(self, other):
        """
        Kronecker product of 2-d matrices using ttt and rearranging indices
        """
        assert self.d == 2 and other.d == 2, 'Matrices must both be 2-d forkronecker product.'
        prod = self.ttt(other)

        # Dimensions: self(m x n) (x) other(p x q) = kprod(mp x nq)
        m = self.shape[0]
        n = self.shape[1]
        p = other.shape[0]
        q = other.shape[1]

        # Modify indices
        kprod = {}
        for pos, val in prod.entries.items():
            i = p * pos[0] + pos[2] + 1
            j = q * pos[1] + pos[3] + 1
            kprod[(i - 1, j - 1)] = val

        return Ndsparse(kprod, [m * p, n * q])

    def ttt(self, other, *args):
        """
        Tensor x tensor generalized multiplication. Should include inner and outer products and contraction.
        Specify contraction dims with 1 list of dims in order for self followed by 1 list of dims in order for other
        
        Outer product: Contract on no dims
        Inner product: Contract on all dims (specify order)
        Product with contraction on arbitrary dimensions also
        
        Result index order: (self_inds\self_dim, other_inds\other_dim)
        Time complexity: O(self.nnz * other.nnz)
        """

        if len(args) == 0:  # no contractions/outer product
            self_dims = []
            other_dims = []
        elif len(args) == 2:  # contractions specified
            self_dims = args[0]
            other_dims = args[1]
            # Error handling: make sure Dims are present, contraction is proper, same lengths
        else:
            raise Exception("ttt requires 1 or 3 args.")

        # Accumulate nonzero positions 
        terms = []  # list of tuples of (pos tuple, val) to sum

        for pos1, val1 in self.entries.items():

            con_ind1s = [item for i, item in enumerate(pos1) if i in self_dims]  # self's contracted indices
            keep_ind1s = [item for i, item in enumerate(pos1) if i not in self_dims]  # self's kept indices

            for pos2, val2 in other.entries.items():

                con_ind2s = [item for i, item in enumerate(pos2) if i in other_dims]  # other's contracted indices
                keep_ind2s = [item for i, item in enumerate(pos2) if i not in other_dims]  # other's kept indices

                pos = tuple(keep_ind1s + keep_ind2s)

                if con_ind1s == con_ind2s:  # Match entries that share contraction index (including none)
                    terms.append((pos, val1 * val2))

        # Sum entries
        out = {}
        for entry in terms:
            pos = entry[0]
            val = entry[1]
            if pos not in out:
                out[pos] = val
            else:
                out[pos] += val

        shape = [item for i, item in enumerate(self.shape) if i not in self_dims] + \
                [item for i, item in enumerate(other.shape) if i not in other_dims]
        return Ndsparse(out, shape)

    def transpose(self, permutation):
        """
        Transpose Ndsparse matrix in place
        permutation: tuple of new indices
        Matrix starts out as (0,1,...,N) and can be transposed according to 
           the permutation (N,1,....0) or whatever, with N! possible permutations
        Note indexing starts at 0
        """
        # Error handling: make sure permutation is valid (eg, has right length)
        # Useful extension: default transpose for N=2 matrices
        out = {}
        for key, value in self.entries.items():
            out[permute(key, permutation)] = value
        self.entries = out
        self.shape = list(permute(self.shape, permutation))

    def reshape(self, shapemat):
        """
        Like the MATLAB reshape. http://www.mathworks.com/help/matlab/ref/reshape.html
        """
        raise NotImplementedError


def permute(vec, permutation):
    """
    Permute vec tuple according to permutation tuple.
    """
    return tuple([vec[permutation[i]] for i in range(len(vec))])


def get_entries_shape(entries):
    """
    Get dimensions corresponding to max indices in entries
    """
    max_inds = [0] * len(next(iter(entries.keys())))
    for pos in entries.keys():
        for i, ind in enumerate(pos):
            if ind > max_inds[i]:
                max_inds[i] = ind
    return tuple([ind + 1 for ind in max_inds])


# Testing code
if __name__ == "__main__":
    Al = [[[1, 7, 3], [2, 8, 4]], [[3, 9, 5], [4, 0, 6]], [[5, 1, 7], [6, 2, 8]], [[0, 1, 9], [1, 0, 3]]]
    Bl = [[[5, 1], [7, 0], [8, 4], [0, 4]], [[0, 3], [1, 5], [9, 6], [1, 2]], [[4, 9], [3, 8], [6, 7], [2, 0]]]
    print(Al)
    print(Bl)
    A = Ndsparse(Al)
    B = Ndsparse(Bl)
    print(A)
    print(B)
    print(A.shape)
    print(B.shape)
    # C = A.outerProduct(B)
    # print C
    # D = Ndsparse(6)
    # print D
    # E = A.outerProduct(D)
    # print E
    C = A.ttt(B, [0, 1], [1, 2])
    # C = A.ttt(B)
    # print C
    print(C.shape)
    Gl = [[1, 2], [3, 4]]
    Hl = [[0, 5], [6, 7]]
    G = Ndsparse(Gl)
    H = Ndsparse(Hl)
    print(G.matrix_product(G))
    print(G.kronecker_product(H))
