# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Ndsparse matrix implementation in Python

# <headingcell level=2>

# Definition of Ndsparse class

# <codecell>

class Ndsparse:
    """
    N-dimensional sparse matrix.
    entries: dict of positions and values in matrix
        key: N-tuple of positions (i,k,...)
        val: value at position
    d: dimension
    """
    
    def __init__(self, *args, **kwargs):
        """
        Constructor
        """
        if len(args) == 1:
            
            # NDsparse from dict of pos,val pairs
            if args[0].__class__.__name__ == 'dict':
                # Error handling:
                # Make sure all keys in dict are same length
                # Make sure all indexes in keys are ints
                # Make sure all vals in dict are numbers
                self.entries = args[0]
                self.d = len(args[0].iterkeys().next())
        
            # NDsparse from list of lists (of lists...)
            else:
                pass
                
        # Catch unsupported initialization
        else:
            raise Exception("Improper Ndsparse construction.")
    
    def copy(self):
        """
        Copy "constructor"
        """
        return Ndsparse(self.entries)
        
    def __str__(self):
        # Make this nicer: N=N matrix with entries...
        return str(self.entries)
    
    def nnz(self):
        """
        Number of nonzero entries. Number of indexed entries if no explicit 0's allowed.
        """
        return len(self.entries)
        
    def addEntry(self,pos,val):
        # Error handling: make sure entry doesn't overwrite existing ones
        self.entries[pos] = val
        
    def addEntries(self,newEntries):
        # Error handling: make sure entries don't overwrite existing ones
        self.entries.update(newEntries)
        
    def mergePositions(self,other):
        """
        Return (overlap, selfFree, otherFree) 
            overlap: set of tuples of positions where self and other overlap
            selfFree: set of tuples of positions where only self is nonzero
            otherFree: set of tuples of positions where only other is nonzero
        """
        selfKeys = set(self.entries.keys())
        otherKeys = set(other.entries.keys())
        overlap = selfKeys & otherKeys
        selfFree = selfKeys.difference(otherKeys)
        otherFree = otherKeys.difference(selfKeys)
        return (overlap, selfFree, otherFree) 
        
    def __add__(self,other):
        """
        Elementwise addition of self + other.
        """
        # Error handling: make sure Dims are same
        overlap, selfFree, otherFree = self.mergePositions(other)
        out = {}
        
        for pos in overlap:
            out[pos] = self.entries[pos] + other.entries[pos]
        for pos in selfFree:
            out[pos] = self.entries[pos]
        for pos in otherFree:
            out[pos] = other.entries[pos]
        
        return Ndsparse(out)
        
    def __sub__(self,other):
        """
        Elementwise subtraction of self - other.
        """
        # Error handling: make sure Dims are same
        overlap, selfFree, otherFree = self.mergePositions(other)
        out = {}
        
        for pos in overlap:
            out[pos] = self.entries[pos] - other.entries[pos]
        for pos in selfFree:
            out[pos] = self.entries[pos]
        for pos in otherFree:
            out[pos] = -other.entries[pos]
        
        return Ndsparse(out)
    
    def elementwiseMultiply(self,other):
        """
        Elementwise multiplication of self .* other.
        """
        # Error handling: make sure Dims are same
        overlap, selfFree, otherFree = self.mergePositions(other)
        out = {}
        
        for pos in overlap:
            out[pos] = self.entries[pos] * other.entries[pos]
        
        return Ndsparse(out)
    
    def contract(self,other,selfDim,otherDim):
        """
        Tensor contraction/matrix multiplication of self . other
        selfDim, otherDim: dimension of each matrix to contract
        Single index contraction for now
        Result index order: (selfIdxs\selfDim, otherIdxs\otherDim)
            Can transpose after to get what you want
        Time complexity: O(self.nnz * other.nnz)
        """
        # Error handling: make sure Dims are present, contraction is proper
        
        # Accumulate nonzero positions 
        terms = [] # list of tuples of (pos tuple, val) to sum
        
        for pos1,val1 in self.entries.iteritems():
            
            con_idx1 = pos1[selfDim] # contracted index of self
            keep_idx1 = list(pos1)
            keep_idx1.remove(con_idx1) # first part of out pos
            
            for pos2,val2 in other.entries.iteritems():
                
                con_idx2 = pos2[otherDim] # contracted index of other
                keep_idx2 = list(pos2)
                keep_idx2.remove(con_idx2) # second part of out pos
                
                if con_idx1 == con_idx2: # Match entries that share contraction index
                    val = val1 * val2
                    pos = tuple(keep_idx1 + keep_idx2)
                    terms.append((pos,val))
        
        # Sum entries
        out = {}
        for entry in terms:
            pos = entry[0]
            val = entry[1]
            if pos not in out:
                out[pos] = val
            else:
                out[pos] += val
        
        return Ndsparse(out)
        
    def __mul__(self,other):
        """
        Matrix multiplication for N=2. Special case of contract, with self(i,k).other(k,j)
            contracted over k, which has indices 1 and 0, for self and other, respectively
        [Should apply to N>2, this is called the canonical contraction?]
        """
        # Error handling: make sure both matrices are N=2
        return self.contract(other,1,0)
    
    def transpose(self,permutation):
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
        for key,value in self.entries.iteritems():
            out[permute(key,permutation)] = value
        self.entries = out
        
    def kron(self,other):
        """
        Kronecker product of self (x) other
        Only applies to 2D matrices? Completely obviated by general N-d implementation?
        """
        pass
    
    def reshape(self,shapemat):
        """
        Like the MATLAB reshape. http://www.mathworks.com/help/matlab/ref/reshape.html
        """
        pass
    
def permute(vec,permutation):
    """
    Permute vec tuple according to permutation tuple.
    """
    return tuple([vec[permutation[i]] for i in range(len(vec))])

# <headingcell level=2>

# Testing Ndsparse class

# <headingcell level=3>

# 2-D Matrix Testing

# <codecell>

x = {(0,0): 1, (2,1): 3, (1,2): 2}
y = {(1,0): 1, (2,0): 3, (0,1): 1, (0,2): 2}
a = Ndsparse(x)
b = Ndsparse(y)
print a
print b
c = a*b
print c

# <headingcell level=3>

# N-D Matrix Testing

# <rawcell>

# Need a 3rd party worked example for 3 dims

# <codecell>

a1 = {(0,0,0): 3.14, (1,2,3): 4.25, (3,4,5): 2.34}
a2 = {(0,0,0): 4.36, (3,2,0): 3.25, (4,4,1): 1.34}
q = Ndsparse(a1)
r = Ndsparse(a2)
print q
print r
s = q.contract(r,1,0)
print s

# <headingcell level=2>

# Misc Testing

# <codecell>

tem = {}
tel = {(0,0): 4098, (1,2): 4139}

tem.update(tel)
print tem
tel.update({(0,3): 1234})
print tel
print tem.viewkeys() & tel.viewkeys()
print tem.viewkeys()

# <codecell>

single = Ndsparse({(): 3.14})
print single
print single.d

# <codecell>


