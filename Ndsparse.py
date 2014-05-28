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
            # NDsparse from a single scalar
            if isinstance(args[0], (int, long, float, complex)):
                self.entries = {():args[0]}
                self.d = 0
            
            # NDsparse from dict of pos,val pairs
            elif args[0].__class__.__name__ == 'dict':
                # Error handling:
                # Make sure all keys in dict are same length
                # Make sure all indexes in keys are ints
                # Make sure all vals in dict are numbers
                self.entries = args[0]
                self.d = len(self.entries.iterkeys().next())
        
            # NDsparse from list of lists (of lists...) dense format
            # 1st dim = rows, 2nd dim = cols, 3rd dim = pages, ...
            else:
                self.entries = buildEntriesDictFromNestedLists(args[0])
                self.d = len(self.entries.iterkeys().next())
                
        # Catch unsupported initialization
        else:
            raise Exception("Improper Ndsparse construction.")
        
        # Cleanup
        self.removeZeros()
    
    def copy(self):
        """
        Copy "constructor"
        """
        return Ndsparse(self.entries)
        
    def __repr__(self):
        """
        String representation of Ndsparse class
        """
        rep = []
        rep.append(''.join([str(self.d),'-d sparse tensor with ', str(self.nnz()), ' nonzero entries\n']))
        poss = self.entries.keys()
        poss.sort()
        for pos in poss:
            rep.append(''.join([str(pos),'\t',str(self.entries[pos]),'\n']))
        return ''.join(rep)

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
    
    def removeZeros(self):
        """
        Remove explicit 0 entries in Ndsparse matrix
        """
        newEntries = {}
        for pos,val in self.entries.iteritems():
            if val != 0:
                newEntries[pos] = val
        self.entries = newEntries
        
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
    
    def __mul__(self,other):
        """
        Elementwise multiplication of self .* other.
        """
        # Error handling: make sure Dims are same
        overlap, selfFree, otherFree = self.mergePositions(other)
        out = {}
        
        for pos in overlap:
            out[pos] = self.entries[pos] * other.entries[pos]
        
        return Ndsparse(out)
    
    def __div__(self,other):
        """
        Elementwise division of self ./ other, casting ints to floats.
        """
        # Error handling: make sure Dims are same
        overlap, selfFree, otherFree = self.mergePositions(other)
        out = {}
        
        for pos in overlap:
            out[pos] = float(self.entries[pos]) / other.entries[pos]
        
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
    
    def outerProduct(self,other):
        """
        Outer product
        Result index order: (selfIdxs, otherIdxs)
        Note: scalar multiplication is an outer product
        """
        # Accumulate nonzero positions 
        terms = [] # list of tuples of (pos tuple, val) to sum
        
        for pos1,val1 in self.entries.iteritems():
            for pos2,val2 in other.entries.iteritems():
                    val = val1 * val2
                    pos = tuple(pos1+pos2)
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
    
    def ttt(self,other,*args):
        """
        Tensor x tensor generalized multiplication. Should include inner and outer products and contraction.
        Outer product: Contract on no dims
        Inner product: Contract on all dims (specify order)
        Product with contraction on arbitrary dimensions also
        """
        pass
    
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
        
    def kroneckerProduct(self,other):
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

def traverseWithIndices(superList, treeTypes=(list, tuple)):
    """
    Traverse over tree structure (lists of lists of ...), with indices. Returns a nested lists of lists, each containing
    the next position on the end. Call flatten to get into a nice form.
    """
    idxs = []
    if isinstance(superList, treeTypes):
        for idx,value in enumerate(superList):
            idxs = idxs[0:-1]
            idxs.append(idx)
            for subValue in traverseWithIndices(value):
                yield [subValue,idxs]
    else:
        yield [superList,idxs]
        
def flatten(superList, treeTypes=(list, tuple)):
    '''
    Flatten arbitrarily nested lists into a single list.
    '''
    flatList = []
    for subList in superList:
        if isinstance(subList, treeTypes):
            flatList.extend(flatten(subList))
        else:
            flatList.append(subList)
    return flatList

def buildEntriesDictFromNestedLists(nestedLists):
    """
    Build dict of pos:val pairs for Ndsparse.entries format from a flat list where list[0] is
    the val and list[1:-1] are the pos indices in reverse order.
    """
    # Special case for scalar in a list
    #    Warning: first dim shouldn't be singleton
    if len(nestedLists) == 1:
        return {():nestedLists[0]}
    
    entriesDict = {}
    for entry in traverseWithIndices(nestedLists):
        flatEntry = flatten(entry)
        pos = tuple(flatEntry[-1:0:-1])
        val = flatEntry[0]
        entriesDict[pos] = val
    return entriesDict

# Testing code
if __name__ == "__main__":
    Al = [[[1,7,3], [2,8,4]], [[3,9,5], [4,0,6]], [[5,1,7], [6,2,8]], [[0,1,9], [1,0,3]]]
    Bl = [[[5,1],[7,0],[8,4],[0,4]], [[0,3],[1,5],[9,6],[1,2]], [[4,9],[3,8],[6,7],[2,0]]]
    print Al
    print Bl
    A = Ndsparse(Al)
    B = Ndsparse(Bl)
    print A
    print B
    #C = A.outerProduct(B)
    #print C
    D = Ndsparse(6)
    print D
    E = A.outerProduct(D)
    print E