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
        NDsparse(scalar)
        NDsparse(dict of (pos):val pairs, optional list of dims for shape)
        NDsparse(nested list dense representation)
        """
        # NDsparse from a single scalar
        if isinstance(args[0], (int, long, float, complex)):
            self.entries = {():args[0]}
            self.d = 0
            self.shape = ()
        
        # NDsparse from dict of pos,val pairs
        elif args[0].__class__.__name__ == 'dict':
            # Error handling:
            # Make sure all keys in dict are same length
            # Make sure all indexes in keys are ints
            # Make sure all vals in dict are numbers
            self.entries = args[0]
            self.d = len(self.entries.iterkeys().next())
            if len(args) > 1:
                self.shape = args[1]
            else:
                self.shape = getEntriesShape(args[0])
    
        # NDsparse from list of lists (of lists...) dense format
        # 1st dim = rows, 2nd dim = cols, 3rd dim = pages, ...
        elif args[0].__class__.__name__ == 'list':
            self.entries = buildEntriesDictFromNestedLists(args[0])
            self.d = len(self.entries.iterkeys().next())
            self.shape = getListsShape(args[0])
                
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
        
    def __eq__(self,other):
        """
        Test equality of 2 Ndsparse objects. Must have the same nonzero elements, rank, and dimensions.
        """
        if self.d == other.d and self.shape == other.shape and self.entries == other.entries:
            return True
        else:
            return False
        
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
        
        return Ndsparse(out,self.shape)
        
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
        
        return Ndsparse(out,self.shape)
    
    def __mul__(self,other):
        """
        Elementwise multiplication of self .* other.
        """
        # Error handling: make sure Dims are same
        overlap, selfFree, otherFree = self.mergePositions(other)
        out = {}
        
        for pos in overlap:
            out[pos] = self.entries[pos] * other.entries[pos]
        
        return Ndsparse(out,self.shape)
    
    def __div__(self,other):
        """
        Elementwise division of nonzero entries of self ./ other, casting ints to floats.
        """
        # Error handling: make sure Dims are same
        overlap, selfFree, otherFree = self.mergePositions(other)
        out = {}
        
        for pos in overlap:
            out[pos] = float(self.entries[pos]) / other.entries[pos]
        
        return Ndsparse(out,self.shape)
    
    def matrixProduct(self,other):
        """
        Standard 2-d matrix multiply using ttt
        """
        assert self.d == 2 and other.d == 2, 'Matrices are wrong dimension for usual multiplication.'
        return self.ttt(other,[1],[0])
    
    def outerProduct(self,other):
        """
        Outer product using ttt
        """
        return self.ttt(other)
    
    def kroneckerProduct(self,other):
        """
        Kronecker product of 2-d matrices using ttt and rearranging indices
        """
        assert self.d == 2 and other.d == 2, 'Matrices are wrong dimension for kronecker product.'
        prod = self.ttt(other)
        
        # Dimensions: self(m x n) (x) other(p x q) = kprod(mp x nq)
        m = self.shape[0]
        n = self.shape[1]
        p = other.shape[0]
        q = other.shape[1]
        
        # Modify indices
        kprod = {}
        for pos,val in prod.entries.iteritems():
            i = p*pos[0] + pos[2] + 1
            j = q*pos[1] + pos[3] + 1
            kprod[(i-1,j-1)] = val
            
        return Ndsparse(kprod,[m*p,n*q])
    
    def ttt(self,other,*args):
        """
        Tensor x tensor generalized multiplication. Should include inner and outer products and contraction.
        Specify contraction dims with 1 list of dims in order for self followed by 1 list of dims in order for other
        
        Outer product: Contract on no dims
        Inner product: Contract on all dims (specify order)
        Product with contraction on arbitrary dimensions also
        
        Result index order: (selfIdxs\selfDim, otherIdxs\otherDim)
        Time complexity: O(self.nnz * other.nnz)
        """
        
        if len(args) == 0: # no contractions/outer product
            selfDims = []
            otherDims = []
        elif len(args) == 2: # contractions specified
            selfDims = args[0]
            otherDims = args[1]
            # Error handling: make sure Dims are present, contraction is proper, same lengths
        else:
            raise Exception("ttt requires 1 or 3 args.")
            
        # Accumulate nonzero positions 
        terms = [] # list of tuples of (pos tuple, val) to sum
        
        for pos1,val1 in self.entries.iteritems():
            
            conIdx1s = [ item for i,item in enumerate(pos1) if i in selfDims ] # self's contracted indices
            keepIdx1s = [ item for i,item in enumerate(pos1) if i not in selfDims ] # self's kept indices
            
            for pos2,val2 in other.entries.iteritems():
                
                conIdx2s = [ item for i,item in enumerate(pos2) if i in otherDims ] # other's contracted indices
                keepIdx2s = [ item for i,item in enumerate(pos2) if i not in otherDims ] # other's kept indices
                
                pos = tuple(keepIdx1s + keepIdx2s)
                
                if conIdx1s == conIdx2s: # Match entries that share contraction index (including none)
                    terms.append((pos, val1*val2))
        
        # Sum entries
        out = {}
        for entry in terms:
            pos = entry[0]
            val = entry[1]
            if pos not in out:
                out[pos] = val
            else:
                out[pos] += val
        
        shape = [ item for i,item in enumerate(self.shape) if i not in selfDims ] + \
                [ item for i,item in enumerate(other.shape) if i not in otherDims ]
        return Ndsparse(out,shape)
    
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
        self.shape = list(permute(self.shape,permutation))
    
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
    Traverse over tree structure (nested lists), with indices. Returns a nested list 
    with the left element a nested list and the right element the next position.
    Call flatten to get into a nice form.
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
    Flatten arbitrarily nested lists into a single list, removing empty lists.
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

def getListsShape(nestedLists, treeTypes=(list, tuple)):
    """
    Get dimensions of nested lists
    """
    shape = []
    lst = list(nestedLists)
    while isinstance(lst, treeTypes):
        shape.append(len(lst))
        lst = lst[0]
    return shape

def getEntriesShape(entries):
    """
    Get dimensions corresponding to max indices in entries
    """
    maxIdxs = [0]*len(entries.iterkeys().next())
    for pos in entries.iterkeys():
        for i,idx in enumerate(pos):
            if idx > maxIdxs[i]:
                maxIdxs[i] = idx
    return [idx+1 for idx in maxIdxs]

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
    print A.shape
    print B.shape
    #C = A.outerProduct(B)
    #print C
    #D = Ndsparse(6)
    #print D
    #E = A.outerProduct(D)
    #print E
    C = A.ttt(B,[0,1],[1,2])
    #C = A.ttt(B)
    #print C
    print C.shape
    Gl = [[1,2],[3,4]]
    Hl = [[0,5],[6,7]]
    G = Ndsparse(Gl)
    H = Ndsparse(Hl)
    print G.matrixProduct(G)
    print G.kroneckerProduct(H)
