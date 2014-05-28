import Ndsparse
from Ndsparse import *

Al = [[[1,7,3], [2,8,4]], [[3,9,5], [4,0,6]], [[5,1,7], [6,2,8]], [[0,1,9], [1,0,3]]]
Bl = [[[5,1],[7,0],[8,4],[0,4]], [[0,3],[1,5],[9,6],[1,2]], [[4,9],[3,8],[6,7],[2,0]]]
print Al
print Bl
A = Ndsparse(Al)
B = Ndsparse(Bl)
print A
print B

# Test scalar
Xl = {():4}
X = Ndsparse(Xl)
print X

Yl = 5
Y = Ndsparse(Yl)
print Y

Zl = [6]
Z = Ndsparse(Zl)
print Z
