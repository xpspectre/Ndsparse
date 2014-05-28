import Ndsparse
from Ndsparse import *

x = {(0,0): 1, (2,1): 3, (1,2): 2}
y = {(1,0): 1, (2,0): 3, (0,1): 1, (0,2): 2}
a = Ndsparse(x)
b = Ndsparse(y)
print a
print b


a1 = {(0,0,0): 3.14, (1,2,3): 4.25, (3,4,5): 2.34}
a2 = {(0,0,0): 4.36, (3,2,0): 3.25, (4,4,1): 1.34}
q = Ndsparse(a1)
r = Ndsparse(a2)
print q
print r
s = q.contract(r,1,0)
print s
