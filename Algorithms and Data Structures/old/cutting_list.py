import random
import numpy as np
a = np.arange(30).reshape((5,2,3))

print(a)
print(a.shape)

c = a[...,2]
print('---------------------------------')
print(c)
print(c.shape)