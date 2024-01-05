import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from matplotlib import cm
import pickle

SEA , key_int , key2 , thickness_val = np.load('Test.npy')

key2 = key2.flatten()

print( key2[ np.argmax(SEA) ] )

plt.scatter( SEA , key_int , s=40 , c=thickness_val , cmap = cm.jet )
m = cm.ScalarMappable(cmap=cm.jet)
m.set_array(thickness_val)
plt.colorbar(m)
plt.xlabel('Specific energy absorption')
plt.ylabel('Design index')
plt.title('Color assigned by wall thickness')

plt.show()