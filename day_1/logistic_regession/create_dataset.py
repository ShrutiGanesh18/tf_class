# creating points for 2*x + y = 750

import pickle
import math
import random
import os
import numpy as np

directory = 'data'

if not os.path.exists(directory):
    os.makedirs(directory)

X = [i-100 for i in range(1000)]
Y = [i for i in range(1000)]
random.shuffle(Y)
Z = [int(1+math.copysign(1,(750-(2*x)-(y))))/2 for x,y in zip(X,Y)]

with open(directory+'/train_ip','w') as f:
    pickle.dump(zip(X[:800],Y[:800]),f)

with open(directory+'/train_op','w') as f:
    pickle.dump(Z[:800],f)


with open(directory+'/test_ip','w') as f:
    pickle.dump(zip(X[800:],Y[800:]),f)

with open(directory+'/test_op','w') as f:
    pickle.dump(Z[800:],f)
