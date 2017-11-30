# creating points for 2x + y = 30

import pickle
import os

directory = 'data'

if not os.path.exists(directory):
    os.makedirs(directory)

X = [float(i) for i in range(1000)]
Y = [30.0-(2*i) for i in X]

with open(directory+'/train_ip','w') as f:
    pickle.dump(X[:900],f)

with open(directory+'/train_op','w') as f:
    pickle.dump(Y[:900],f)

with open(directory+'/test_ip','w') as f:
    pickle.dump(X[900:],f)

with open(directory+'/test_op','w') as f:
    pickle.dump(Y[900:],f)
