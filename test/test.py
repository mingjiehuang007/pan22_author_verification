import numpy as np

a = [[1,2],[4,5],[7,8]]
a = np.array(a)
a = a.reshape(1,2,3)
print(a)
a = np.mean(a, axis=1, keepdims=True)

print(a)



