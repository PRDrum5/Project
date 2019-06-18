import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

x = np.array([[1,1],[1,2],[3,2]], dtype=np.float)
y = -2*x - [1,0]
print(x)
print(y)
c_x = np.mean(x, 0)
c_y = np.mean(y, 0)
plt.scatter(x[:,0], x[:,1], marker='o', label='X')
plt.scatter(y[:,0], y[:,1], marker='x', label='Y')
plt.scatter((c_x[0], c_y[0]),(c_x[1], c_y[1]), label='Centroids')
plt.legend()
plt.grid()
plt.show()

# Centre
x_mean = np.mean(x, 0)
y_mean = np.mean(y, 0)
x -= x_mean
y -= y_mean
c_x = np.mean(x, 0)
c_y = np.mean(y, 0)
plt.scatter(x[:,0], x[:,1], marker='o', label='X')
plt.scatter(y[:,0], y[:,1], marker='x', label='Y')
plt.scatter((c_x[0], c_y[0]),(c_x[1], c_y[1]), label='Centroids')
plt.legend()
plt.grid()
plt.show()

# Scale
norm_x = la.norm(x)
norm_y = la.norm(y)
x /= norm_x
y /= norm_y
c_x = np.mean(x, 0)
c_y = np.mean(y, 0)
plt.scatter(x[:,0], x[:,1], marker='o', label='X')
plt.scatter(y[:,0], y[:,1], marker='x', label='Y')
plt.scatter((c_x[0], c_y[0]),(c_x[1], c_y[1]), label='Centroids')
plt.legend()
plt.grid()
plt.show()

# Rotate
rotation, _ = la.orthogonal_procrustes(x, y)
y = (y @ rotation.T)
c_x = np.mean(x, 0)
c_y = np.mean(y, 0)
plt.scatter(x[:,0], x[:,1], marker='o', label='X')
plt.scatter(y[:,0], y[:,1], marker='x', label='Y')
plt.scatter((c_x[0], c_y[0]),(c_x[1], c_y[1]), label='Centroids')
plt.legend()
plt.grid()
plt.show()