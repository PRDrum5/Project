import numpy as np
import matplotlib.pyplot as plt
import os

# example data
x = np.linspace(-2, 2, 1000)

# Equations to plot
l = 0.2
y1 = x * l
for i, point in enumerate(y1):
    if y1[i] > 0:
        y1[i] = x[i]

y2 = np.ones(x.shape) * -l
for i, point in enumerate(y2):
    if y1[i] > 0:
        y2[i] = 1

fig, ax = plt.subplots()
#ax.set_title('Loss Plots for Original GANs Loss Function')

line1 = ax.plot(x, y1, label='Leaky ReLU Function')
line2 = ax.plot(x, y2, label='Derivative of Leaky ReLU Function', linestyle='--')
#line2, = ax.plot(x, y2, label='Improved Generator Loss')

ax.legend()
#plt.show()

filename = "lrelu.png"
dir_path = os.path.dirname(os.path.realpath(__file__))
save_path = os.path.join(dir_path, "figures", filename)

if not os.path.exists("figures"):
    os.makedirs("figures")

plt.savefig(save_path)