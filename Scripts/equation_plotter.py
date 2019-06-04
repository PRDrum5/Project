import numpy as np
import matplotlib.pyplot as plt
import os

# example data
x = np.linspace(0.01, 0.99, 100)

# Equations to plot
y1 = np.log(1-x)
y2 = -np.log(x)

fig, ax = plt.subplots()
#ax.set_title('Loss Plots for Original GANs Loss Function')

line1, = ax.plot(x, y1, label='Original Generator Loss')
line2, = ax.plot(x, y2, label='Improved Generator Loss')

ax.legend()
#plt.show()

filename = "goodfellow_gen_losses.png"
dir_path = os.path.dirname(os.path.realpath(__file__))
save_path = os.path.join(dir_path, "figures", filename)

if not os.path.exists("figures"):
    os.makedirs("figures")

plt.savefig(save_path)