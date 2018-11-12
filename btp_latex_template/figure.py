import numpy as np  
import matplotlib.pyplot as plt  

a = np.around(np.random.randn(4,4),1)
plt.figure(1)
fig, ax = plt.subplots()
im = ax.imshow(a)

for i in range(4):
	for j in range(4):
		text = ax.text(j, i, a[i, j])

fig.tight_layout()

plt.figure(2)
fig1, ax1 = plt.subplots()
a[np.abs(a)<0.5] = 0
im = ax1.imshow(a)

for i in range(4):
	for j in range(4):
		text = ax1.text(j, i, a[i, j])
plt.show()