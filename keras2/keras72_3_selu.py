import numpy as np
import matplotlib.pyplot as plt

def selu(x, alpha = 1.6733, scale = 1.0507):
    return (x>0)*scale*x + (x<=0)*(scale*alpha*(np.exp(x)-1))

x = np.arange(-5, 5, 0.1)
y = selu(x)

plt.plot(x,y)
plt.grid()
plt.show()