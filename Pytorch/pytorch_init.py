import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def linear_function_2D(x1,x2,beta,omega1,omega2):
  # TODO -- replace the code line below with formula for 2D linear equation
  y = x1*omega1 + x2*omega2 + beta

  return y

def linear_function_3D(x1,x2,x3,beta,omega1,omega2,omega3):
  # TODO -- replace the code below with formula for a single 3D linear equation
  
  y = x1

  return y

def draw_2D_function(x1_mesh, x2_mesh, y):
    fig, ax = plt.subplots()
    fig.set_size_inches(7,7)
    pos = ax.contourf(x1_mesh, x2_mesh, y, levels=256 ,cmap = 'hot', vmin=-10,vmax=10.0)
    fig.colorbar(pos, ax=ax)
    ax.set_xlabel('x1');ax.set_ylabel('x2')
    levels = np.arange(-10,10,1.0)
    ax.contour(x1_mesh, x2_mesh, y, levels, cmap='winter')
    plt.show()
if __name__ == "__main__":
    x1 = np.arange(0.0, 10.0, 0.1)
    x2 = np.arange(0.0, 10.0, 0.1)
    x1,x2 = np.meshgrid(x1,x2)  # https://www.geeksforgeeks.org/numpy-meshgrid-function/

    # Compute the 2D function for given values of omega1, omega2
    beta = 0.0; omega1 = 4.0; omega2 = 1
    y  = linear_function_2D(x1,x2,beta, omega1, omega2)

# Draw the function.
# Color represents y value (brighter = higher value)
# Black = -10 or less, White = +10 or more
# 0 = mid orange
# Lines are contours where value is equal
    draw_2D_function(x1,x2,y)