from netfuncs.activations.simple import *
import numpy as np
import matplotlib.pyplot as plt

z = np.arange(-10, 10, 0.1)

""" Test Sigmoid """
fn = Sigmoid()
fz = fn.f(z)
dfz = fn.df(z)
# Plot
plt.figure(1)
plt.subplot(211)
plt.plot(z, fz)
plt.title("Sigmoid")
plt.ylabel("f(z)")
plt.subplot(212)
plt.plot(z, dfz)
plt.ylabel("df/dz")


""" Test Tanh """
fn = Tanh()
fz = fn.f(z)
dfz = fn.df(z)
# Plot
plt.figure(2)
plt.subplot(211)
plt.plot(z, fz)
plt.title("Tanh")
plt.ylabel("f(z)")
plt.subplot(212)
plt.plot(z, dfz)
plt.ylabel("df/dz")

""" Test ReLU """
fn = ReLU()
fz = fn.f(z)
dfz = fn.df(z)
# Plot
plt.figure(3)
plt.subplot(211)
plt.plot(z, fz)
plt.title("ReLU")
plt.ylabel("f(z)")
plt.subplot(212)
plt.plot(z, dfz)
plt.ylabel("df/dz")

""" Test Leaky ReLU """
fn = LeakyReLU(0.2)
fz = fn.f(z)
dfz = fn.df(z)
# Plot
plt.figure(4)
plt.subplot(211)
plt.plot(z, fz)
plt.title("Leaky ReLU")
plt.ylabel("f(z)")
plt.subplot(212)
plt.plot(z, dfz)
plt.ylabel("df/dz")

plt.show()
