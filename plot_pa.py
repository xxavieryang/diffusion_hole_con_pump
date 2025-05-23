import numpy as np
import scipy.integrate as integrate
import scipy.optimize as opt
import matplotlib.pyplot as plt

# Define parameters
a = 50  # Adjust as needed
beta1 = 0  # Adjust as needed
beta2 = 1 # Adjust as needed
beta3 = 2 # Adjust as needed
beta4 = 4 # Adjust as needed

# Define inverse Laplace transform integral

def p_a_integrand(s, t, a, beta):
    return np.exp(s * t - beta * np.sqrt(s)) / (2 * np.sqrt(s) + 2 * a * np.exp(-np.sqrt(s)))

def p_a(t, a, beta):
    sigma = 0.1  # Vertical contour position
    integral, _ = integrate.quad(lambda w: np.real(p_a_integrand(sigma + 1j * w, t, a, beta)), -50, 50, limit=200)
    return (1 / (2 * np.pi)) * integral

# Compute p_a(t, beta) for a range of t values
t_values = np.linspace(0.28, 10, 100)
p_a_values1 = [p_a(t, a, beta1) for t in t_values]
p_a_values2 = [p_a(t, a, beta2) for t in t_values]
p_a_values3 = [p_a(t, a, beta3) for t in t_values]
p_a_values4 = [p_a(t, a, beta4) for t in t_values]

# Plot the result
plt.plot(t_values, p_a_values1, label='$\\beta=$'+str(beta1))
plt.plot(t_values, p_a_values2, label='$\\beta=$'+str(beta2))
plt.plot(t_values, p_a_values3, label='$\\beta=$'+str(beta3))
plt.plot(t_values, p_a_values4, label='$\\beta=$'+str(beta4))
plt.xlabel('$t$')
plt.ylabel('$p_a(t, \\beta)$')
plt.legend()
plt.grid()
plt.savefig("a=" + str(a) + ".png")
plt.show()