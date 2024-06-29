import numpy as np

def f(x):
    return x**2 + 2*x + 1

def robbins_monro(root_function, initial_guess, num_iterations):
    x = initial_guess
    results = [x]
    for n in range(1, num_iterations+1):
        step_size = 1 / (n + 1)
        x = x - step_size * root_function(x)
        results.append(x)
    return results

# Parameters
initial_guess = 0
num_iterations = 1000

# Running the Robbins-Monro algorithm
estimates = robbins_monro(f, initial_guess, num_iterations)
print(f"Final estimate of the root: {estimates[-1]}")
#print(f"Estimates: {estimates}")
