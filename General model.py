##Higher complexity function for our bayesian model

import numpy as np
import matplotlib.pyplot as plt

def define_prior(mu_array, sigma_matrix):
    return np.array(mu_array), np.array(sigma_matrix)

def compute_y_parameters(alpha_mu, alpha_sigma, x):
    y_mu = np.dot(alpha_mu, x)
    y_sigma = np.dot(x.T, np.dot(alpha_sigma, x)) + 1  # Variance, not standard deviation
    return y_mu, y_sigma

def compute_posterior(alpha_mu, alpha_sigma, y, y_mu, y_sigma, x):
    y_alpha_cov = np.dot(alpha_sigma, x)
    product = y_alpha_cov / y_sigma
    updated_mu = alpha_mu + product * (y - y_mu)
    updated_sigma = alpha_sigma - np.outer(product, y_alpha_cov)
    return updated_mu, updated_sigma

def simulate_y(alpha, x, epsilon):
    return np.dot(alpha, x) + epsilon

def plot_line(alpha, iteration, x_range, color='green', label='Iteration', linewidth=1):
    x_values = np.linspace(x_range[0], x_range[1], 400)
    y_values = np.polyval(alpha[::-1], x_values)
    plt.plot(x_values, y_values, color=color, label=f'{label} {iteration}', linewidth=linewidth)

# Feature vector now includes cubic term
def F(x):
    return np.array([x**3, x**2, x, 1])  # Cubic feature vector

def g(x, alpha):
    return np.dot(F(x), alpha)

def find_zero_crossing(func, alpha, x_range=(-10, 10), tol=1e-2, max_iterations=100):
    x = np.random.uniform(x_range[0], x_range[1])
    for _ in range(max_iterations):
        fx = func(x, alpha)
        fpx = 3 * alpha[0] * x**2 + 2 * alpha[1] * x + alpha[2]
        if abs(fx) < tol:
            return x
        if abs(fpx) < 1e-6:
            return None
        x = x - fx / fpx
    return None

def main():
    # True function remains quadratic
    true_alpha = np.array([0, 1, 2, 1])  # True function: x^2 + 2x + 1, with zero cubic term
    prior_mu, prior_sigma = define_prior([0, 0, 0, 0], np.eye(4))  # Adjusted for cubic estimation
    num_iterations = 6000
    x_range = (-10, 10)
    bayesian_x_values = []  # List to store the x values for plotting convergence

    true_solution = -1 
    plt.figure(figsize=(10, 6))
    plot_line(true_alpha, 0, x_range, color="black", linewidth=2, label="True quadratic function")

    x = np.random.uniform(x_range[0], x_range[1])
    for i in range(num_iterations):
        x_vector = F(x)
        epsilon = np.random.normal(0, 1)
        y = simulate_y(true_alpha, x_vector, epsilon)

        y_mu, y_sigma = compute_y_parameters(prior_mu, prior_sigma, x_vector)
        updated_alpha_mu, updated_sigma = compute_posterior(prior_mu, prior_sigma, y, y_mu, y_sigma, x_vector)

        color = 'blue' if i == 0 else 'red' if i == num_iterations - 1 else 'green'
        linewidth = 3 if i == 0 or i == num_iterations - 1 else 0.5
        plot_line(updated_alpha_mu, i, x_range, color=color, linewidth=linewidth)

        prior_mu, prior_sigma = updated_alpha_mu, updated_sigma

        x_zero = find_zero_crossing(g, updated_alpha_mu)
        x = x_zero if x_zero is not None else np.random.uniform(x_range[0], x_range[1])
        bayesian_x_values.append(x)  # Save the evaluated x_zero


    plt.title("Convergence of Higher-Order Model to Quadratic Function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()


    plt.figure(figsize=(10, 6))
    plt.plot(bayesian_x_values, label='Bayesian x values', color='blue', marker='o', linestyle='-', markersize=2)
    plt.axhline(true_solution, color='green', linestyle='-', linewidth=2, label='True Solution')
    plt.xlabel('Iteration')
    plt.ylabel('x value')
    plt.title('Convergence of Bayesian Model Estimates')
    plt.yscale('log')  # Optional: log scale for y-axis to focus on convergence
    plt.legend()
    plt.grid(True)
    plt.show()


    print("Final estimated parameters:", prior_mu)
    print("Final X value", x)
    print("Number of iterations:", i)

if __name__ == "__main__":
    main()
