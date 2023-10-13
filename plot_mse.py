import matplotlib.pyplot as plt

def plot_mse(mse_values):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(mse_values)), mse_values, label='MSE')
    plt.xlabel('Training Steps')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('MSE During Training')
    plt.legend()
    plt.show()
