import matplotlib.pyplot as plt
import os

if not os.path.exists('images'):
    os.makedirs('images')


def plot_rmse_comparison(rmse_values, models, filename = 'rmse_comparison.png'):
    plt.figure(figsize=(10, 5))
    plt.bar(models, rmse_values, color='skyblue')
    plt.xlabel('Model')
    plt.ylabel('RMSE')
    plt.title('RMSE Comparison of Models')
    plt.savefig('images/' + filename)  # Guardar gráfica como imagen
    plt.close()  # Cerrar la figura para evitar que se mantenga en memoria
    plt.show()

def plot_training_time_comparison(models, training_times, filename = 'training_time_comparison.png'):
    plt.figure(figsize=(10, 5))
    plt.bar(models, training_times, color='lightgreen')
    plt.xlabel('Model')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time Comparison of Models')
    plt.savefig('images/' + filename)  # Guardar gráfica como imagen
    plt.close()  # Cerrar la figura para evitar que se mantenga en memoria
    plt.show()
