import numpy as np
import pandas as pd
import scipy.io as sanudo
import matplotlib.pyplot as plt



prm = producto_entre_2_matrices
class Datosbro:
    def __init__(self):
        pass

        def sumacumulativa(self, matrix):
        return np.cumsum(matrix)

    def promedio_matrix(self, matrix, window_size):
        return np.convolve(matrix, np.ones(window_size)/window_size, mode='valid')

    def create_random_matrix(self, shape, size):
        return np.random.rand(*shape) * size

    def create_3d_copy(self, matrix):
        return matrix.copy()[:, :, :, -1]

    def display_attributes(self, matrix):
        print("Dimensiones:", matrix.ndim)
        print("Forma:", matrix.shape)
        print("Tamaño:", matrix.size)
        print("Tipo de datos:", matrix.dtype)

    def reshape_to_2d(self, matrix):
        return matrix.reshape(matrix.shape[0], -1)

    def matrix_to_dataframe(self, matrix):
        return pd.DataFrame(matrix)

    def loadmat_s(self, file_path):
        return sanudo.loadmat(file_path)

    def loadcsv_s(self, file_path):
        return pd.read_csv(file_path)

    def operate_along_axis(self, matrix, operation, axis=0):
        if operation == 'suma':
            return np.sum(matrix, axis=axis)
        elif operation == 'resta':
            return np.resta(matrix, axis=axis)
        elif operation == 'mul':
            return np.mul(matrix, axis=axis)
        elif operation == 'div':
            return np.div(matrix, axis=axis)
        elif operation == 'log':
            return np.log(matrix)
        elif operation == 'mean':
            return np.mean(matrix, axis=axis)
        elif operation == 'std':
            return np.std(matrix, axis=axis)

              elif operation == 'prm':
            return np.dot(matrix)

        elif operation == 'inversa':
            return np.linalg.inv(matrix)

        elif operation == 'transpuesta':
            return np.transpose(matrix)

        elif operation == 'correlacion':
            return np.corrcoef(matrix)

        elif operation == 'media_movil':
            return np.convolve(matrix, np.ones(window_size)/window_size, mode='valid')

        elif operation == 'mediana_movil':
            return scipy.signal.medfilt(matrix, kernel_size)

        elif operation == 'desviacion':
            return np.std(matrix)

        elif operation == 'promedio':
            return pd.Series(matrix).ewm(span=span).mean()

        elif operation == 'progresion_lineal':
            return np.polyfit(matrix[:,0], matrix[:,1], 1)

        else:
            raise ValueError("Operación no válida")

    def plot_senal(self, senal):
        plt.plot(senal)
        plt.title("Señal")
        plt.xlabel("Tiempo")
        plt.ylabel("Amplitud")
        plt.grid(True)
        plt.show()

    def plot_histogram(self, data, bins=10):
        plt.hist(data, bins=bins)
        plt.title("Histograma")
        plt.xlabel("Valor")
        plt.ylabel("Frecuencia")
        plt.grid(True)
        plt.show()

    def plot_stems(self, x, y):
        plt.stem(x, y)
        plt.title("Stems")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()

    def plot_barras(self, x, y):
        plt.bar(x, y)
        plt.title("Barras")
        plt.xlabel("Categoría")
        plt.ylabel("Valor")
        plt.grid(True)
        plt.show()

    def plot_pie(self, labels, sizes):
        plt.pie(sizes, labels=labels, autopct='%1.1f%%')
        plt.title("Gráfico")
        plt.show()

class DataAnalysis(Datosbro):
    def __init__(self):
        super().__init__()

def main():
    np.random.seed(40)
    k = Datosbro()

k1=datosbro

# 1.
random_matrix = k.create_random_matrix((10, 10, 30, 40), 1200000)

# 2.
copy_3d_matrix = k.create_3d_copy(random_matrix)

# 3.
k.display_attributes(copy_3d_matrix)

# 4.
reshaped_matrix = k.reshape_to_2d(copy_3d_matrix)

# 5.
df = k.matrix_to_dataframe(reshaped_matrix)

# 6.
mat_data = k.loadmat_s('data.mat')
csv_data = k.loadcsv_s('data.csv')

# 7.
suma_r = k.operate_along_axis(random_matrix, 'suma', axis=0)
resta_r = k.operate_along_axis(random_matrix, 'resta', axis=1)
mul_r = k.operate_along_axis(random_matrix, 'mul', axis=2)
div_r = k.operate_along_axis(random_matrix, 'div', axis=3)
log_r = k.operate_along_axis(random_matrix, 'log')
mean_r = k.operate_along_axis(random_matrix, 'mean')
std_r = k.operate_along_axis(random_matrix, 'std')

# 9.
senal = mat_data['senal']
k.plot_senal(senal)
k.plot_histogram(senal)
k.plot_stems(np.arange(len(senal)), senal)
k.plot_barras(np.arange(10), np.random.randint(1, 10, 10))
k.plot_pie(['A', 'B', 'C', 'D'], [25, 35, 20, 20])

if __name__ == "__main__":
    main()
