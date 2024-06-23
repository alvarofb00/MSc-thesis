# covid dataset
import pandas as pd
import numpy as np
import scipy
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time
from collections import Counter
from sklearn.metrics import f1_score, confusion_matrix


def covid(training_size, test_size, PLOT_DATA=True, PLOT_BARPLOT=True):
    class_labels = [r'Negative', r'Positive']
    training_balance = {'Positive': 10*training_size, 'Negative': 10*training_size} # train the model with balanced number of observations
    test_balance = {'Positive': test_size, 'Negative': 19*test_size} #95% negative, complying with the proportion
    # Load data
    covid_in = pd.read_excel("C:\\datasets\\DataIsrael.xlsx", sheet_name='DataIsraelListaParaAnalisis')
    # filter non-conclusive instances
    covid = covid_in[covid_in['corona_result'] != 3]
    # columns 'head_ache', 'sore_throat' and 'shortness_of_breath' only have 3,5 and 7 non-zero instances. 
    # column 'age_60_and_above' has all instances with the same value. We remove these columns.
    covid = covid.drop(columns=['head_ache', 'sore_throat', 'shortness_of_breath', 'age_60_and_above'])
    covid['corona_result'] = covid['corona_result'].replace(2, 0) #positive associated to 0

    # train-test split
    X_covid = covid.drop('corona_result', axis = 1)
    y_covid = covid['corona_result']
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_covid, y_covid, test_size=0.1, random_state=123, stratify=y_covid)
    # few observations so we stratify
    # Use 90% of the dataset (for both train and test, X_test, Y_test unused)
    
    # standardize.
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # No PCA is needed
    # pca = PCA(n_components=n).fit(X_train)
    # X_train = pca.transform(X_train)
    # X_test = pca.transform(X_test)

    # Scale in [-1,1]
    samples = np.append(X_train, X_test, axis=0)
    minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
    X_train = minmax_scale.transform(X_train)
    X_test = minmax_scale.transform(X_test)

    # Take a sample to train the model

    training_input = {key: (X_train[Y_train == k, :])[:training_balance[key]] for k, key in enumerate(class_labels)}

    test_input = {key: (X_train[Y_train == k, :])[training_balance[key]:(
        training_balance[key] + test_balance[key])] for k, key in enumerate(class_labels)}

    if PLOT_DATA:
        for k in range(0, 2):
            # Define the number of samples for each class
            size = training_balance[class_labels[k]]
            
            x_axis_data = X_train[Y_train == k, 0][:size]
            y_axis_data = X_train[Y_train == k, 1][:size]
    
            label = 'Negative' if k == 1 else 'Positive'
            plt.scatter(x_axis_data, y_axis_data, label=label)
    
        plt.title("Conjunto de datos sobre cáncer de mama (Dimensionalidad Reducida con PCA)")
        plt.legend()
        plt.show()

        
    if PLOT_BARPLOT:
        frequency_dict = Counter(y_covid)
        values = ['Negative', 'Positive']
        frecuencies = list(frequency_dict.values())
        plt.bar(values, frecuencies, color='blue', edgecolor='black')
        
        plt.title('Covid')
        plt.xlabel('Result')
        plt.ylabel('Frequency')
        plt.xticks(values)
        plt.show()

    return X_train, training_input, test_input, class_labels

class svm_utils:

	def make_meshgrid(x, y, h=.02):
	    x_min, x_max = x.min() - 1, x.max() + 1
	    y_min, y_max = y.min() - 1, y.max() + 1
	    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
		                 np.arange(y_min, y_max, h))
	    return xx, yy


	def plot_contours(ax, clf, xx, yy, **params):
	    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	    Z = Z.reshape(xx.shape)
	    out = ax.contourf(xx, yy, Z, **params)
	    return out




def split_dataset_to_data_and_labels(dataset, class_names=None):

    data = []
    labels = []
    if class_names is None:
        sorted_classes_name = sorted(list(dataset.keys()))
        class_to_label = {k: idx for idx, k in enumerate(sorted_classes_name)}
    else:
        class_to_label = class_names
    sorted_label = sorted(class_to_label.items(), key=lambda elem: elem[1])
    for class_name, _ in sorted_label:
        values = dataset[class_name]
        for value in values:
            data.append(value)
            try:
                labels.append(class_to_label[class_name])
            except Exception as ex:  # pylint: disable=broad-except
                raise KeyError('The dataset has different class names to '
                               'the training data. error message: {}'.format(ex)) from ex
    data = np.asarray(data)
    labels = np.asarray(labels)
    if class_names is None:
        return [data, labels], class_to_label
    else:
        return [data, labels]
    
import numpy as np
from sklearn.datasets import make_blobs
from sklearn import svm
from matplotlib import pyplot as plt

training_dataset_size = 10
testing_dataset_size = 5

sample_Total, training_input, test_input, class_labels = covid(training_dataset_size, testing_dataset_size)

data_train, _ = split_dataset_to_data_and_labels(training_input)
data_test, _ = split_dataset_to_data_and_labels(test_input)

# LINEAR SVM

t1_in_linear = time.time()
model= svm.LinearSVC()
model.fit(data_train[0], data_train[1])
t1_end_linear = time.time()
t1_lin_tot = t1_end_linear-t1_in_linear
print("Execution time linear SVM ", t1_lin_tot)

# Accuracy (correctly classified observations)
accuracy_train = model.score(data_train[0], data_train[1])
accuracy_test = model.score(data_test[0], data_test[1])

# F scores

y_pred = model.predict(data_test[0])
f1_score_linear_weighted = f1_score(data_test[1], y_pred, average = 'weighted', zero_division = np.nan)
# take into account label imbalance 
f1_score_linear_micro = f1_score(data_test[1], y_pred, average = 'micro', zero_division = np.nan)

# Confusion matrix (rows==True, col==pred)

C_matrix_linear = confusion_matrix(data_test[1], y_pred, normalize='all')
print(C_matrix_linear)


# Gaussian kernel
t1_in_gauss = time.time()
clf = svm.SVC(gamma = 'scale')
clf.fit(data_train[0], data_train[1]);
t1_end_gauss = time.time()
t1_gauss_tot = t1_end_gauss-t1_in_gauss
print("Execution time gaussian SVM ", t1_gauss_tot)

# Accuracy (correctly classified observations)
accuracy_train_gaussian = clf.score(data_train[0], data_train[1])
accuracy_test_gaussian = clf.score(data_test[0], data_test[1])

# F scores. 

y_pred_gauss = clf.predict(data_test[0])
f1_score_gaussian_weighted = f1_score(data_test[1], y_pred_gauss, average = 'weighted', zero_division = np.nan)
# take into account label imbalance 
f1_score_gaussian_micro = f1_score(data_test[1], y_pred_gauss, average = 'micro', zero_division = np.nan)

# Confusion matrix

C_matrix_gauss = confusion_matrix(data_test[1], y_pred_gauss, normalize='all')
print(C_matrix_gauss)


# QISKIT

import qiskit as qk
from qiskit.providers.aer import AerSimulator
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap, RealAmplitudes, StatePreparation
from qiskit.algorithms.state_fidelities import ComputeUncompute, StateFidelityResult
from qiskit.primitives import Sampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel, FidelityStatevectorKernel
from qiskit_machine_learning.algorithms import QSVC

fidelity = ComputeUncompute(sampler=Sampler())


n=4 #four columns
feature_map = ZFeatureMap(n) # Can be tuned to ZZfeatureMap(n)
new_kernel = FidelityStatevectorKernel(feature_map=feature_map) 
# new_kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

# print('n= ',n, ', train obs= ', 20*training_dataset_size, ', Z', ', FidelityStatevectorKernel')

train_set = data_train[0]
train_labels = data_train[1]
test_set = data_test[0] #if we have run the previous code

t2 = time.time()

qsvc2 = QSVC(quantum_kernel=new_kernel)
qsvc2.fit(train_set, train_labels)
qsvc2.score(train_set, train_labels) #score del conjunto train

y_pred_quantum = qsvc2.predict(test_set)

# Define unique class labels in y_pred_quantum
unique_labels = np.unique(y_pred_quantum)

accuracy_QSVC = np.sum(data_test[1] == y_pred_quantum) / len(y_pred_quantum)
t2_fin = time.time()
print("testing success ratio: ", accuracy_QSVC)
print("Execution time QSVC ", t2_fin-t2)

# F scores

f1_score_quantum_weighted = f1_score(data_test[1], y_pred_quantum, average = 'weighted', zero_division = np.nan)
# take into account label imbalance 
f1_score_quantum_micro = f1_score(data_test[1], y_pred_quantum, average = 'micro', zero_division = np.nan)

# Confusion matrix

C_matrix_quantum = confusion_matrix(data_test[1], y_pred_quantum, normalize='all')

print(C_matrix_quantum)

# Scatterplots
plt.scatter(test_set[:, 0], test_set[:,1], c=y_pred_quantum)
plt.show()

plt.scatter(test_input['Positive'][:,0], test_input['Positive'][:,1])
plt.scatter(test_input['Negative'][:,0], test_input['Negative'][:,1])
plt.show()  





































# =============================================================================
# 
# 
# 
# import qiskit as qk
# 
# # Creating Qubits
# q = qk.QuantumRegister(2)
# # Creating Classical Bits
# c = qk.ClassicalRegister(2)
# 
# # Definir e imprimir circuito vacío.
# #Hasta ahora sólo tenemos un circuito cuántico vacío con 2 qubits (q0_0 y q0_1) y 2 registros clásicos (c0_0 y c0_1).
# 
# 
# circuit = qk.QuantumCircuit(q, c)
# print(circuit)
# 
# # Initialize empty circuit
# circuit = qk.QuantumCircuit(q, c)
# # Hadamard Gate on the first Qubit
# circuit.h(q[0])
# # CNOT Gate on the first and second Qubits
# circuit.cx(q[0], q[1])
# # Measuring the Qubits
# circuit.measure(q, c)
# print (circuit)
# 
# # Ejecutamos el circuito en el simulador cuántico
# # Usando el Simulador Qasm de Qiskit Aer: Defina dónde desea ejecutar la simulación.
# simulator = qk.BasicAer.get_backend('qasm_simulator')
# 
# # Simulando el circuito usando el simulador para obtener el resultado.
# job = qk.execute(circuit, simulator, shots=100)
# result = job.result()
# 
# # Obtenemos los resultados binarios agregados del circuito.
# counts = result.get_counts(circuit)
# print (counts)
# 
# #Construir circuito para mapa de características
# 
# from qiskit.utils import algorithm_globals
# 
# algorithm_globals.random_seed = 123456
# 
# from sklearn.datasets import make_blobs
# 
# features, labels = make_blobs(
#     n_samples=20,
#     centers=2,
#     center_box=(-1, 1),
#     cluster_std=0.1,
#     random_state=algorithm_globals.random_seed,
# )
# 
# from qiskit import BasicAer
# from qiskit.utils import QuantumInstance
# 
# sv_qi = QuantumInstance(
#     BasicAer.get_backend("statevector_simulator"),
#     seed_simulator=algorithm_globals.random_seed,
#     seed_transpiler=algorithm_globals.random_seed,
# )
# 
# # pip install qiskit-machine-learning
# 
# # Crear núcleo cuántico.
# 
# from qiskit.circuit.library import ZZFeatureMap
# # =============================================================================
# # No QuantumKernel
# # from qiskit_machine_learning.kernels import QuantumKernel
# # 
# # feature_map = ZZFeatureMap(2)
# # previous_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=sv_qi)
# # 
# # # Se ajusta el clasificador SVM.
# # 
# # from qiskit_machine_learning.algorithms import QSVC
# # 
# # qsvc = QSVC(quantum_kernel=previous_kernel)
# # qsvc.fit(features, labels)
# # qsvc.score(features, labels)
# # =============================================================================
# 
# # Implementación del kernel cuántico
# 
# from qiskit.algorithms.state_fidelities import ComputeUncompute
# from qiskit.primitives import Sampler
# 
# fidelity = ComputeUncompute(sampler=Sampler())
# 
# # Creamos un nuevo núcleo cuántico con la instancia de fidelidad.
# 
# from qiskit_machine_learning.kernels import FidelityQuantumKernel
# 
# feature_map = ZZFeatureMap(2)
# new_kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
# 
# # Luego ajustamos un clasificador SVM de la misma manera que antes.
# 
# from qiskit_machine_learning.algorithms import QSVC
# 
# qsvc = QSVC(quantum_kernel=new_kernel)
# qsvc.fit(features, labels)
# qsvc.score(features, labels)
# 
# # Imprimimos el circuito del mapa de características
# # Para imprimir el circuito del mapa de características, definimos un vector arbitrario x que queremos codificar y construir el circuito para este punto de datos.
# 
# # pip install pylatexenc
# 
# from qiskit.circuit.library import ZZFeatureMap
# zz = ZZFeatureMap(2, entanglement="full", reps=2)
# zz.decompose().draw()
# 
# # Ejecutamos QSVM
# # La matriz central para el entrenamiento. 
# # Dado que el conjunto de entrenamiento contiene 40 elementos, la matriz del núcleo tiene una dimensión de 40x40.
# 
# plt.scatter(training_input['Positive'][:,0], training_input['Positive'][:,1])
# plt.scatter(training_input['Negative'][:,0], training_input['Negative'][:,1])
# plt.show()
# length_data = len(training_input['Positive']) + len(training_input['Negative'])
# 
# # La tasa de éxito muestra la precisión con la que QSVM predice las etiquetas.
# 
# print("testing success ratio: ", accuracy_test)
# test_set = np.concatenate((test_input['Positive'], test_input['Negative']))
# 
# y_test = qsvc.predict(test_set)
# 
# # Aquí trazamos los resultados. El primer gráfico muestra las predicciones de 
# # etiquetas del QSVM y el segundo gráfico muestra las etiquetas de prueba.
# 
# plt.scatter(test_set[:, 0], test_set[:,1], c=y_test)
# plt.show()
# 
# plt.scatter(test_input['Positive'][:,0], test_input['Positive'][:,1])
# plt.scatter(test_input['Negative'][:,0], test_input['Negative'][:,1])
# plt.show()
# 
# 
# # extra
# accuracy_QSVC = np.sum(data_test[1] == y_test) / len(y_test)
# print("testing success ratio: ", accuracy_QSVC)
# =============================================================================
