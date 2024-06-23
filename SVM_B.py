# breast cancer dataset

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


def breast_cancer(training_size, test_size, n, PLOT_DATA=True, PLOT_BARPLOT=True):
    class_labels = [r'Benign', r'Malignant']
    training_balance = {'Benign': 2*training_size, 'Malignant': training_size} # 63% of obs are benign, so proportion is ~2:1
    test_balance = {'Benign': 2*test_size, 'Malignant': test_size}
    # Load data
    cancer = datasets.load_breast_cancer()

    # train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(cancer.data, cancer.target, test_size=0.1, random_state=123, stratify=cancer.target)
    # few observations so we stratify
    # Use 90% of the dataset (for both train and test, X_test, Y_test unused)
    
    # Standardize.
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Apply PCA to reduce number of components
    pca = PCA(n_components=n).fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    # Scale data in [-1,1]
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
    
            label = 'Malignant' if k == 1 else 'Benign'
            plt.scatter(x_axis_data, y_axis_data, label=label)
    
        plt.title("Conjunto de datos sobre cáncer de mama (Dimensionalidad Reducida con PCA)")
        plt.legend()
        plt.show()

        
    if PLOT_BARPLOT:
        frequency_dict = Counter(cancer.target)
        values = ['Malignant', 'Benign']
        frecuencies = list(frequency_dict.values())
        plt.bar(values, frecuencies, color='blue', edgecolor='black')
        
        plt.title('Cancer type')
        plt.xlabel('Type')
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

n = 2 # number of principal components. Can be tuned
training_dataset_size = 60
testing_dataset_size = 30

sample_Total, training_input, test_input, class_labels = breast_cancer(training_dataset_size, testing_dataset_size, n)

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

# F scores. Micro coincides with accuracy

y_pred = model.predict(data_test[0])
f1_score_linear_weighted = f1_score(data_test[1], y_pred, average = 'weighted', zero_division = np.nan)
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

# F scores. Micro coincides with accuracy

y_pred_gauss = clf.predict(data_test[0])
f1_score_gaussian_weighted = f1_score(data_test[1], y_pred_gauss, average = 'weighted', zero_division = np.nan)
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



feature_map = ZFeatureMap(n) # Can be tuned to ZZfeatureMap(n)
new_kernel = FidelityStatevectorKernel(feature_map=feature_map) # Can be tuned
# new_kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

# print('n= ',n, ', train obs= ', 3*training_dataset_size, ', Z', ', FidelityStatevectorKernel')

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
f1_score_quantum_micro = f1_score(data_test[1], y_pred_quantum, average = 'micro', zero_division = np.nan)

# Confusion matrix

C_matrix_quantum = confusion_matrix(data_test[1], y_pred_quantum, normalize='all')

print(C_matrix_quantum)


# Scatter plot each class separately with its corresponding label (2D)

plt.scatter(test_set[:, 0], test_set[:,1], c=y_pred_quantum)
plt.show()

plt.scatter(test_input['Benign'][:,0], test_input['Benign'][:,1])
plt.scatter(test_input['Malignant'][:,0], test_input['Malignant'][:,1])
plt.show()  




