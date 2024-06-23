
import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import time

# Load data
datos = datasets.load_breast_cancer()

"""******************** TRAINING THE MODEL ***************************************************************************"""

X_train, X_test, Y_train, Y_test = train_test_split(datos.data, datos.target, test_size=0.25, random_state=123, stratify=datos.target)

train_df=pd.DataFrame(X_train)
train_df['clases'] = Y_train
clases=2
descriptores=30
alfa_vec=np.linspace(0,1,21) #hyper-parameter tuning
matrizprom=np.zeros((clases,descriptores))
variables=train_df.dtypes #Explore variables
print(variables)


time_begin_fit = time.time()

# Normalize.
for j in range(0, descriptores):
    for i in range(0,len(train_df)):
        train_df.loc[i, repr(j)+repr(j)] = (train_df.loc[i,j] - train_df[j].min()) / (train_df[j].max() - train_df[j].min())
# print(train_df)
train_dfnorm=train_df
train_dfnormorig=train_df # df for plots
for i in range(0, descriptores):
    train_dfnorm=train_dfnorm.drop(i, axis=1) # Remove values of non-standardized variables
# print(train_dfnorm)

# M_av
for i in range(clases):
    grupo=train_dfnorm['clases']==i # (vect True/False)
    grupofiltrado=train_dfnorm[grupo] # filter by classes
    #print(grupofiltrado)
    for j in range(0, descriptores):
        matrizprom[i-1][j]=grupofiltrado[repr(j)+repr(j)].mean()
#print(matrizprom)

#MAD (for each descriptor of each instance to each class)
for i in range(0, clases):
    for j in range(0, descriptores):
        for k in range (0, len(train_dfnorm)):
            train_dfnorm.loc[k, 'MAD'+repr(i)+repr(j)]= matrizprom[i][j]**(train_dfnorm.loc[k,repr(j)+repr(j)])* \
            (1-matrizprom[i][j])**(1-train_dfnorm.loc[k,repr(j)+repr(j)])
           
# GAD (three methods, comment/uncomment)


f1_score_vec=[]
accuracy_vec=[]
precision_vec=[]
train_dfnorm_list=[]
train_dflisto_list=[]

for alfa in alfa_vec:
    for i in range(0, clases):
        for j in range(0, len(train_dfnorm)):
            
            # Lukasiewicz
            # train_dfnorm.loc[j, 'Tnorm'+repr(i)] = max(1-descriptores + np.sum([train_dfnorm.loc[j, 'MAD'+repr(i)+repr(k)] for k in range(descriptores)]),0)
            # train_dfnorm.loc[j, 'Sconorm'+repr(i)] = min(np.sum([train_dfnorm.loc[j, 'MAD'+repr(i)+repr(k)] for k in range(descriptores)]),0)
            # train_dfnorm.loc[j,'GAD'+repr(i)]=alfa*train_dfnorm.loc[j,'Tnorm'+repr(i)]+(1-alfa)*train_dfnorm.loc[j,'Sconorm'+repr(i)]

            # Min-max
            # train_dfnorm.loc[j, 'MaximoC'+repr(i)] = np.max([train_dfnorm.loc[j, 'MAD'+repr(i)+repr(k)] for k in range(descriptores)])
            # train_dfnorm.loc[j, 'MinimoC'+repr(i)] = np.min([train_dfnorm.loc[j, 'MAD'+repr(i)+repr(k)] for k in range(descriptores)])
            # train_dfnorm.loc[j,'GAD'+repr(i)]=alfa*train_dfnorm.loc[j,'MinimoC'+repr(i)]+(1-alfa)*train_dfnorm.loc[j,'MaximoC'+repr(i)]
            
            # producto de MAD. En este caso la mejor
            product = 1
            for k in range(descriptores):
                product *= train_dfnorm.loc[j, 'MAD'+repr(i)+repr(k)]
            train_dfnorm.loc[j, 'PROD_MAD'+repr(i)] = product
            train_dfnorm.loc[j, 'GAD'+repr(i)] = alfa * product + (1 - alfa) * (1 - product)

    for i in range(0,len(train_dfnorm)):
        train_dfnorm.loc[i,'index']=max(train_dfnorm.loc[i,'GAD0'],train_dfnorm.loc[i,'GAD1'])
        if (train_dfnorm.loc[i,'index']==train_dfnorm.loc[i,'GAD0']):
            train_dfnorm.loc[i,'clase1']=0
        else:
            train_dfnorm.loc[i,'clase1']=1
    train_dflisto = train_dfnorm.iloc[:, list(range(1, descriptores + 1)) + [0] + [len(train_dfnorm.columns) - 1]] #plot and scores
    #print(train_dflisto)
    train_dflisto.loc[:, 'clase1'] = train_dflisto['clase1'].astype(int) #convierto la column de la clasificaciòn en valores enteros
    # train_dflisto['clase1']=train_dflisto['clase1'].astype(int) # problems copy/ref semantics
    #print(train_dflisto['clase1'])
    accuracy1= accuracy_score(train_dflisto['clases'], train_dflisto['clase1'])
    f1_1= f1_score(train_dflisto['clases'], train_dflisto['clase1'], average= 'micro')
    precision= precision_score(train_dflisto['clases'], train_dflisto['clase1'], average='macro')
    
    train_dfnorm_list.append(train_dfnorm.copy()) # Save a copy, otherwise reference semantics make the list a repetition of the dataframe for the last alpha
    train_dflisto_list.append(train_dflisto) # save dataframes so that we can access the one corresponding to the best score
    f1_score_vec.append(f1_1)
    accuracy_vec.append(accuracy1)
    precision_vec.append(precision)
time_end_fit = time.time()
time_fit = time_end_fit - time_begin_fit

# recover train_dfnorm and train_dflisto for highest f1_score -> correct plots
train_dfnorm = train_dfnorm_list[f1_score_vec.index(max(f1_score_vec))]
train_dflisto = train_dflisto_list[f1_score_vec.index(max(f1_score_vec))]
alfa_best = alfa_vec[f1_score_vec.index(max(f1_score_vec))] # best alpha, which we will use for test set
# Leave data frames with best classification

print('TRAIN SCORES')
print('The maximum value for F1 Score is : ' , max(f1_score_vec), ' and it corresponds to alpha = ', alfa_vec[f1_score_vec.index(max(f1_score_vec))])
print('The maximum value for accuracy is : ' , max(accuracy_vec), ' and it corresponds to alpha = ', alfa_vec[accuracy_vec.index(max(accuracy_vec))])
print('The maximum value for precision is : ' , max(precision_vec), ' and it corresponds to alpha = ', alfa_vec[precision_vec.index(max(precision_vec))])

print('alfas ', alfa_vec , "\n" , 'F1 scores ', f1_score_vec , "\n", 'accuracies ' , accuracy_vec , "\n" , 'precision' , precision_vec)
print('Time required to fit the model ', time_fit)
print('END TRAIN SCORES')


"""********************END TRAINING/BEGINNING TESTING (VALIDATION)***************************************************************************"""

test_df=pd.DataFrame(X_test)
test_df['clases'] = Y_test

time_begin_test = time.time()

# Normalize.
for j in range(0, descriptores):
    for i in range(0,len(test_df)):
        test_df.loc[i, repr(j)+repr(j)] = (test_df.loc[i,j] - test_df[j].min()) / (test_df[j].max() - test_df[j].min())
# print(test_df)
test_dfnorm=test_df
test_dfnormorig=test_df # df for plots
for i in range(0, descriptores):
    test_dfnorm=test_dfnorm.drop(i, axis=1) # Remove non-standardized vars
# print(test_dfnorm)

# MAD
for i in range(0, clases):
    for j in range(0, descriptores):
        for k in range (0, len(test_dfnorm)):
            test_dfnorm.loc[k, 'MAD'+repr(i)+repr(j)]= matrizprom[i][j]**(test_dfnorm.loc[k,repr(j)+repr(j)])* \
            (1-matrizprom[i][j])**(1-test_dfnorm.loc[k,repr(j)+repr(j)])

# GAD

for i in range(0, clases):
    for j in range(0, len(test_dfnorm)):
        
        # Lukasiewicz
        # test_dfnorm.loc[j, 'Tnorm'+repr(i)] = max(1-descriptores + np.sum([test_dfnorm.loc[j, 'MAD'+repr(i)+repr(k)] for k in range(descriptores)]),0)
        # test_dfnorm.loc[j, 'Sconorm'+repr(i)] = min(np.sum([test_dfnorm.loc[j, 'MAD'+repr(i)+repr(k)] for k in range(descriptores)]),0)
        # test_dfnorm.loc[j,'GAD'+repr(i)]=alfa_best*test_dfnorm.loc[j,'Tnorm'+repr(i)]+(1-alfa_best)*test_dfnorm.loc[j,'Sconorm'+repr(i)]

        # Min-max
        # test_dfnorm.loc[j, 'MaximoC'+repr(i)] = np.max([test_dfnorm.loc[j, 'MAD'+repr(i)+repr(k)] for k in range(descriptores)])
        # test_dfnorm.loc[j, 'MinimoC'+repr(i)] = np.min([test_dfnorm.loc[j, 'MAD'+repr(i)+repr(k)] for k in range(descriptores)])
        # test_dfnorm.loc[j,'GAD'+repr(i)]=alfa_best*test_dfnorm.loc[j,'MinimoC'+repr(i)]+(1-alfa_best)*test_dfnorm.loc[j,'MaximoC'+repr(i)]
        
        # MAD product. Best in this case
        product = 1
        for k in range(descriptores):
            product *= test_dfnorm.loc[j, 'MAD'+repr(i)+repr(k)]
        test_dfnorm.loc[j, 'PROD_MAD'+repr(i)] = product
        test_dfnorm.loc[j, 'GAD'+repr(i)] = alfa_best * product + (1 - alfa_best) * (1 - product)

for i in range(0,len(test_dfnorm)):
    test_dfnorm.loc[i,'index']=max(test_dfnorm.loc[i,'GAD0'],test_dfnorm.loc[i,'GAD1'])
    if (test_dfnorm.loc[i,'index']==test_dfnorm.loc[i,'GAD0']):
        test_dfnorm.loc[i,'clase1']=0
    else:
        test_dfnorm.loc[i,'clase1']=1


test_dflisto = test_dfnorm.iloc[:, list(range(1, descriptores + 1)) + [0] + [len(test_dfnorm.columns) - 1]] #plots and scores
#print(test_dflisto)
test_dflisto.loc[:, 'clase1'] = test_dflisto['clase1'].astype(int) #convert to integers
# test_dflisto['clase1']=train_dflisto['clase1'].astype(int) # copy/ref semantics
#print(test_dflisto['clase1'])
accuracy1_test = accuracy_score(test_dflisto['clases'], test_dflisto['clase1'])
f1_1_test = f1_score(test_dflisto['clases'], test_dflisto['clase1'], average= 'micro')
precision_test = precision_score(test_dflisto['clases'], test_dflisto['clase1'], average='macro')

time_end_test = time.time()
time_test = time_end_test - time_begin_test

print('TEST SCORES')
print('The (test) value for F1 Score is : ' , f1_1_test)
print('The (test) value for accuracy is : ' , accuracy1_test)
print('The (test) value for precision is : ' , precision_test)

print('Time required to test the model ', time_test)
print('END TEST SCORES')

"""********************END OF CLASSICAL METRIC CLASSIFICATION/QUANTUM IMPLEMENTATION***************************************************************************"""

# PREPARE FUNCTIONS FOR MAD -> (GAD) -> CLASS

from qiskit import QuantumCircuit, Aer, transpile, assemble, execute
from qiskit.providers.aer import AerSimulator


# def standardize_vector(vector):
#     min_val = np.min(vector)
#     max_val = np.max(vector)
#     if min_val != max_val: #this is the general case
#         return (vector - min_val) / (max_val - min_val)
#     else:
#         return vector
    
def encode_vector(qc, vector): #requires vector values to be in [0,1], otherwise standardize
    for i, value in enumerate(vector):
        # Ry gate
        theta = 2 * np.arcsin(np.sqrt(value))
        qc.ry(theta, i)

def create_circuit(vector):
    num_qubits = len(vector) 
    qc = QuantumCircuit(num_qubits, num_qubits)
    # vector = standardize_vector(vector)
    encode_vector(qc, vector)
    
    # qc.measure(range(num_qubits), range(num_qubits))
    
    return qc


def simulate_circuit(qc, vector):
    sim = AerSimulator()
    qc.measure(range(len(vector)), range(len(vector)))
    job = execute(qc, sim, shots = 130) # we simulate a moderate number of times to introduce some randomness. Can be tuned
    result = job.result()
    counts = result.get_counts()

    return counts

def counts_to_vector(counts, num_qubits):
    ones_count = np.zeros(num_qubits)
    
    for outcome, count in counts.items():
        bits = list(outcome)
        for i, bit in enumerate(bits):
            if bit == '1':
                ones_count[i] += count
                
    return ones_count

def compare_vectors(vector1, vector2, alfa_best):
    # vector1 = standardize_vector(vector1)
    # vector2 = standardize_vector(vector2)
    
    # vector1 corresponds to an observation's MAD for class 0
    # vector1 corresponds to an observation's MAD for class 1
    
    qc1 = create_circuit(vector1)
    qc2 = create_circuit(vector2)    
    
    raw_counts1 = simulate_circuit(qc1,vector1)
    raw_counts2 = simulate_circuit(qc2,vector2)
    
    counts1 = counts_to_vector(raw_counts1,len(vector1))[::-1] # qiskit reverses order of qbits
    counts2 = counts_to_vector(raw_counts2,len(vector2))[::-1] # qiskit reverses order of qbits

    # compare the number of ones in each vector for each descriptor
    # If one vector has more entries ==1 than the other, it means it has a higher MAD for the 
    # corresponding category, so it is closer to such category (as for that descriptor)
    
    aux_larger = np.array(counts1) > np.array(counts2)
    aux_smaller = np.array(counts1) < np.array(counts2)
    
    if sum(aux_larger) > sum(aux_smaller):
        res = 0
    elif sum(aux_larger) > sum(aux_smaller):
        res = 1
    else: # if there is a tie we sum the global number of ones
        if sum(counts1) >= sum(counts2):
            res = 0
        else:
            res = 1
            
    if alfa_best >=0.5: # alfa_best >=0.5 means higher weight of T-norm, otherwise S-conorm. So the result is opposite in each case.
        return res
    else:
        return 1 - res

# CLASSIFY OUR TEST DATA

test_q_df=pd.DataFrame(X_test)
test_q_df['clases'] = Y_test

time_begin_test_q = time.time()

# Normalize.
for j in range(0, descriptores):
    for i in range(0,len(test_q_df)):
        test_q_df.loc[i, repr(j)+repr(j)] = (test_q_df.loc[i,j] - test_q_df[j].min()) / (test_q_df[j].max() - test_q_df[j].min())
# print(test_q_df)
test_q_dfnorm=test_q_df
test_q_dfnormorig=test_q_df # df plots
for i in range(0, descriptores):
    test_q_dfnorm=test_q_dfnorm.drop(i, axis=1) # Remove non standardized values
# print(test_q_dfnorm)

# MAD using la train avg matrix, and simultaneous classification

for k in range (0, len(test_q_dfnorm)):
    for j in range(0, descriptores):
        for i in range(0, clases):
            test_q_dfnorm.loc[k, 'MAD'+repr(i)+repr(j)]= matrizprom[i][j]**(test_q_dfnorm.loc[k,repr(j)+repr(j)])* \
            (1-matrizprom[i][j])**(1-test_q_dfnorm.loc[k,repr(j)+repr(j)])
    
    vector0 = [test_q_dfnorm.loc[k, f'MAD0{j}'] for j in range (descriptores)]
    vector1 = [test_q_dfnorm.loc[k, f'MAD1{j}'] for j in range (descriptores)]
    test_q_dfnorm.loc[k, 'clase1'] = compare_vectors(vector0, vector1, alfa_best)


test_q_dflisto = test_q_dfnorm.iloc[:, list(range(1, descriptores + 1)) + [0] + [len(test_q_dfnorm.columns) - 1]] #plots and scores
#print(test_q_dflisto)
test_q_dflisto.loc[:, 'clase1'] = test_q_dflisto['clase1'].astype(int) #convierto la column de la clasificaciòn en valores enteros
# test_q_dflisto['clase1']=train_dflisto['clase1'].astype(int) #  copy/ref semantics
#print(test_q_dflisto['clase1'])
accuracy1_test_q = accuracy_score(test_q_dflisto['clases'], test_q_dflisto['clase1'])
f1_1_test_q = f1_score(test_q_dflisto['clases'], test_q_dflisto['clase1'], average= 'micro')
precision_test_q = precision_score(test_q_dflisto['clases'], test_q_dflisto['clase1'], average='macro')

time_end_test_q = time.time()
time_test_q = time_end_test_q - time_begin_test_q

print('QUANTUM TEST SCORES')
print('The (quantum test) value for F1 Score is : ' , f1_1_test_q)
print('The (quantum test) value for accuracy is : ' , accuracy1_test_q)
print('The (quantum test) value for precision is : ' , precision_test_q)

print('Time required to test the model ', time_test_q)
print('END QUANTUM TEST SCORES')








"""********************END OF METRIC CLASSIFICATION/GRAPH OF ORIGINAL DATA*************************************************************************************"""
train_dfnormorig=train_dfnormorig.drop(['clases'], axis=1) # remove class col
test_dfnormorig=test_dfnormorig.drop(['clases'], axis=1) # remove class col
for i in range(0, descriptores):
    train_dfnormorig=train_dfnormorig.drop(i, axis=1) #remove col descriptor 1 (non-standardiezd(train)
    test_dfnormorig=test_dfnormorig.drop(i, axis=1) #remove col descriptor 1 (non-standardiezd(test)

pca=PCA(n_components=2)

pca_brest_train=pca.fit_transform(train_dfnormorig) #coordinates train intances
pca_brest_test=pca.fit_transform(test_dfnormorig) #coordinates test intances

pca_BrestLAMDA_train_df=pd.DataFrame(data=pca_brest_train, columns= ['Component_1', 'Component_2']) 
pca_BrestLAMDA_test_df=pd.DataFrame(data=pca_brest_test, columns= ['Component_1', 'Component_2']) 

pca_nombres_brest_train=pd.concat([pca_BrestLAMDA_train_df, train_dfnorm[['clases']]], axis=1) # Add train label
pca_nombres_brest_test=pd.concat([pca_BrestLAMDA_test_df, test_dfnorm[['clases']]], axis=1) #Add test label

fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(1,1,1)
ax.set_xlabel('Component 1', fontsize=15)
ax.set_ylabel('Component 2', fontsize=15)
ax.set_title('Data Brest Cancer Original Train', fontsize=20)
color_theme= ['red', 'blue']
ax.scatter(x=pca_BrestLAMDA_train_df.Component_1 , y= pca_BrestLAMDA_train_df.Component_2, c=[color_theme[i] for i in pca_nombres_brest_train.clases], s= 25, marker='.')

plt.show()


fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(1,1,1)
ax.set_xlabel('Component 1', fontsize=15)
ax.set_ylabel('Component 2', fontsize=15)
ax.set_title('Data Brest Cancer Original Test', fontsize=20)
color_theme= ['red', 'blue']
ax.scatter(x=pca_BrestLAMDA_test_df.Component_1 , y= pca_BrestLAMDA_test_df.Component_2, c=[color_theme[i] for i in pca_nombres_brest_test.clases], s= 25, marker='.')

plt.show()

""" *********************************END GRAPH OF ORIGINAL DATA****************************************************************************"""

"""***************************GRAPH OF CLASSICAL LAMDA RESULTS*******************************************************"""

pca_nombres_brest_train_1=pd.concat([pca_BrestLAMDA_train_df, train_dflisto[['clase1']]], axis=1) #Add label predicted by the model
pca_nombres_brest_test_1=pd.concat([pca_BrestLAMDA_test_df, test_dflisto[['clase1']]], axis=1) #Add label predicted by the model

fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(1,1,1)
ax.set_xlabel('Component 1', fontsize=15)
ax.set_ylabel('Component 2', fontsize=15)
ax.set_title('Data Brest Cancer Clasificacion (Train) with LAMDA', fontsize=20)
color_theme= ['red', 'blue']
pca_nombres_brest_train_1['clase1'] = pca_nombres_brest_train_1['clase1'].astype(int)
ax.scatter(x=pca_BrestLAMDA_train_df.Component_1 , y= pca_BrestLAMDA_train_df.Component_2, c=[color_theme[i] for i in pca_nombres_brest_train_1.clase1], s= 25, marker='.')
plt.show()


fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(1,1,1)
ax.set_xlabel('Component 1', fontsize=15)
ax.set_ylabel('Component 2', fontsize=15)
ax.set_title('Data Brest Cancer Clasificacion (Test) with LAMDA', fontsize=20)
color_theme= ['red', 'blue']
pca_nombres_brest_test_1['clase1'] = pca_nombres_brest_test_1['clase1'].astype(int)
ax.scatter(x=pca_BrestLAMDA_test_df.Component_1 , y= pca_BrestLAMDA_test_df.Component_2, c=[color_theme[i] for i in pca_nombres_brest_test_1.clase1], s= 25, marker='.')
plt.show()


"""***************************GRAPH OF QUANTUM LAMDA RESULTS*******************************************************"""
pca_nombres_brest_test_1_q=pd.concat([pca_BrestLAMDA_test_df, test_q_dflisto[['clase1']]], axis=1) #Add label predicted by the model


fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(1,1,1)
ax.set_xlabel('Component 1', fontsize=15)
ax.set_ylabel('Component 2', fontsize=15)
ax.set_title('Clasificacion (Test) with Quantum LAMDA', fontsize=20)
color_theme= ['red', 'blue']
pca_nombres_brest_test_1_q['clase1'] = pca_nombres_brest_test_1_q['clase1'].astype(int)
ax.scatter(x=pca_BrestLAMDA_test_df.Component_1 , y= pca_BrestLAMDA_test_df.Component_2, c=[color_theme[i] for i in pca_nombres_brest_test_1_q.clase1], s= 25, marker='.')
plt.show()

"""**********************************************************END OF PROGRAM*******************************************"""

