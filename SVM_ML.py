import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import re
import time

# Funtion to remove emojis and mentions in tweets
def clean_tweet(tweet):
    # Remove mentions
    tweet = re.sub(r'@\w+', '', tweet)
    # Remove emojis
    tweet = tweet.encode('ascii', 'ignore').decode('ascii')
    return tweet


# Load dataset
df=pd.read_csv('C:\\datasets\\mLabel_tweets.csv',  header=0, delimiter=',', usecols=range(1,3))# dataset columns
df['labels'] = df['labels'].apply(lambda x: x.split(' '))
df['cleaned_tweet'] = df['tweet'].apply(clean_tweet)

# we clean train and test at the same time for convenience and simplicity
# Train-test split

X_train, X_test, y_train_0, y_test_0 = train_test_split(df['cleaned_tweet'], df['labels'], test_size=0.2, random_state=42)


# Transform labels
mlb = MultiLabelBinarizer()
# print(mlb.classes_)
y_train = mlb.fit_transform(y_train_0)

# SVM pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', strip_accents='unicode')), # remove frequent English words like "a", "the", etc. Remove accents.
    ('clf', OneVsRestClassifier(SVC(kernel='linear', probability=True))) #, n_jobs=-1 only if test_size>=0.2. Linear kernel provides best results.
])

y_test = mlb.fit_transform(y_test_0)

# Train the model
time_start = time.time()
pipeline.fit(X_train, y_train)
time_tot = time.time()-time_start


# Test model
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, target_names=mlb.classes_))
print("Time required ", time_tot)


#############################################################################################
# QUANTUM

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, confusion_matrix
from qiskit import Aer
from qiskit.algorithms.state_fidelities import ComputeUncompute, StateFidelityResult
from qiskit.primitives import Sampler
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, ZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel, FidelityStatevectorKernel
from qiskit_machine_learning.algorithms import QSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from qiskit.circuit import QuantumCircuit
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score
import numpy as np
import re
import time

# Function to remove emojis and mentions in tweets
def clean_tweet(tweet):
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = tweet.encode('ascii', 'ignore').decode('ascii')
    return tweet

# Load dataset
df=pd.read_csv('C:\\datasets\\mLabel_tweets.csv',  header=0, delimiter=',', usecols=range(1,3))#dataset columns
df['cleaned_tweet'] = df['tweet'].apply(clean_tweet)

df['labels'] = df['labels'].apply(lambda x: x.split(' '))

# Transform labels
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['labels'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_tweet'], y, test_size=0.2, random_state=42)

# Use TF-IDF to transform the text data
min_df = 0 # =20 # Hyperparameter: minimum number of tweets in which a word should appear to be counted. This is a way to reduce features, otherwise there are 16000+.
vectorizer = TfidfVectorizer(stop_words='english', strip_accents='unicode') # , min_df=min_df

X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

# dimension is still too high -> PCA
n_pca = 10 
pca=PCA(n_components=n_pca)

X_train_q = pca.fit_transform(X_train_tfidf)
X_test_q = pca.fit_transform(X_test_tfidf)

# Data are frequencies so they are already normalized


t2 = time.time()

labels = 12
accuracy_QSVC_vec=[]
C_matrix_vec=[]
y_pred_array = np.zeros(y_test.shape)
f1_score_quantum_weighted_vec = []
recall_quantum_weighted_vec=[]
for label in range(labels):
    y_train_label = y_train[:, label]
    y_test_label = y_test[:, label]
    
    feature_map = ZFeatureMap(n_pca)
    new_kernel = FidelityStatevectorKernel(feature_map=feature_map)

    qsvc = QSVC(quantum_kernel=new_kernel)
    qsvc.fit(X_train_q, y_train_label)
    qsvc.score(X_train_q, y_train_label) #training score

    y_pred_quantum = qsvc.predict(X_test_q)
    
    accuracy_QSVC = np.sum(y_test_label == y_pred_quantum) / len(y_pred_quantum)
    accuracy_QSVC_vec.append(accuracy_QSVC)
    unique_labels = np.unique(y_pred_quantum)
    
    C_matrix_quantum = confusion_matrix(y_test_label, y_pred_quantum, normalize='all')
    C_matrix_vec.append(C_matrix_quantum)
    
    f1_score_quantum_weighted = f1_score(y_test_label, y_pred_quantum, average = 'weighted', zero_division = np.nan)
    f1_score_quantum_weighted_vec.append(f1_score_quantum_weighted)
    
    recall_quantum_weighted = recall_score(y_test_label, y_pred_quantum, average = 'weighted', zero_division = np.nan)
    recall_quantum_weighted_vec.append(recall_quantum_weighted)
    
    y_pred_array[:,label] = y_pred_quantum
    
t2_fin = time.time()
y_pred_array_int = y_pred_array.astype(int)
print(classification_report(y_test, y_pred_array_int, target_names=mlb.classes_))
print("Total execution time Quantum implemantation of multilabel SVM ", t2_fin-t2)




print(classification_report(y_test, y_pred_array, target_names=mlb.classes_, zero_division=np.nan))

