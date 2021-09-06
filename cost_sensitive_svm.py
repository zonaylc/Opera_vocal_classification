# -*- coding: utf-8 -*-

# Imports
import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# %matplotlib inline
from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC

"""Loading the data"""

# Loading
metadata = np.loadtxt("metadata.txt", delimiter=",", skiprows=1, dtype=int)

X = np.empty((0, 627))
y_male = np.empty(0)
y_female = np.empty(0)
#y_choral = np.empty(0)
performances = np.empty(0)

num_file_pairs_to_load = 39

for idx in range(1, num_file_pairs_to_load + 1):
    samples = np.load('train%02d.npy' % idx, mmap_mode='r')
    X = np.append(X, samples, axis=0)
    performances = np.append(performances, np.full((len(samples),1), metadata[idx-1,1]))
    
    labels = np.load('train%02d.labels.npz' % idx)
    y_male = np.append(y_male, labels['male'][:, 0])
    y_female = np.append(y_female, labels['female'][:, 0])
    #y_choral = np.append(y_choral, labels['choral'][:, 0])

print("X shape:", np.shape(X))
print("y_male shape:", np.shape(y_male))
print("y_female shape:", np.shape(y_female))
#print("y_choral shape:", np.shape(y_choral))
print("performances shape:", np.shape(performances))

"""Sample the same number of samples from each performance"""

# sample 9505 observations in each fold
df = pd.DataFrame(X)
df["Y_female"] = y_female
df["Y_male"] = y_male
df["performance"] = performances
df_sampled = pd.DataFrame(columns=df.columns)

random.seed = 1234
n_samples_per_performance = 9505
for i in range(1,12):
  df_temp = df[df["performance"] == i].sample(n_samples_per_performance)
  df_sampled = df_sampled.append(df_temp)

# back to numpy arrays
y_female = df_sampled["Y_female"].to_numpy()
y_male = df_sampled["Y_male"].to_numpy()
df_sampled = df_sampled.drop(["Y_female", "Y_male", "performance"], axis=1)
X = df_sampled.to_numpy()

print("X shape:", np.shape(X))
print("Y_male shape:", np.shape(y_male))
print("Y_female shape:", np.shape(y_female))

"""Data Preparation - Train Test Split"""

#prepare training data
#transform the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Male:take performance1(9505 records) as training data:
sets = train_test_split(X, y_male, test_size=0.9090909090909091, random_state=False, shuffle=False)
X_train, X_test, m_y_train, m_y_test = sets

# Female
f_y_train = y_female[:9505]

#take part of testing data
X_test = X_test[:4074]
m_y_test = m_y_test[:4074]
f_y_test = y_female[9505:13579]


print("X Train:", X_train.shape)
print("Y Train(male/female):", m_y_train.shape, f_y_train.shape)
print("X Test:", X_test.shape)
print("Y Test(male/female):", m_y_test.shape, f_y_test.shape)

"""Cost-Sensitive SVM"""

#pre-processing
# PCA
pca = PCA(n_components=159)
# LDA
lda = LinearDiscriminantAnalysis(n_components=1)

"""**Male**

Non-cs
"""

### Male Classifier
# define model
model = LinearSVC()
clf1 = make_pipeline(pca, model)
clf2 = make_pipeline(lda, model)

#PCA result
clf1.fit(X_train, m_y_train)
pca_y_pred = clf1.predict(X_test)
#LDA result
clf2.fit(X_train, m_y_train)
lda_y_pred = clf2.predict(X_test)


#measurement
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
#PCA
print("PCA & SVC")
print("Accuracy:", accuracy_score(m_y_test, pca_y_pred))
print("Recall:", recall_score(m_y_test, pca_y_pred))
print("Precision:", precision_score(m_y_test, pca_y_pred))

#LDA
print("\n")
print("LDA & SVC")
print("Accuracy:", accuracy_score(m_y_test, lda_y_pred))
print("Recall:", recall_score(m_y_test, lda_y_pred))
print("Precision:", precision_score(m_y_test, lda_y_pred))

"""CS - grid search"""

model = LinearSVC(class_weight=weights)
clf1 = make_pipeline(pca, model)
clf2 = make_pipeline(lda, model)

# define grid
balance = [{0:1,1:2}, {0:1,1:10}, {0:1,1:20}, {0:1,1:30}, {0:1,1:100}, {0:1,1:200}]
param_grid = dict(class_weight=balance)

# define evaluation procedure
cv = cross_val_score(X_train, m_y_train, cv=5)
# define grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')




#PCA result
clf1.fit(X_train, m_y_train)
pca_y_pred = clf1.predict(X_test)
#LDA result
clf2.fit(X_train, m_y_train)
lda_y_pred = clf2.predict(X_test)


#measurement
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
#PCA
print("PCA & CS-SVC")
print("Accuracy:", accuracy_score(m_y_test, pca_y_pred))
print("Recall:", recall_score(m_y_test, pca_y_pred))
print("Precision:", precision_score(m_y_test, pca_y_pred))

#LDA
print("\n")
print("LDA & CS-SVC")
print("Accuracy:", accuracy_score(m_y_test, lda_y_pred))
print("Recall:", recall_score(m_y_test, lda_y_pred))
print("Precision:", precision_score(m_y_test, lda_y_pred))

"""Different weights"""

weights = {0:1, 1:20}
model = LinearSVC(class_weight=weights)
clf1 = make_pipeline(pca, model)
clf2 = make_pipeline(lda, model)

#PCA result
clf1.fit(X_train, m_y_train)
pca_y_pred = clf1.predict(X_test)
#LDA result
clf2.fit(X_train, m_y_train)
lda_y_pred = clf2.predict(X_test)


#measurement
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
#PCA
print("PCA & CS-SVC")
print("Accuracy:", accuracy_score(m_y_test, pca_y_pred))
print("Recall:", recall_score(m_y_test, pca_y_pred))
print("Precision:", precision_score(m_y_test, pca_y_pred))

#LDA
print("\n")
print("LDA & CS-SVC")
print("Accuracy:", accuracy_score(m_y_test, lda_y_pred))
print("Recall:", recall_score(m_y_test, lda_y_pred))
print("Precision:", precision_score(m_y_test, lda_y_pred))

weights = {0:1, 1:200}
model = LinearSVC(class_weight=weights)
clf1 = make_pipeline(pca, model)
clf2 = make_pipeline(lda, model)

#PCA result
clf1.fit(X_train, m_y_train)
pca_y_pred = clf1.predict(X_test)
#LDA result
clf2.fit(X_train, m_y_train)
lda_y_pred = clf2.predict(X_test)


#measurement
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
#PCA
print("PCA & CS-SVC")
print("Accuracy:", accuracy_score(m_y_test, pca_y_pred))
print("Recall:", recall_score(m_y_test, pca_y_pred))
print("Precision:", precision_score(m_y_test, pca_y_pred))

#LDA
print("\n")
print("LDA & CS-SVC")
print("Accuracy:", accuracy_score(m_y_test, lda_y_pred))
print("Recall:", recall_score(m_y_test, lda_y_pred))
print("Precision:", precision_score(m_y_test, lda_y_pred))

"""**Female**

Non-CS
"""

### Female Classifier
# define model
model = LinearSVC()
clf1 = make_pipeline(pca, model)
clf2 = make_pipeline(lda, model)

#PCA result
clf1.fit(X_train, f_y_train)
pca_y_pred = clf1.predict(X_test)
#LDA result
clf2.fit(X_train, f_y_train)
lda_y_pred = clf2.predict(X_test)


#measurement
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
#PCA
print("PCA & SVC")
print("Accuracy:", accuracy_score(m_y_test, pca_y_pred))
print("Recall:", recall_score(m_y_test, pca_y_pred))
print("Precision:", precision_score(m_y_test, pca_y_pred))

#LDA
print("\n")
print("LDA & SVC")
print("Accuracy:", accuracy_score(m_y_test, lda_y_pred))
print("Recall:", recall_score(m_y_test, lda_y_pred))
print("Precision:", precision_score(m_y_test, lda_y_pred))

"""CS"""

# Female Classifier
# define model
weights = {0:1, 1:100}
model = LinearSVC(class_weight=weights)
clf1 = make_pipeline(pca, model)
clf2 = make_pipeline(lda, model)

#PCA result
clf1.fit(X_train, f_y_train)
pca_y_pred = clf1.predict(X_test)
#LDA result
clf2.fit(X_train, f_y_train)
lda_y_pred = clf2.predict(X_test)


#measurement
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
#PCA
print("PCA & CS-SVC")
print("Accuracy:", accuracy_score(m_y_test, pca_y_pred))
print("Recall:", recall_score(m_y_test, pca_y_pred))
print("Precision:", precision_score(m_y_test, pca_y_pred))

#LDA
print("\n")
print("LDA & CS-SVC")
print("Accuracy:", accuracy_score(m_y_test, lda_y_pred))
print("Recall:", recall_score(m_y_test, lda_y_pred))
print("Precision:", precision_score(m_y_test, lda_y_pred))

"""Different weights"""

# Female Classifier
# define model
weights = {0:1, 1:20}
model = LinearSVC(class_weight=weights)
clf1 = make_pipeline(pca, model)
clf2 = make_pipeline(lda, model)

#PCA result
clf1.fit(X_train, f_y_train)
pca_y_pred = clf1.predict(X_test)
#LDA result
clf2.fit(X_train, f_y_train)
lda_y_pred = clf2.predict(X_test)


#measurement
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
#PCA
print("PCA & CS-SVC")
print("Accuracy:", accuracy_score(m_y_test, pca_y_pred))
print("Recall:", recall_score(m_y_test, pca_y_pred))
print("Precision:", precision_score(m_y_test, pca_y_pred))

#LDA
print("\n")
print("LDA & CS-SVC")
print("Accuracy:", accuracy_score(m_y_test, lda_y_pred))
print("Recall:", recall_score(m_y_test, lda_y_pred))
print("Precision:", precision_score(m_y_test, lda_y_pred))

# Female Classifier
# define model
weights = {0:1, 1:200}
model = LinearSVC(class_weight=weights)
clf1 = make_pipeline(pca, model)
clf2 = make_pipeline(lda, model)

#PCA result
clf1.fit(X_train, f_y_train)
pca_y_pred = clf1.predict(X_test)
#LDA result
clf2.fit(X_train, f_y_train)
lda_y_pred = clf2.predict(X_test)


#measurement
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
#PCA
print("PCA & CS-SVC")
print("Accuracy:", accuracy_score(m_y_test, pca_y_pred))
print("Recall:", recall_score(m_y_test, pca_y_pred))
print("Precision:", precision_score(m_y_test, pca_y_pred))

#LDA
print("\n")
print("LDA & CS-SVC")
print("Accuracy:", accuracy_score(m_y_test, lda_y_pred))
print("Recall:", recall_score(m_y_test, lda_y_pred))
print("Precision:", precision_score(m_y_test, lda_y_pred))