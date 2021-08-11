#reseau de neurone artificiel

#Partie 1 : préparation des donnees 

import numpy as np
import pandas as pd

#importation des donnees 
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

#encoder les variable categorielle dans les donnes 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
preprocess = make_column_transformer(
        (OneHotEncoder(), ['Geography', 'Gender']),
        (StandardScaler(), ['CreditScore', 'Age', 'Tenure', 'Balance',
                            'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
                            'EstimatedSalary']))
X = preprocess.fit_transform(X)
X = np.delete(X, [0,3], 1)

#separation des donnees d'entrainement et de test
y = y.values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Partie 2 construire le reseau de neurone 
# importation des modules de keras 


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialiser le reseau de neurone
classifier = Sequential()

#ajouter la couche d'entrée et une couche cachée 
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

classifier.add(Dropout(rate = 0.1))
#ajouter une deuxieme couche cachée
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dropout(rate = 0.1))
#ajouter la couche de sortie 

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#compiler le reseau de neurone 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#classifier.summary()

#Entrainer le reseau de neurone 
#classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

#Partie 3: faire des Predictions et evaluer le model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) 

# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
Xnew = pd.DataFrame(data={
        'CreditScore': [600], 
        'Geography': ['France'], 
        'Gender': ['Male'],
        'Age': [40],
        'Tenure': [3],
        'Balance': [60000],
        'NumOfProducts': [2],
        'HasCrCard': [1],
        'IsActiveMember': [1],
        'EstimatedSalary': [50000]})
Xnew = preprocess.transform(Xnew)
Xnew = np.delete(Xnew, [0,3], 1)
new_prediction = classifier.predict(Xnew)
new_prediction = (new_prediction > 0.5)

# faire la matrice de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Evaluate

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer='adam'):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# Evaluate
#classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
#accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train,cv = 10)
#mean = accuracies.mean()
#variance = accuracies.std()



#Partie 4 k-fold cross validation

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_













