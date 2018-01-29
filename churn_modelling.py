import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from  sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('Churn_modelling.csv')


y= data['Exited']
data= data.drop(['CustomerId','Surname','RowNumber','Exited'],axis=1)

X= pd.get_dummies(data,columns=['Geography','Gender'],drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42) #test size reduced interntionally

#clf =  XGBClassifier()

#clf.fit(X_train,y_train)
#y_pred =  clf.predict(X_test)
#print(classification_report(y_test,y_pred))


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier


#Building ANN structure
def build_classifier():
    model = Sequential()
    model.add(Dense(units = 10,activation='relu',input_shape=(11,)))
    model.add(Dense(units = 10,activation='relu'))   
    model.add(Dense(units=1,activation='sigmoid')) ##use softmax if you have more than two classes
    model.compile(optimizer='adam',metrics=['accuracy'],loss='binary_crossentropy')
    return model

classifier= KerasClassifier(build_fn = build_classifier)

sc = StandardScaler()
X_train= sc.fit_transform(X_train)
X_test = sc.transform(X_test)

parameters = {'epochs':[15,30,60],'batch_size':[16,32]}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid=parameters,
                           cv=5,
                           scoring ='accuracy')

grid_search.fit(X_train,y_train)
print(grid_search.best_score_)
print(grid_search.best_params_)

#classifier= KerasClassifier(build_fn = build_classifier,epochs = 60,batch_size=32)
#score2= cross_val_score(estimator = classifier,X=X_train,y=y_train,scoring='accuracy',cv=4)


