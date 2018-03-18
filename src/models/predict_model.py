import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import train_model


trainfile = trainfile = '../../data/processed/train_1.0.csv'
testfile = '../../data/processed/test_1.0.csv'

df_test = pd.read_csv(testfile)

#Random forest is the most predictive for this data set

random_forest = train_model.make_model(trainfile)


X_test  = df_test[['class_1','class_2','Female',"FamilySize",
                          'inAge_rand_Infant','inAge_rand_Child', 'inAge_rand_Young Adult','from_C'
                            ]].copy()

Y_pred = random_forest.predict(X_test)

#make submission file according to Kaggle rule
submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('../../data/processed/submission.csv', index=False)
