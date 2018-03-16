import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import train_model


trainfile = trainfile = '../../data/processed/train_1.0.csv'
testfile = '../../data/processed/test_1.0.csv'

df_test = pd.read_csv(testfile)

#Random forest is the most predictive for this data set

random_forest = train_model.make_model(trainfile)


X_test  = df_test[['Pclass','Female',"FamilySize",
                          'Agem_0-16',
                          'Agem_16-25',
                          'Agem_25-40',
                          'Agem_40-60']].copy()

Y_pred = random_forest.predict(X_test)

#make submission file according to Kaggle rule
submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('../../data/processed/submission.csv', index=False)
