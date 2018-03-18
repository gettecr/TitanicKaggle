import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


def make_model(trainfile):
    
    df_train = pd.read_csv(trainfile)


    #Based on exploratory notebooks, the following predictors were significant (except 'from_C' p=0.07, but I'm keeping it for now)
    predictors = ['class_1','class_2','Female',"FamilySize",
                          'inAge_rand_Infant','inAge_rand_Child', 'inAge_rand_Young Adult',
                            'from_C','Eins']
    X = df_train[predictors]
    y = df_train['Survived']
    
    logit = sm.Logit(y, X)
    result_rand = logit.fit()

    #output results of logistic regression to a nifty png file
    plt.rc('figure', figsize=(8.5, 5))
    plt.text(0.01, 0.05, str(result_rand.summary()), {'fontsize': 10}, fontproperties = 'monospace') 
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('TestData_meanAge.png')
    plt.cla()
    


    #Exploratory notebooks suggest Random Forest is the most accurate model
    X2=df_train[predictors[:-1]]
    print(predictors[:-1])
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X2,y)
    random_forest.score(X2, y)
    print('Random Forest Score: ' + str(round(random_forest.score(X2, y) * 100, 2)))

    return random_forest

if __name__ == '__main__':
    
    make_model()
    

    
