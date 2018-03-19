#this module takes the traning and test Titanic survival data from the Kaggle competiton and builds an appropriate dataset for modeling

#import statements
import os
import logging
from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd

 #Each name has a title which is indiciative of their age. (Age is highly predictive of survival) 
#Extract titles from individual names
    
def get_title(df):
    df['Title']=df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    return df

#Make arrays of all the known ages from each "title" group
def age_array(Title, df):
    ages = df.loc[(df['Title']==Title),'Age'].tolist()
    ages = [x for x in ages if str(x) != 'nan']
    return ages

#Drawing a random age from the distribution of ages for every person according to Title
def draw_rand(Title, df):
    return df.loc[(df['Title']==Title),'Age_rand'].apply(lambda x: np.random.choice( np.array(age_array(Title,df))))

def imputate_age(df):
    #Fill missing age data in two ways: Random choices from a distribution according to title (Simple random imputation),
    #and according to means grouped by title
    
    #Replace the rarer titles with a common "other" title
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    #Calculate means of ages grouped by title
    means = df.groupby('Title', as_index=False).agg(
                      {'Age':['mean','std']})
    means.columns=['Title','Age_mean',"Age_std"]
    means=means.set_index('Title')
    
    
    #Make new column drawing a random age from the distribution of ages for every person according to Title
    
    df['Age_rand']=df['Age']
    
    for Title in df['Title'].unique():
        df.loc[(df['Title']==Title),'Age_rand']=draw_rand(Title, df)
    
    #Make a column which chooses from "Age" if known, or "Age_rand" if unknown
    df['Age_mod_rand']=np.where(df['Age'].isnull(),df['Age_rand'],df['Age'])
    
    #Choose from means instead of random
    df['Age_mod_mean']=df['Age']
    for title in df["Title"].unique():
        df.loc[(df["Title"]==title)&(df['Age_mod_mean'].isnull()),['Age_mod_mean']]= means.loc[title,'Age_mean']
    return df

def cut_age(df, colName, cutPoints, labels,suffix):
    df['Age_bin_'+str(suffix)]=pd.cut(df[colName],cutPoints,labels=labels)
    return df

def model_vars(df):
    df["Female"]=np.where(df['Sex']=='female',1,0)
    
    #assign dummies to categories
    pclass = pd.get_dummies(df['Pclass'],prefix='class')
    df = df.join(pclass,how='outer')
    df.head()
    
    embarked = pd.get_dummies(df['Embarked'], prefix = 'from')
    df = df.join(embarked,how='outer')
    
    ages1 = pd.get_dummies(df['Age_bin_rand'],prefix='inAge_rand')
    ages2 = pd.get_dummies(df['Age_bin_mean'],prefix='inAge_mean')
    df = df.join(ages1,how='outer')
    df = df.join(ages2,how='outer')
    
    #set new family size variable
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    #include constant term for stats models regression
    df['Eins']=1.0
    return df


def main():
    #read raw data files
    trainfile = 'data/raw/train.csv'
    testfile = 'data/raw/test.csv'

    df_train = pd.read_csv(trainfile)
    df_test = pd.read_csv(testfile)
    
    #Extract title feature from name
    df_train = get_title(df_train)
    df_test = get_title(df_test)
    
    #Imputate new ages to fill in missing data
    df_train = imputate_age(df_train)
    df_test = imputate_age(df_test)

    #Assign age bins for different age groups
    cutPoints =[0,5,12,18,35,60,100]
    labels = ["Infant","Child","Teenager","Young Adult","Adult","Senior"]
 
    df_train = cut_age(df_train,'Age_mod_rand',cutPoints,labels,"rand")
    df_train = cut_age(df_train,'Age_mod_mean',cutPoints,labels,"mean")

    df_test = cut_age(df_test,'Age_mod_rand',cutPoints,labels,"rand")
    df_test = cut_age(df_test,'Age_mod_mean',cutPoints,labels,"mean")
    
    #Fill missing embarkation data with most common port
    most_embarked = df_train.Embarked.dropna().mode()[0]

    df_train['Embarked'] = df_train['Embarked'].fillna(most_embarked)
    df_test['Embarked'] = df_test['Embarked'].fillna(most_embarked)

    #Fill other variables and dummy variables to be used in modeling
    df_train = model_vars(df_train)
    df_test = model_vars(df_test)

    cols_to_keep_train = ['PassengerId',
                         'Survived',
                         'class_1','class_2',
                         'Name',
                         'Female',"FamilySize",'inAge_rand_Infant',
                         'inAge_rand_Child',
                         'inAge_rand_Teenager',
                         'inAge_rand_Young Adult',
                         'inAge_rand_Adult',
                         'inAge_rand_Senior',
                         'inAge_mean_Infant',
                         'inAge_mean_Child',
                         'inAge_mean_Teenager',
                         'inAge_mean_Young Adult',
                         'inAge_mean_Adult',
                         'inAge_mean_Senior','from_C','Eins', 'Age_mod_rand','Age_mod_mean','Age',
                         'Age_bin_rand','Age_bin_mean']

    cols_to_keep_test = ['PassengerId',
                        'class_1','class_2',
                        'Name',
                        'Female',"FamilySize",'inAge_rand_Infant',
                        'inAge_rand_Child',
                        'inAge_rand_Teenager',
                        'inAge_rand_Young Adult',
                        'inAge_rand_Adult',
                        'inAge_rand_Senior',
                        'inAge_mean_Infant',
                        'inAge_mean_Child',
                        'inAge_mean_Teenager',
                        'inAge_mean_Young Adult',
                        'inAge_mean_Adult',
                        'inAge_mean_Senior','from_C','Eins']
    data_out_train = df_train[cols_to_keep_train]
    data_out_test = df_test[cols_to_keep_test]

    #Output to "processed" file
    data_out_train.to_csv('data/processed/train_1.0.csv', encoding='utf-8')
    data_out_test.to_csv('data/processed/test_1.0.csv', encoding='utf-8')

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
