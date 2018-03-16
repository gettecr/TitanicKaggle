#this module takes the traning and test Titanic survival data from the Kaggle competiton and builds an appropriate dataset for modeling

#import statements
import os
import logging
from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd



def main():
    #read raw data files
    trainfile = '../../data/raw/train.csv'
    testfile = '../../data/raw/test.csv'

    df_train = pd.read_csv(trainfile)
    df_test = pd.read_csv(testfile)

    combine = [df_train, df_test]

    #Each name has a title which is indiciative of their age. (Age is highly predictive of survival) 
    #Extract titles from individual names
    
    for data in combine:
        data['Title']=data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

   
    for data in combine:
        #Set Female dummy variable (Sex is also highly predictive)
        data["Female"]=np.where(data['Sex']=='female',1,0)


        #Fill missing age data according to mean by title (this is the most straightforward way to get ages. Splitting by title improves prediction over
        #using the total population)
    
        #Replace the rarer titles with a common "other" title
        data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                               'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
        
        #Also replace different names for similar things with the more common name
        data['Title'] = data['Title'].replace('Mlle', 'Miss')
        data['Title'] = data['Title'].replace('Ms', 'Miss')
        data['Title'] = data['Title'].replace('Mme', 'Mrs')
    
        #Calculate means of ages grouped by title
        means = data.groupby('Title', as_index=False).agg(
                      {'Age':['mean','std']})
        means.columns=['Title','Age_mean',"Age_std"]
        means=means.set_index('Title')
    
        #Choose from means if the value from "Age" is null
        data['Age_mod_mean']=data['Age']
        for title in data["Title"].unique():
            data.loc[(data["Title"]==title)&(data['Age_mod_mean'].isnull()),['Age_mod_mean']]= means.loc[title,'Age_mean']

    #bin ages into 5 groups for predictions
    for data in combine: 
        data['Age_bin_mean']=data['Age_mod_mean']

        binfrom = 'Age_mod_mean'
        bintype = 'Age_bin_mean'
        data.loc[ data[binfrom] <= 16, bintype] = 1
        data.loc[(data[binfrom] > 16) & (data[binfrom] <= 25), bintype] = 2
        data.loc[(data[binfrom] > 25) & (data[binfrom] <= 40), bintype] = 3
        data.loc[(data[binfrom] > 40) & (data[binfrom] <= 60), bintype] = 4
        data.loc[ data[binfrom] > 60, bintype]=5

    #Find most common place passengers embarked
    most_embarked = df_train.Embarked.dropna().mode()[0]

    #Fill in missing data for "Embarked" with most common
    for data in combine:
        data['Embarked'] = data['Embarked'].fillna(most_embarked)
        
    #Make dummy variables for the embarked locations
    embarked1 = pd.get_dummies(df_train['Embarked'], prefix = 'from')
    df_train = df_train.join(embarked1,how='outer')

    embarked2 = pd.get_dummies(df_test['Embarked'], prefix = 'from')
    df_test = df_test.join(embarked2,how='outer')

    #Make dummy variables for each age bin
    ages = pd.get_dummies(df_train['Age_bin_mean'],prefix='inAge_mean')
    df_train = df_train.join(ages)

    ages = pd.get_dummies(df_test['Age_bin_mean'],prefix='inAge_mean')
    df_test = df_test.join(ages)

    #Rename columns
    df_train.rename(index=str, columns={'inAge_mean_1.0':'Agem_0-16', 'inAge_mean_2.0':'Agem_16-25', 'inAge_mean_3.0':'Agem_25-40',
                                        'inAge_mean_4.0':'Agem_40-60', 'inAge_mean_5.0':'Agem_60+'}, inplace=True)

    df_test.rename(index=str, columns={'inAge_mean_1.0':'Agem_0-16', 'inAge_mean_2.0':'Agem_16-25', 'inAge_mean_3.0':'Agem_25-40',
                                       'inAge_mean_4.0':'Agem_40-60', 'inAge_mean_5.0':'Agem_60+'}, inplace=True)
    
    #Calculate family size from sibilng/spouse and Parent/child data
    df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1
    df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1


    cols_to_keep_train = ['PassengerId',
                          'Survived',
                          'Pclass',
                          'Name',
                          'Female',"FamilySize",
                          'Agem_0-16',
                          'Agem_16-25',
                          'Agem_25-40',
                          'Agem_40-60',
                          'Agem_60+','from_C']

    cols_to_keep_test = ['PassengerId',
                         'Pclass',
                         'Name',
                         'Female',"FamilySize",
                         'Agem_0-16',
                         'Agem_16-25',
                         'Agem_25-40',
                         'Agem_40-60',
                         'Agem_60+','from_C']

    data_out_train = df_train[cols_to_keep_train]
    data_out_test = df_test[cols_to_keep_test]

    #Output to "processed" file
    data_out_train.to_csv('../../data/processed/train_1.0.csv', encoding='utf-8')
    data_out_test.to_csv('../../data/processed/test_1.0.csv', encoding='utf-8')

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
