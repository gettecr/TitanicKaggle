

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


trainfile = '../../data/raw/train.csv'
trainfile_processed = '../../data/processed/train_1.0.csv'

df_train = pd.read_csv(trainfile)
df_processed= pd.read_csv(trainfile_processed)


g = sns.FacetGrid(df_train, col='Survived', aspect=1.6)
g.map(plt.hist, 'Age',bins=20, alpha=0.5)
g.savefig('../../reports/figures/Age-survived.png') 
plt.close()

g2=sns.barplot(data=df_train, x='Sex', y='Survived', alpha=0.5, color='tab:blue',ci=None)
plt.savefig('../../reports/figures/Sex-survived.png')
plt.close()

g3=sns.barplot(data=df_train, x='Pclass', y='Survived', alpha=0.5, color='tab:blue',ci=None)
plt.savefig('../../reports/figures/Class-survived.png')
plt.close()

df_processed['Age_mod_rand'].plot.hist(alpha = 0.3, color='tab:orange',bins=20)
df_processed['Age_mod_mean'].plot.hist(alpha = 0.3, color='tab:blue',bins=20)
df_processed['Age'].plot.hist(alpha = 0.3, color='tab:green',bins=20)
plt.legend(['Age_mod_rand','Age_mod_mean','Age'])
plt.savefig('../../reports/figures/Age_distribution.png')
plt.close()


pivot1 = df_processed.pivot_table(index="Age_bin_mean",values='Survived')
pivot2 = df_processed.pivot_table(index="Age_bin_rand",values='Survived')
pivot_data=pivot1
pivot_data['Age_bin_mean']=pivot1['Survived']
pivot_data['Age_bin_rand']=pivot2['Survived']
pivot_data.index.names = ['Age Category']
pivot_data.drop('Survived',axis=1,inplace=True)
pivot_data = pivot_data.reindex(["Infant","Child","Teenager","Young Adult","Adult","Senior"])
pivot_data.plot.bar(alpha=0.5)
plt.ylabel('% Survived')
plt.savefig('../../reports/figures/Agebin-survived.png')
plt.close()

g5=sns.barplot(data=df_train, x='Embarked', y='Survived', alpha=0.5, color='tab:blue',ci=None)
plt.savefig('../../reports/figures/Embarked-survived.png')
plt.close()

g6=sns.barplot(data=df_train, x='Embarked', y='Pclass', alpha=0.5, color='tab:blue',ci=None)
plt.savefig('../../reports/figures/Embarked-Class.png')
plt.close()
