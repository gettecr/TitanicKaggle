

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


trainfile = '../../data/raw/train.csv'

df_train = pd.read_csv(trainfile)


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

binfrom = 'Age'
bintype = 'Age_bin'
df_train.loc[ df_train[binfrom] <= 16, bintype] = 1
df_train.loc[(df_train[binfrom] > 16) & (df_train[binfrom] <= 25), bintype] = 2
df_train.loc[(df_train[binfrom] > 25) & (df_train[binfrom] <= 40), bintype] = 3
df_train.loc[(df_train[binfrom] > 40) & (df_train[binfrom] <= 65), bintype] = 4
df_train.loc[ df_train[binfrom] > 65, bintype]=5

tick_lab=['0-16','16-25','25-40','40-60', '60+']

g4=sns.barplot(data=df_train, x='Age_bin', y='Survived', alpha=0.5, color='tab:blue',ci=None)
g4.set(xticklabels=tick_lab)
plt.savefig('../../reports/figures/Agebin-survived.png')
plt.close()

g5=sns.barplot(data=df_train, x='Embarked', y='Survived', alpha=0.5, color='tab:blue',ci=None)
plt.savefig('../../reports/figures/Embarked-survived.png')
plt.close()

g6=sns.barplot(data=df_train, x='Embarked', y='Pclass', alpha=0.5, color='tab:blue',ci=None)
plt.savefig('../../reports/figures/Embarked-Class.png')
plt.close()
