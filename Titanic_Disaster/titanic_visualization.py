import numpy as np
import pandas as pd
import seaborn as sns
import timeit

#titanik veri setini yüklüyoruz
titanic = sns.load_dataset("titanic")
titanic.info()

query = titanic[(titanic.sex == 'female') & (titanic.age>30) & (titanic.survived == 1)].head()
#print(query)
#sorgu denemesi

#titanik yolcularının yaş dağılımları
sns.distplot(titanic.age.dropna())
#sns.plt.show()

#joinplot
#yaş ve ücret arası ilişki incelemesi
#reg ile verilere uygun bir regresyon istediğimizi belirtiyoruz
#pearson korelasyon katsayısına göre hemen hemen hiçbir ilişki yok 
sns.jointplot(data=titanic, x='age', y='fare', kind='reg')
#sns.plt.show()

#korelasyon matrisi
sns.heatmap(titanic.corr(), annot=True, fmt=".1f")
#sns.plt.show()