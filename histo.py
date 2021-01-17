from pandas import *
import numpy as np
import string
import pandas as pd
from os import path
import matplotlib.pyplot as plt
plt.show(block=True)



df = pd.read_csv("train.tsv", sep="\t")

#CREATION OF HISTOGRAMS FOR CATEGORIAL ATTRIBUTES(1,3,4,6,7,9,10,12,14,15,17,19,20)

df1 = df[["Attribute1"]]
df2 = df1.loc[df['Label'] == 1]
df3 = df1.loc[df['Label'] == 2]
plt.figure
df2.Attribute1.value_counts().plot(kind='bar')
plt.title('Attribute1/Good')
plt.xlabel('Good')
plt.ylabel('Attribute1')
plt.show()
plt.figure
df3.Attribute1.value_counts().plot(kind='bar')
plt.title('Attribute1/Bad')
plt.xlabel('Bad')
plt.ylabel('Attribute1')
plt.show()

df1 = df[["Attribute3"]]
df2 = df1.loc[df['Label'] == 1]
df3 = df1.loc[df['Label'] == 2]
plt.figure
df2.Attribute3.value_counts().plot(kind='bar')
plt.title('Attribute3/Good')
plt.xlabel('Good')
plt.ylabel('Attribute3')
plt.show()
plt.figure
df3.Attribute3.value_counts().plot(kind='bar')
plt.title('Attribute3/Bad')
plt.xlabel('Bad')
plt.ylabel('Attribute3')
plt.show()

df1 = df[["Attribute4"]]
df2 = df1.loc[df['Label'] == 1]
df3 = df1.loc[df['Label'] == 2]
plt.figure
df2.Attribute4.value_counts().plot(kind='bar')
plt.title('Attribute4/Good')
plt.xlabel('Good')
plt.ylabel('Attribute4')
plt.show()
plt.figure
df3.Attribute4.value_counts().plot(kind='bar')
plt.title('Attribute4/Bad')
plt.xlabel('Bad')
plt.ylabel('Attribute4')
plt.show()

df1 = df[["Attribute6"]]
df2 = df1.loc[df['Label'] == 1]
df3 = df1.loc[df['Label'] == 2]
plt.figure
df2.Attribute6.value_counts().plot(kind='bar')
plt.title('Attribute6/Good')
plt.xlabel('Good')
plt.ylabel('Attribute6')
plt.show()
plt.figure
df3.Attribute6.value_counts().plot(kind='bar')
plt.title('Attribute6/Bad')
plt.xlabel('Bad')
plt.ylabel('Attribute6')
plt.show()

df1 = df[["Attribute7"]]
df2 = df1.loc[df['Label'] == 1]
df3 = df1.loc[df['Label'] == 2]
plt.figure
df2.Attribute7.value_counts().plot(kind='bar')
plt.title('Attribute7/Good')
plt.xlabel('Good')
plt.ylabel('Attribute7')
plt.show()
plt.figure
df3.Attribute7.value_counts().plot(kind='bar')
plt.title('Attribute7/Bad')
plt.xlabel('Bad')
plt.ylabel('Attribute7')
plt.show()

df1 = df[["Attribute9"]]
df2 = df1.loc[df['Label'] == 1]
df3 = df1.loc[df['Label'] == 2]
plt.figure
df2.Attribute9.value_counts().plot(kind='bar')
plt.title('Attribute9/Good')
plt.xlabel('Good')
plt.ylabel('Attribute9')
plt.show()
plt.figure
df3.Attribute9.value_counts().plot(kind='bar')
plt.title('Attribute9/Bad')
plt.xlabel('Bad')
plt.ylabel('Attribute9')
plt.show()

df1 = df[["Attribute10"]]
df2 = df1.loc[df['Label'] == 1]
df3 = df1.loc[df['Label'] == 2]
plt.figure
df2.Attribute10.value_counts().plot(kind='bar')
plt.title('Attribute10/Good')
plt.xlabel('Good')
plt.ylabel('Attribute10')
plt.show()
plt.figure
df3.Attribute10.value_counts().plot(kind='bar')
plt.title('Attribute10/Bad')
plt.xlabel('Bad')
plt.ylabel('Attribute10')
plt.show()

df1 = df[["Attribute12"]]
df2 = df1.loc[df['Label'] == 1]
df3 = df1.loc[df['Label'] == 2]
plt.figure
df2.Attribute12.value_counts().plot(kind='bar')
plt.title('Attribute12/Good')
plt.xlabel('Good')
plt.ylabel('Attribute12')
plt.show()
plt.figure
df3.Attribute12.value_counts().plot(kind='bar')
plt.title('Attribute12/Bad')
plt.xlabel('Bad')
plt.ylabel('Attribute12')
plt.show()

df1 = df[["Attribute14"]]
df2 = df1.loc[df['Label'] == 1]
df3 = df1.loc[df['Label'] == 2]
plt.figure
df2.Attribute14.value_counts().plot(kind='bar')
plt.title('Attribute14/Good')
plt.xlabel('Good')
plt.ylabel('Attribute14')
plt.show()
plt.figure
df3.Attribute14.value_counts().plot(kind='bar')
plt.title('Attribute14/Bad')
plt.xlabel('Bad')
plt.ylabel('Attribute14')
plt.show()

df1 = df[["Attribute15"]]
df2 = df1.loc[df['Label'] == 1]
df3 = df1.loc[df['Label'] == 2]
plt.figure
df2.Attribute15.value_counts().plot(kind='bar')
plt.title('Attribute15/Good')
plt.xlabel('Good')
plt.ylabel('Attribute15')
plt.show()
plt.figure
df3.Attribute15.value_counts().plot(kind='bar')
plt.title('Attribute15/Bad')
plt.xlabel('Bad')
plt.ylabel('Attribute15')
plt.show()

df1 = df[["Attribute17"]]
df2 = df1.loc[df['Label'] == 1]
df3 = df1.loc[df['Label'] == 2]
plt.figure
df2.Attribute17.value_counts().plot(kind='bar')
plt.title('Attribute17/Good')
plt.xlabel('Good')
plt.ylabel('Attribute17')
plt.show()
plt.figure
df3.Attribute17.value_counts().plot(kind='bar')
plt.title('Attribute17/Bad')
plt.xlabel('Bad')
plt.ylabel('Attribute17')
plt.show()

df1 = df[["Attribute19"]]
df2 = df1.loc[df['Label'] == 1]
df3 = df1.loc[df['Label'] == 2]
plt.figure
df2.Attribute19.value_counts().plot(kind='bar')
plt.title('Attribute19/Good')
plt.xlabel('Good')
plt.ylabel('Attribute19')
plt.show()
plt.figure
df3.Attribute19.value_counts().plot(kind='bar')
plt.title('Attribute19/Bad')
plt.xlabel('Bad')
plt.ylabel('Attribute19')
plt.show()

df1 = df[["Attribute20"]]
df2 = df1.loc[df['Label'] == 1]
df3 = df1.loc[df['Label'] == 2]
plt.figure
df2.Attribute20.value_counts().plot(kind='bar')
plt.title('Attribute20/Good')
plt.xlabel('Good')
plt.ylabel('Attribute20')
plt.show()
plt.figure
df3.Attribute20.value_counts().plot(kind='bar')
plt.title('Attribute20/Bad')
plt.xlabel('Bad')
plt.ylabel('Attribute20')
plt.show()


