from pandas import *
import numpy as np
import string
import pandas as pd
from os import path
import matplotlib.pyplot as plt
plt.show(block=True)



df = pd.read_csv("train.tsv", sep="\t")

#CREATION OF HISTOGRAMS FOR CATEGORIAL ATTRIBUTES(2,5,8,11,13,16)
df1 = df[["Attribute2"]]
df2 = df1.loc[df['Label'] == 1]
df3 = df1.loc[df['Label'] == 2]
plt.figure
df2.Attribute2.value_counts().plot(kind='box')
plt.title('Attribute2/Good')
plt.xlabel('Good')
plt.ylabel('Attribute2')
plt.show()
plt.figure
df3.Attribute2.value_counts().plot(kind='box')
plt.title('Attribute2/Bad')
plt.xlabel('Bad')
plt.ylabel('Attribute2')
plt.show()

df1 = df[["Attribute5"]]
df2 = df1.loc[df['Label'] == 1]
df3 = df1.loc[df['Label'] == 2]
plt.figure
df2.Attribute5.value_counts().plot(kind='box')
plt.title('Attribute5/Good')
plt.xlabel('Good')
plt.ylabel('Attribute5')
plt.show()
plt.figure
df3.Attribute5.value_counts().plot(kind='box')
plt.title('Attribute5/Bad')
plt.xlabel('Bad')
plt.ylabel('Attribute5')
plt.show()

df1 = df[["Attribute8"]]
df2 = df1.loc[df['Label'] == 1]
df3 = df1.loc[df['Label'] == 2]
plt.figure
df2.Attribute8.value_counts().plot(kind='box')
plt.title('Attribute8/Good')
plt.xlabel('Good')
plt.ylabel('Attribute8')
plt.show()
plt.figure
df3.Attribute8.value_counts().plot(kind='box')
plt.title('Attribute8/Bad')
plt.xlabel('Bad')
plt.ylabel('Attribute8')
plt.show()

df1 = df[["Attribute11"]]
df2 = df1.loc[df['Label'] == 1]
df3 = df1.loc[df['Label'] == 2]
plt.figure
df2.Attribute11.value_counts().plot(kind='box')
plt.title('Attribute11/Good')
plt.xlabel('Good')
plt.ylabel('Attribute11')
plt.show()
plt.figure
df3.Attribute11.value_counts().plot(kind='box')
plt.title('Attribute11/Bad')
plt.xlabel('Bad')
plt.ylabel('Attribute11')
plt.show()

df1 = df[["Attribute13"]]
df2 = df1.loc[df['Label'] == 1]
df3 = df1.loc[df['Label'] == 2]
plt.figure
df2.Attribute13.value_counts().plot(kind='box')
plt.title('Attribute13/Good')
plt.xlabel('Good')
plt.ylabel('Attribute13')
plt.show()
plt.figure
df3.Attribute13.value_counts().plot(kind='box')
plt.title('Attribute13/Bad')
plt.xlabel('Bad')
plt.ylabel('Attribute13')
plt.show()

df1 = df[["Attribute16"]]
df2 = df1.loc[df['Label'] == 1]
df3 = df1.loc[df['Label'] == 2]
plt.figure
df2.Attribute16.value_counts().plot(kind='box')
plt.title('Attribute16/Good')
plt.xlabel('Good')
plt.ylabel('Attribute16')
plt.show()
plt.figure
df3.Attribute16.value_counts().plot(kind='box')
plt.title('Attribute16/Bad')
plt.xlabel('Bad')
plt.ylabel('Attribute16')
plt.show()
