from __future__ import division
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from pandas import *
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from scipy import spatial, interp
from collections import Counter
import re
import nltk
import math
import sys


from sklearn.metrics  import *

import string
import pandas as pd
from os import path
import matplotlib.pyplot as plt

test_size=0.25
k_fold=10
naive_bayes_a=0.05
random_forests_estimators=100


# The classification function uses the pipeline in order to ease the procedure
# and also makes use of the GridSearchCV for the cross validation, without any tuned
# parameters, which makes it quicker
def classification(clfname,classifier):
	print('-' * 60)
	print("Training %s" % clfname)
	print
	print(classifier)

	pipeline = Pipeline([
		('clf', classifier)
	])


	grid_search = GridSearchCV(pipeline, {}, cv=k_fold,n_jobs=-1)
	grid_search.fit(X_train,y_train)
	print
	print('*' * 60)
	predicted=grid_search.predict(X_test)
	y_proba = grid_search.best_estimator_.predict_proba(X_test)
	
	accuracy = metrics.accuracy_score(y_test, predicted)

	return accuracy,y_proba

# predict_category: trains the whole dataset and makes predictions for the categories
# which are being exported to a csv file
def predict_category(X,y,file_name):
	print("Predict the category with (Multinomial)-Naive Bayes Classifier...")
	X_train = X
	Y_train = y

	df_test = pd.read_csv("test.tsv", sep="\t")
	X_true_id = df["Id"]

	df_test = pd.get_dummies(df_test)
	clf=MultinomialNB(alpha=naive_bayes_a)

	pipeline = Pipeline([
		('clf', clf)
	])
	#Simple Pipeline Fit
	pipeline.fit(X_train,Y_train)
	#Predict the train set
	predicted=pipeline.predict(X_train)
	# create lists to append the id from the test set
	# and the results from the prediction
	ID = []
	category = []
	for i in X_true_id:
		ID.append(i)
	id_dic = {'Client_ID' : ID}

	for pred in predicted:
		category.append(le.inverse_transform(pred))
	category_dic = {'Predicted_Label' : category}
	#finally append them to a dictionary for export
	out_dic = {}
	out_dic.update(id_dic)
	out_dic.update(category_dic)
	# Append the result to the csv
	print("Exporting predicted category to csv")
	outcsv = pd.DataFrame.from_dict(out_dic)
	outcsv.to_csv("testSet_categories.csv",sep='\t')

#Calculates Entropy for the whole data set
def dataEntropy(data):

	df_dict = data.to_dict()
	dataEntropy = 0.0
	
	for i in data:
		tout=Counter(df_dict[i].values()).most_common()
		for freq in tout:
			temp1  = freq[1]
			dataEntropy += (temp1/len(data)) * math.log(temp1/len(data) ,2)
	#print(dataEntropy)

	return dataEntropy

#Calculates Entropy for specific Attribute
def attributeEntropy(data, Attr):

	df_dict = data.to_dict()
	dataEntropy = 0.0
	tout=Counter(df_dict[Attr].values()).most_common()
	
	for freq in tout:
		temp1  = freq[1]
		dataEntropy += (temp1/len(data)) * math.log(temp1/len(data) ,2)
	#print(dataEntropy)

	return dataEntropy

#Calculates Information Gain of an Attribute
def myInfoGain(data, Attr):

	iG = 0.0
	parentEntropy = dataEntropy(data)
	childEntropy = attributeEntropy(data, Attr)
	iG = parentEntropy - childEntropy
	return iG


################################################################################
###################### Here starts the main of the program #####################
if __name__ == "__main__":

	print("Starting Classification Program")
	print ("#"*60)
	df = pd.read_csv("train.tsv", sep="\t")

	
	X=pd.get_dummies(df)
	le=preprocessing.LabelEncoder()
	le.fit(df["Label"])
	y=le.transform(df["Label"])

	# make a prediction for the category
	predict_category(X,y,df)
	X=X.drop("Label",1)
	# split the train set (75 - 25) in order to have a small test set to check the classifiers
	print("#"*60)
	print("Splitting the train set and doing some preprocessing...")
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=test_size, random_state=0)


	# initiate the array, which will hold all the results for the csv
	validation_results = {"Accuracy": {} }

	print("*"*60)
	print("Classification")
	

	# list of tuples for the classifiers
	# the tuple contains (classifier, name of the method)
	classifiers_list = [(BernoulliNB(alpha=naive_bayes_a),"(Binomial)-Naive Bayes","b"),
			(SVC(probability=True), "SVM","y"),
			(RandomForestClassifier(n_estimators=random_forests_estimators,n_jobs=-1), "Random forest","g")]

	#Loop through the classifiers list.
	for clf, clfname, color in classifiers_list:
			print('=' * 60)
			print(clfname)
			accuracy_res, y_probas = classification(clfname,clf)
			validation_results["Accuracy"][clfname] = accuracy_res

	valid_res = pd.DataFrame.from_dict(validation_results)
	valid_res.to_csv("EvaluationMetric_10fold.csv",sep='\t')

	validation_results2 = []
	myres= []
	#Appends attributes and IG tuples in a list
	attrIG = {"Attribute": {}, "InformationGain": {} }
	infogain = []
	temp = ("Attribute1",myInfoGain(df, "Attribute1"))
	infogain.append( temp)
	temp = ("Attribute2",myInfoGain(df, "Attribute2"))
	infogain.append( temp)
	temp = ("Attribute3",myInfoGain(df, "Attribute3"))
	infogain.append( temp)
	temp = ("Attribute4",myInfoGain(df, "Attribute4"))
	infogain.append( temp)
	temp = ("Attribute5",myInfoGain(df, "Attribute5"))
	infogain.append( temp)
	temp = ("Attribute6",myInfoGain(df, "Attribute6"))
	infogain.append( temp)
	temp = ("Attribute7",myInfoGain(df, "Attribute7"))
	infogain.append( temp)
	temp = ("Attribute8",myInfoGain(df, "Attribute8"))
	infogain.append( temp)
	temp = ("Attribute9",myInfoGain(df, "Attribute9"))
	infogain.append( temp)
	temp = ("Attribute10",myInfoGain(df, "Attribute10"))
	infogain.append( temp)
	temp = ("Attribute11",myInfoGain(df, "Attribute11"))
	infogain.append( temp)
	temp = ("Attribute12",myInfoGain(df, "Attribute12"))
	infogain.append( temp)
	temp = ("Attribute13",myInfoGain(df, "Attribute13"))
	infogain.append( temp)
	temp = ("Attribute14",myInfoGain(df, "Attribute14"))
	infogain.append( temp)
	temp = ("Attribute15",myInfoGain(df, "Attribute15"))
	infogain.append( temp)
	temp = ("Attribute16",myInfoGain(df, "Attribute16"))
	infogain.append( temp)
	temp = ("Attribute17",myInfoGain(df, "Attribute17"))
	infogain.append( temp)
	temp = ("Attribute18",myInfoGain(df, "Attribute18"))
	infogain.append( temp)
	temp = ("Attribute19",myInfoGain(df, "Attribute19"))
	infogain.append( temp)
	temp = ("Attribute20",myInfoGain(df, "Attribute20"))
	infogain.append( temp)
	for items in infogain:
	#finds minimum IG of all elemnts and removes it from the list
		minVal = infogain[0][1]
		i=0
		for item in infogain:
			if minVal > infogain[i][1]:
				minVal = infogain[i][1]
				minAttr = infogain [i][0]
			i=i+1
		infogain.remove((minAttr,minVal))
		#print infogain
		print minVal
		print minAttr
		myres.append((minAttr,minVal))
		attrIG["Attribute"][i] = minAttr
		attrIG["InformationGain"][i] = minVal

		df = pd.read_csv("train.tsv", sep="\t")
		df = df.drop(minAttr,1)
		X=pd.get_dummies(df)
		le=preprocessing.LabelEncoder()
		le.fit(df["Label"])
		y=le.transform(df["Label"])
		X=X.drop("Label",1)
	
		# split the train set (75 - 25) in order to have a small test set to check the classifiers
		print("#"*60)
		X_train, X_test, y_train, y_test = train_test_split(
			X, y, test_size=test_size, random_state=0)
		#calls Random forest
		classifiers_list2 = [(RandomForestClassifier(n_estimators=random_forests_estimators,n_jobs=-1), "Random forest","g")]
		for clf, clfname, color in classifiers_list2:
				accuracy_res, y_probas = classification(clfname,clf)
	
		validation_results2.append(accuracy_res)
	
	
	for vals in validation_results2:
		print vals
	#Creates csv with attributes removed along with their IG
	attrIG_res = pd.DataFrame.from_dict(attrIG)
	attrIG_res.to_csv("AttributesInfoGain.csv",sep='\t')
	#Helping list with attributes and their IG
	import csv
	with open('res.csv', 'wb') as myfile:
    		wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    		wr.writerow(myres)
	#Accuracy plot
	plt.plot(validation_results2)
	plt.show()

	


