'''
Name   : Ashwin Sai C
Course : ML - CS6375-003
Title  : Mini Project 2
Term   : Fall 2023
'''

import os
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from prettytable import PrettyTable

dataset_path = 'project2_data\\all_data'

def process_dataset(action_item):
	output_list = []	
	for path, subdirs, files in os.walk(dataset_path):
		if files != []:
			temp_row = []
			# print("\n",path,files)
			
			if action_item == "Tune":	
				print("Tuning..")							
				X, Y, X_test, Y_test, file_name = get_Tune_Datasets(path,files)	
				print("File : ",file_name)
				temp_row.append(file_name[2])		
			
			if action_item == "Merge":
				print("Testing..")
				X, Y, X_test, Y_test, file_name = MERGE_TRAIN_VALIDATE_DATASETS(path,files)
				print("File : ",file_name)
				temp_row.append(file_name[0])

			temp_row.append(TRAIN_DecisionTreeClassifier(X, Y, X_test, Y_test,file_name[2]))
			temp_row.append(TRAIN_BaggingClassifier(X, Y, X_test, Y_test, file_name[2]))
			temp_row.append(TRAIN_RandomForestClassifier(X, Y, X_test, Y_test, file_name[2]))
			temp_row.append(TRAIN_GradientBoostingClassifier(X, Y, X_test, Y_test, file_name[2]))
			print("\n")

			output_list.append(temp_row)

	table_format_accuracy_f1score(output_list)

def get_tune_parameters_DecisionTree():
		tune_parameters_dict = {
									"valid_c300_d100.csv": 
									{
									'splitter': 'best', 'random_state': 42, 'max_leaf_nodes': 100, 'max_features': None, 'max_depth': None, 'criterion': 'entropy'
									},									
									"valid_c300_d1000.csv": 
									{
									'criterion': 'entropy', 'max_depth': None, 'max_features': 200, 'max_leaf_nodes': 100, 'random_state': 42, 'splitter': 'random'
									},
									"valid_c300_d5000.csv": 
									{
									'criterion': 'gini', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': 100, 'random_state': 42, 'splitter': 'random'
									},
									"valid_c500_d100.csv": 
									{
									'criterion': 'entropy', 'max_depth': 5, 'max_features': 10, 'max_leaf_nodes': None, 'random_state': 42, 'splitter': 'best'
									},
									"valid_c500_d1000.csv": 
									{
									'criterion': 'gini', 'max_depth': None, 'max_features': 200, 'max_leaf_nodes': 100, 'random_state': 42, 'splitter': 'random'
									},
									"valid_c500_d5000.csv": 
									{
									'splitter': 'best', 'random_state': 42, 'max_leaf_nodes': 1000, 'max_features': 1000, 'max_depth': 10, 'criterion': 'entropy'
									},
									"valid_c1000_d100.csv": 
									{
									'criterion': 'entropy', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': 10, 'random_state': 42, 'splitter': 'best'
									},
									"valid_c1000_d1000.csv": 
									{
									'criterion': 'entropy', 'max_depth': 5, 'max_features': 200, 'max_leaf_nodes': None, 'random_state': 42, 'splitter': 'best'
									},
									"valid_c1000_d5000.csv": 
									{
									'criterion': 'gini', 'max_depth': 10, 'max_features': None, 'max_leaf_nodes': 100, 'random_state': 42, 'splitter': 'random'
									},
									"valid_c1500_d100.csv": 
									{
									'criterion': 'gini', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': None, 'random_state': 42, 'splitter': 'random'
									},
									"valid_c1500_d1000.csv": 
									{
									'criterion': 'entropy', 'max_depth': None, 'max_features': 200, 'max_leaf_nodes': None, 'random_state': 42, 'splitter': 'random'
									},
									"valid_c1500_d5000.csv": 
									{
									'criterion': 'gini', 'max_depth': 10, 'max_features': None, 'max_leaf_nodes': 100, 'random_state': 42, 'splitter': 'best'
									},
									"valid_c1800_d100.csv": 
									 {
									 'criterion': 'entropy', 'max_depth': None, 'max_features': 10, 'max_leaf_nodes': 100, 'random_state': 42, 'splitter': 'random'
									 },
									"valid_c1800_d1000.csv": 
									{
									'criterion': 'entropy', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': 100, 'random_state': 42, 'splitter': 'best'
									},
									"valid_c1800_d5000.csv": 									
									{
									'criterion': 'gini', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': None, 'random_state': 42, 'splitter': 'best'
									},
									"MNIST_Dataset":
									{
									"criterion" :'gini', "splitter" :'best', "max_depth" : 20, "max_features" : None, "random_state": 42, "max_leaf_nodes" : None 
									}	
							   }									   
		return tune_parameters_dict

def get_tune_parameters_Bagging():
		tune_parameters_dict = {
									"valid_c300_d100.csv": 
									{
										'bootstrap': True, 'bootstrap_features': False, 'estimator': DecisionTreeClassifier(), 'max_features': 0.5, 'max_samples': 1.0, 'n_estimators': 100, 'n_jobs': -1, 'random_state': 42, 'warm_start': True	
									},
									"valid_c300_d1000.csv": 
									{
										'bootstrap': False, 'bootstrap_features': True, 'estimator': DecisionTreeClassifier(), 'max_features': 0.2, 'max_samples': 1.0, 'n_estimators': 100, 'n_jobs': -1, 'random_state': 42, 'warm_start': True	
									},
									"valid_c300_d5000.csv": 
									{
										'bootstrap': False, 'bootstrap_features': False, 'estimator': DecisionTreeClassifier(), 'max_features': 0.5, 'max_samples': 1.0, 'n_estimators': 100, 'n_jobs': -1, 'random_state': 42, 'warm_start': True
									},
									"valid_c500_d100.csv": 
									{
										'bootstrap': False, 'bootstrap_features': False, 'estimator': DecisionTreeClassifier(), 'max_features': 0.2, 'max_samples': 0.5, 'n_estimators': 100, 'n_jobs': -1, 'random_state': 42, 'warm_start': True
									},
									"valid_c500_d1000.csv": 
									{
										'bootstrap': True, 'bootstrap_features': True, 'estimator': DecisionTreeClassifier(), 'max_features': 0.5, 'max_samples': 1.0, 'n_estimators': 100, 'n_jobs': -1, 'random_state': 42, 'warm_start': True
									},
									"valid_c500_d5000.csv": 
									{
										'bootstrap': False, 'bootstrap_features': False, 'estimator': DecisionTreeClassifier(), 'max_features': 0.2, 'max_samples': 1.0, 'n_estimators': 100, 'n_jobs': -1, 'random_state': 42, 'warm_start': True
									},
									"valid_c1000_d100.csv": 
									{
										'bootstrap': False, 'bootstrap_features': True, 'estimator': DecisionTreeClassifier(), 'max_features': 0.1, 'max_samples': 1.0, 'n_estimators': 100, 'n_jobs': -1, 'random_state': 42, 'warm_start': True
									},
									"valid_c1000_d1000.csv": 
									{
										'bootstrap': False, 'bootstrap_features': True, 'estimator': DecisionTreeClassifier(), 'max_features': 0.1, 'max_samples': 1.0, 'n_estimators': 100, 'n_jobs': -1, 'random_state': 42, 'warm_start': True
									},
									"valid_c1000_d5000.csv": 
									{
										'bootstrap': False, 'bootstrap_features': False, 'estimator': DecisionTreeClassifier(), 'max_features': 0.1, 'max_samples': 1.0, 'n_estimators': 100, 'n_jobs': -1, 'random_state': 42, 'warm_start': True
									},
									"valid_c1500_d100.csv": 
									{
										'bootstrap': True, 'bootstrap_features': True, 'estimator': DecisionTreeClassifier(), 'max_features': 0.1, 'max_samples': 0.5, 'n_estimators': 100, 'n_jobs': -1, 'random_state': 42, 'warm_start': True
									},
									"valid_c1500_d1000.csv": 
									{
										'bootstrap': True, 'bootstrap_features': True, 'estimator': DecisionTreeClassifier(), 'max_features': 0.1, 'max_samples': 0.2, 'n_estimators': 100, 'n_jobs': -1, 'random_state': 42, 'warm_start': True
									},
									"valid_c1500_d5000.csv": 
									{
										'bootstrap': True, 'bootstrap_features': True, 'estimator': DecisionTreeClassifier(), 'max_features': 0.1, 'max_samples': 0.5, 'n_estimators': 100, 'n_jobs': -1, 'random_state': 42, 'warm_start': True
									},
									"valid_c1800_d100.csv": 
									{
										'bootstrap': True, 'bootstrap_features': True, 'estimator': DecisionTreeClassifier(), 'max_features': 0.1, 'max_samples': 0.2, 'n_estimators': 100, 'n_jobs': -1, 'random_state': 42, 'warm_start': True
									},
									"valid_c1800_d1000.csv": 
									{
										'bootstrap': True, 'bootstrap_features': True, 'estimator': DecisionTreeClassifier(), 'max_features': 0.1, 'max_samples': 0.1, 'n_estimators': 100, 'n_jobs': -1, 'random_state': 42, 'warm_start': True
									},
									"valid_c1800_d5000.csv": 
									{
										'bootstrap': True, 'bootstrap_features': True, 'estimator': DecisionTreeClassifier(), 'max_features': 0.1, 'max_samples': 0.1, 'n_estimators': 100, 'n_jobs': -1, 'random_state': 42, 'warm_start': True
									},
									"MNIST_Dataset":
									{
										'estimator'                : DecisionTreeClassifier(),
										"n_estimators"             : 10, 
										"max_samples"              : 1.0, 
										"max_features"             : 1.0, 
										"bootstrap"                : True,
										"bootstrap_features"       : True, 
										"oob_score"                : False, 
										"warm_start"               : False, 
										"n_jobs"                   : 1, 
										"random_state"             : 42
									}	
							   }									   
		return tune_parameters_dict

def get_tune_parameters_RandomForest():
		tune_parameters_dict = {
									"valid_c300_d100.csv": 
									{
										"n_estimators"             : 100, 
										"criterion"                : "gini", 
										"max_depth"                : None, 
										"max_features"             : "sqrt", 
										"max_leaf_nodes"           : None, 
										"bootstrap"                : False,
										"oob_score"                : False, 
										"n_jobs"                   : 1, 
										"random_state"             : 42, 
										"verbose"                  : False, 
										"warm_start"               : False, 
										"class_weight"             : None, 
										"max_samples"              : None		
									},
									"valid_c300_d1000.csv": 
									{
										'bootstrap': True, 'criterion': 'entropy', 'max_depth': 10, 'max_features': None, 'max_leaf_nodes': 100, 'max_samples': None, 'n_estimators': 100, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0
									},
									"valid_c300_d5000.csv": 
									{
										'bootstrap': True, 'criterion': 'gini', 'max_depth': 100, 'max_features': 'sqrt', 'max_leaf_nodes': 100, 'max_samples': None, 'n_estimators': 100, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0
									},
									"valid_c500_d100.csv": 
									{
										'bootstrap': True, 'criterion': 'gini', 'max_depth': 10, 'max_features': 'sqrt', 'max_leaf_nodes': 20, 'max_samples': None, 'n_estimators': 100, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0
									},
									"valid_c500_d1000.csv": 
									{
										'bootstrap': True, 'criterion': 'gini', 'max_depth': 10, 'max_features': 'log2', 'max_leaf_nodes': 20, 'max_samples': None, 'n_estimators': 100, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0
									},
									"valid_c500_d5000.csv": 
									{
										'bootstrap': True, 'criterion': 'gini', 'max_depth': 10, 'max_features': 'log2', 'max_leaf_nodes': 100, 'max_samples': None, 'n_estimators': 100, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0
									},
									"valid_c1000_d100.csv": 
									{
										'bootstrap': True, 'criterion': 'entropy', 'max_depth': 10, 'max_features': 'sqrt', 'max_leaf_nodes': 10, 'max_samples': None, 'n_estimators': 100, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0
									},
									"valid_c1000_d1000.csv": 
									{
										'bootstrap': True, 'criterion': 'gini', 'max_depth': 10, 'max_features': 'log2', 'max_leaf_nodes': 100, 'max_samples': None, 'n_estimators': 100, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0
									},
									"valid_c1000_d5000.csv": 
									{
										'bootstrap': True, 'criterion': 'entropy', 'max_depth': 10, 'max_features': 'log2', 'max_leaf_nodes': 100, 'max_samples': None, 'n_estimators': 100, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0
									},
									"valid_c1500_d100.csv": 
									{
										'bootstrap': True, 'criterion': 'gini', 'max_depth': 10, 'max_features': 'log2', 'max_leaf_nodes': 10, 'max_samples': None, 'n_estimators': 100, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0
									},
									"valid_c1500_d1000.csv": 
									{
										'bootstrap': True, 'criterion': 'gini', 'max_depth': 10, 'max_features': 'log2', 'max_leaf_nodes': 10, 'max_samples': None, 'n_estimators': 100, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0
									},
									"valid_c1500_d5000.csv": 
									{
										'bootstrap': True, 'criterion': 'gini', 'max_depth': 10, 'max_features': 'log2', 'max_leaf_nodes': None, 'max_samples': None, 'n_estimators': 100, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0
									},
									"valid_c1800_d100.csv": 
									{
										'bootstrap': True, 'criterion': 'gini', 'max_depth': 10, 'max_features': 'sqrt', 'max_leaf_nodes': 10, 'max_samples': None, 'n_estimators': 100, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0
									},
									"valid_c1800_d1000.csv": 
									{
										'bootstrap': True, 'criterion': 'gini', 'max_depth': 10, 'max_features': 'sqrt', 'max_leaf_nodes': 10, 'max_samples': None, 'n_estimators': 100, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0
									},
									"valid_c1800_d5000.csv": 
									{
										'bootstrap': True, 'criterion': 'gini', 'max_depth': 10, 'max_features': 'sqrt', 'max_leaf_nodes': 1000, 'max_samples': None, 'n_estimators': 100, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0
									},
									"MNIST_Dataset":
									{
										"n_estimators"             : 100, 
										"criterion"                : "gini", 
										"max_depth"                : None, 
										"max_features"             : "sqrt", 
										"max_leaf_nodes"           : None, 
										"bootstrap"                : True,
										"oob_score"                : False, 
										"n_jobs"                   : 1, 
										"random_state"             : 42, 
										"verbose"                  : False, 
										"warm_start"               : False, 
										"class_weight"             : None, 
										"max_samples"              : None		
									}	
							   }									   
		return tune_parameters_dict

def get_tune_parameters_GradientBoosting():
		tune_parameters_dict = {
									"valid_c300_d100.csv": 
									{
										'verbose': 0, 'subsample': 0.2, 'random_state': 42, 'n_estimators': 100, 'max_leaf_nodes': None, 'max_features': None, 'max_depth': 1000, 'loss': 'log_loss', 'learning_rate': 0.1, 'init': None, 'criterion': 'friedman_mse'
									},
									"valid_c300_d1000.csv": 
									{
										'warm_start': False, 'verbose': 0, 'subsample': 0.4, 'random_state': 42, 'n_estimators': 100, 'max_leaf_nodes': 100, 'max_features': None, 'max_depth': 1000, 'loss': 'exponential', 'learning_rate': 0.1, 'init': 'zero', 'criterion': 'friedman_mse'
									},
									"valid_c300_d5000.csv": 
									{
										'verbose': 0, 'subsample': 1.0, 'random_state': 42, 'n_estimators': 100, 'max_leaf_nodes': 10, 'max_features': 'log2', 'max_depth': 100, 'loss': 'exponential', 'learning_rate': 0.001, 'init': None, 'criterion': 'friedman_mse'
									},
									"valid_c500_d100.csv": 
									{
										'verbose': 0, 'subsample': 0.2, 'random_state': 42, 'n_estimators': 100, 'max_leaf_nodes': 10, 'max_features': None, 'max_depth': 100, 'loss': 'exponential', 'learning_rate': 0.001, 'init': 'zero', 'criterion': 'friedman_mse'										
									},
									"valid_c500_d1000.csv": 
									{
										'warm_start': False, 'verbose': 0, 'subsample': 0.4, 'random_state': 42, 'n_estimators': 100, 'max_leaf_nodes': 100, 'max_features': None, 'max_depth': 1000, 'loss': 'exponential', 'learning_rate': 0.1, 'init': 'zero', 'criterion': 'friedman_mse'
									},
									"valid_c500_d5000.csv": 
									{
										'warm_start': False, 'verbose': 0, 'subsample': 0.4, 'random_state': 42, 'n_estimators': 100, 'max_leaf_nodes': 100, 'max_features': None, 'max_depth': 1000, 'loss': 'exponential', 'learning_rate': 0.1, 'init': 'zero', 'criterion': 'friedman_mse'
									},
									"valid_c1000_d100.csv": 
									{
										'verbose': 0, 'subsample': 0.4, 'random_state': 42, 'n_estimators': 100, 'max_leaf_nodes': 100, 'max_features': None, 'max_depth': 1000, 'loss': 'log_loss', 'learning_rate': 0.1, 'init': None, 'criterion': 'squared_error'
									},
									"valid_c1000_d1000.csv": 
									{
										'verbose': 0, 'subsample': 0.7, 'random_state': 42, 'n_estimators': 10, 'max_leaf_nodes': 100, 'max_features': 'sqrt', 'max_depth': 100, 'loss': 'log_loss', 'learning_rate': 0.1, 'init': 'zero', 'criterion': 'friedman_mse'
									},
									"valid_c1000_d5000.csv": 
									{
										'verbose': 0, 'subsample': 0.2, 'random_state': 42, 'n_estimators': 100, 'max_leaf_nodes': 100, 'max_features': 'sqrt', 'max_depth': None, 'loss': 'log_loss', 'learning_rate': 0.1, 'init': 'zero', 'criterion': 'friedman_mse'
									},
									"valid_c1500_d100.csv": 
									{
										'criterion': 'squared_error', 'learning_rate': 0.1, 'loss': 'log_loss', 'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': 10, 'n_estimators': 100, 'random_state': 42, 'subsample': 1.0
									},
									"valid_c1500_d1000.csv": 
									{
										'criterion': 'squared_error', 'learning_rate': 0.1, 'loss': 'log_loss', 'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': 10, 'n_estimators': 100, 'random_state': 42, 'subsample': 0.1
									},
									"valid_c1500_d5000.csv": 
									{
										'criterion': 'squared_error', 'learning_rate': 0.1, 'loss': 'log_loss', 'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'n_estimators': 100, 'random_state': 42, 'subsample': 0.1
									},
									"valid_c1800_d100.csv": 
									{
										'criterion': 'squared_error', 'learning_rate': 0.1, 'loss': 'log_loss', 'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': 10, 'n_estimators': 100, 'random_state': 42, 'subsample': 0.2
									},
									"valid_c1800_d1000.csv": 
									{										
										'criterion': 'squared_error', 'learning_rate': 0.1, 'loss': 'log_loss', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': 10, 'n_estimators': 100, 'random_state': 42, 'subsample': 0.1
									},
									"valid_c1800_d5000.csv": 
									{										
										'warm_start': False, 'verbose': 0, 'subsample': 1.0, 'random_state': 42, 'n_estimators': 10, 'max_leaf_nodes': None, 'max_features': 'sqrt', 'max_depth': 250, 'loss': 'log_loss', 'learning_rate': 0.1, 'init': None, 'criterion': 'friedman_mse'								
									},
									"MNIST_Dataset":
									{
										"loss"                     : "log_loss", 
										"learning_rate"            : 0.4, 
										"n_estimators"             : 100, 
										"subsample"                : 1.0,
										"criterion"                : "friedman_mse", 
										"max_depth"                : 10, 
										"init"                     : None, 
										"random_state"             : 42, 
										"max_features"             : "sqrt", 
										"verbose"                  : 0,									
										"max_leaf_nodes"           : None, 
										"warm_start"               : False
									}	
							   }									   
		return tune_parameters_dict

def get_Tune_Datasets(path,file_name):

	#Validate Datasets for Tuning the parameters
	##filename[0] - test.csv
	##filename[1] - train.csv
	##filename[2] - valid.csv

	train_dataset                    = pd.read_csv(path+"\\"+file_name[1], header=None)
	val_dataset                      = pd.read_csv(path+"\\"+file_name[2], header=None)
	X_train                          = train_dataset.iloc[:, :-1]  # Features
	Y_train                          = train_dataset.iloc[:, -1]   # Class labels
	X_val                            = val_dataset.iloc[:, :-1]    # Features
	Y_val                            = val_dataset.iloc[:, -1]     # Class labels

	return X_train, Y_train, X_val, Y_val, file_name

def TRAIN_DecisionTreeClassifier(X_Train, Y_Train, X_Test, Y_Test, file_name):	

	parameters = get_tune_parameters_DecisionTree()[file_name]

	# #Uncomment for Tune sets
	# param_grid = {
	# 				"criterion"                : ["gini", "entropy", "log_loss"], 
	# 				"splitter"                 : ['best',"random"], 
	# 				"max_depth"                : [None, 5, 10,27,100,1000], 
	# 				"max_features"             : [None,10,200,1000], 
	# 				"random_state"             : [42], 
	# 				"max_leaf_nodes"           : [None, 10,100, 200, 1000] 
	# 			}
	# clf = DecisionTreeClassifier()

	# _search = GridSearchCV(clf, param_grid, cv=6, n_jobs=os.cpu_count()-1)
	# _search.fit(X_Train, Y_Train)
	# print(f"Best hyperparameters found by SearchCV: {_search.best_params_}")
	# parameters = _search.best_params_

	clf = DecisionTreeClassifier(									
									criterion                = parameters["criterion"], 
									splitter                 = parameters["splitter"], 
									max_depth                = parameters["max_depth"], 
									max_features             = parameters["max_features"], 
									random_state             = parameters["random_state"], 
									max_leaf_nodes           = parameters["max_leaf_nodes"], 
								)
	clf.fit(X_Train,Y_Train)
	Y_Pred   = clf.predict(X_Test)

	accuracy = accuracy_score(Y_Test, Y_Pred)
	try:	
		f1score  = f1_score(Y_Pred, Y_Test)
	except Exception as e:
		# print(e, "MNIST_Dataset")
		f1score = 0
	print("TRAIN_DecisionTreeClassifier     -> Accuracy : ",round(accuracy*100,2), "F1-Score : ",round(f1score,2))

	return [round(accuracy*100,2), round(f1score,2)]

def TRAIN_BaggingClassifier(X_Train, Y_Train, X_Test, Y_Test, file_name):	

	parameters = get_tune_parameters_Bagging()[file_name]
	
	##Uncomment for Tuning Datasets
	# param_grid = {
	# 				"estimator"                : [DecisionTreeClassifier()],
	# 				"n_estimators"             : [100], 
	# 				"max_samples"              : [0.1,0.2,0.5,1.0], 
	# 				"max_features"             : [0.1,0.2,0.5,1.0], 
	# 				"bootstrap"                : [True,False],
	# 				"bootstrap_features"       : [True,False], 
	# 				"warm_start"               : [True,False], 
	# 				"n_jobs"                   : [-1], 
	# 				"random_state"             : [42]
	# 			}
	# clf = BaggingClassifier()

	# _search = GridSearchCV(clf, param_grid, cv=2, n_jobs=os.cpu_count()-1)
	# _search.fit(X_Train, Y_Train)
	# print(f"Best hyperparameters found by SearchCV: {_search.best_params_}")
	# parameters = _search.best_params_

	clf = BaggingClassifier(		
									estimator                   = DecisionTreeClassifier(),						
									n_estimators                = parameters["n_estimators"], 
									max_samples                 = parameters["max_samples"], 
									max_features                = parameters["max_features"], 
									bootstrap                   = parameters["bootstrap"], 
									bootstrap_features          = parameters["bootstrap_features"],
									n_jobs                      = parameters["n_jobs"], 
									random_state                = parameters["random_state"]
								)
	clf.fit(X_Train,Y_Train)
	Y_Pred   = clf.predict(X_Test)

	accuracy = accuracy_score(Y_Test, Y_Pred)
	try:
		f1score  = f1_score(Y_Pred, Y_Test)
	except Exception as e:
		# print(e,"MNIST_Dataset F1-Score")
		f1score = 0
	print("TRAIN_BaggingClassifier          -> Accuracy : ",round(accuracy*100,2), "F1-Score : ",round(f1score,2))

	return [round(accuracy*100,2), round(f1score,2)]

def TRAIN_RandomForestClassifier(X_Train, Y_Train, X_Test, Y_Test, file_name):	

	parameters = get_tune_parameters_RandomForest()[file_name]

	##Uncomment for Tuning Datasets
	# param_grid = {
	# 				"n_estimators"                : [10,100],
	# 				"criterion"                   : ["gini", "entropy", "log_loss"], 
	# 				"max_depth"                   : [10,100,1000, None], 
	# 				"max_features"                : ["sqrt", "log2", None], 
	# 				"max_leaf_nodes"              : [10,20,100,1000, None], 
	# 				"bootstrap"                   : [True], 
	# 				"oob_score"                   : [False],
	# 				"n_jobs"                      : [-1],
	# 				"random_state"                : [42], 
	# 				"verbose"                     : [0], 
	# 				"max_samples"                 : [None]
	# 			}

	# clf = RandomForestClassifier()

	# _search = GridSearchCV(clf, param_grid, cv=2, n_jobs=os.cpu_count()-1)
	# _search.fit(X_Train, Y_Train)
	# print(f"Best hyperparameters found by SearchCV: {_search.best_params_}")
	# parameters = _search.best_params_


	clf = RandomForestClassifier(		
									n_estimators                   = parameters["n_estimators"],						
									criterion                      = parameters["criterion"], 
									max_depth                      = parameters["max_depth"], 
									max_features                   = parameters["max_features"], 
									max_leaf_nodes                 = parameters["max_leaf_nodes"], 
									bootstrap                      = parameters["bootstrap"], 
									oob_score                      = parameters["oob_score"],
									n_jobs                         = parameters["n_jobs"],
									random_state                   = parameters["random_state"],
									verbose                        = parameters["verbose"],
									max_samples                    = parameters["max_samples"]
								)
	clf.fit(X_Train,Y_Train)
	Y_Pred   = clf.predict(X_Test)

	accuracy = accuracy_score(Y_Test, Y_Pred)
	try:
		f1score  = f1_score(Y_Pred, Y_Test)
	except Exception as e:
		# print(e, "MNIST_Dataset F1-score")
		f1score = 0

	print("TRAIN_RandomForestClassifier     -> Accuracy : ",round(accuracy*100,2), "F1-Score : ",round(f1score,2))

	return [round(accuracy*100,2), round(f1score,2)]

def TRAIN_GradientBoostingClassifier(X_Train, Y_Train, X_Test, Y_Test, file_name):	

	parameters = get_tune_parameters_GradientBoosting()[file_name]

	## Uncomment for Tuning Datasets
	# param_grid = {
	# 				"loss"                     : ["log_loss", "exponential"], 
	# 				"learning_rate"            : [0.1,0.001,10,100], 
	# 				"n_estimators"             : [100], 
	# 				"subsample"                : [0.1,0.2,1.0],
	# 				"criterion"                : ["squared_error","friedman_mse"], 
	# 				"max_depth"                : [None,10,100,1000], 
	# 				"random_state"             : [42], 
	# 				"max_features"             : [None,"sqrt","log2"], 
	# 				"max_leaf_nodes"           : [10,100,None] 
	# 			  }

	# clf = GradientBoostingClassifier()

	# _search = GridSearchCV(clf, param_grid, cv=2, n_jobs=os.cpu_count()-1)
	# _search.fit(X_Train, Y_Train)
	# print(f"Best hyperparameters found by SearchCV: {_search.best_params_}")
	# parameters = _search.best_params_

	clf = GradientBoostingClassifier(		
									loss                            = parameters["loss"],						
									learning_rate                   = parameters["learning_rate"], 
									n_estimators                    = parameters["n_estimators"], 
									subsample                       = parameters["subsample"], 
									criterion                       = parameters["criterion"], 
									max_depth                       = parameters["max_depth"], 
									random_state                    = parameters["random_state"],
									max_features                    = parameters["max_features"],
									max_leaf_nodes                  = parameters["max_leaf_nodes"],
								)
	clf.fit(X_Train,Y_Train)
	Y_Pred   = clf.predict(X_Test)

	accuracy = accuracy_score(Y_Test, Y_Pred)
	try:
		f1score  = f1_score(Y_Pred, Y_Test)
	except Exception as e:
		# print(e, "MNIST_Dataset F1-score")
		f1score = 0

	print("TRAIN_GradientBoostingClassifier -> Accuracy : ",round(accuracy*100,2), "F1-Score : ",round(f1score,2))

	return [round(accuracy*100,2), round(f1score,2)]

def MERGE_TRAIN_VALIDATE_DATASETS(path,file_name):

	#Merging Train Set and Validate Set for Testing purposes

	test_dataset                     = pd.read_csv(path+"\\"+file_name[0], header=None)
	train_dataset                    = pd.read_csv(path+"\\"+file_name[1], header=None)
	val_dataset                      = pd.read_csv(path+"\\"+file_name[2], header=None)
	X_train                          = train_dataset.iloc[:, :-1]  # Features
	Y_train                          = train_dataset.iloc[:, -1]   # Class labels
	X_val                            = val_dataset.iloc[:, :-1]    # Features
	Y_val                            = val_dataset.iloc[:, -1]     # Class labels
	X_test                           = test_dataset.iloc[:, :-1]  # Features
	Y_test                           = test_dataset.iloc[:, -1]   # Class labels

	# print(X_train.shape)
	# print(Y_train.shape)
	# print(X_val.shape)
	# print(Y_val.shape)

	X = pd.concat([X_train,X_val])
	Y = pd.concat([Y_train, Y_val])

	# print(X.shape)
	# print(Y.shape)


	return X, Y, X_test, Y_test,file_name

def MNIST_Dataset():
	X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser="auto")
	X = X / 255.
	# rescale the data, use the traditional train/test split
	# (60K: Train) and (10K: Test)
	X_train, X_test = X[:60000], X[60000:]
	Y_train, Y_test = y[:60000], y[60000:]

	file_name = ""
	output_list = []
	output_list.append("MNIST_Dataset")
	output_list.append(TRAIN_DecisionTreeClassifier(X_train, Y_train, X_test, Y_test, "MNIST_Dataset")[0])
	output_list.append(TRAIN_BaggingClassifier(X_train, Y_train, X_test, Y_test, "MNIST_Dataset")[0])
	output_list.append(TRAIN_RandomForestClassifier(X_train, Y_train, X_test, Y_test, "MNIST_Dataset")[0])
	output_list.append(TRAIN_GradientBoostingClassifier(X_train, Y_train, X_test, Y_test, "MNIST_Dataset")[0])

	table_format_accuracy_f1score([output_list])

def print_parameters():
	print("\n------Best Parameters Settings Found by Tuning..-------\n")
	table_format_parameters("DecisionTree")
	table_format_parameters("Bagging")
	table_format_parameters("RandomForest")
	table_format_parameters("GradientBoosting")

def table_format_accuracy_f1score(output_list):
	print("\n\n[Accuracy%, F1-Score]")
	t = PrettyTable(['Dataset', 'DecisionTree', 'BaggingClassifier', 'RandomForestClassifier', 'GradientBoosting'])
	
	for row in output_list:
		t.add_row(row)

	print(t)

def table_format_parameters(name):
	if name == "DecisionTree":
		parameters = get_tune_parameters_DecisionTree()
	elif name == "Bagging":
		parameters = get_tune_parameters_Bagging()
	elif name == "RandomForest":
		parameters = get_tune_parameters_RandomForest()
	elif name == "GradientBoosting":
		parameters = get_tune_parameters_GradientBoosting()

	column_names = []
	for key in parameters.keys():
		column_names.append(parameters[key].keys())

	column_names = list(itertools.chain.from_iterable(column_names))
	column_names = list(set(column_names))
	column_names.insert(0,"Dataset")
	
	t = PrettyTable(column_names)
	
	for key in parameters.keys():
		temp = []
		temp.append(key)
		for col in column_names[1:]:
			try:
				temp.append(parameters[key][col])
			except Exception as e:
				temp.append("-")
		t.add_row(temp)

	print("-----",name,"-----")
	print(t)
	print("\n")



if __name__ == "__main__":
	print("hi")

	##Run to check the accuracy on Validate sets
	# process_dataset("Tune")	 #Used for tuning the classifier with different hyperparameters
	
	# ##Run to check the accuracy on Test sets
	process_dataset("Merge")     #Applies the classifier on test Datasets
	MNIST_Dataset()				 #Applies the classifier on MNIST Datasets
	
	# ##Run to print out the best parameters used.
	print_parameters()			 #Prints the classifier parameters for each datasets
