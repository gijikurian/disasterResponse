

Table of Contents

	•	Introduction
	•	Installation
	•	Project Motivation
	•	Folder structure
	•	How to run
	•	Results
	•	Licensing, Authors, and Acknowledgements

Introduction

	This project is to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. The
	data set contains real messages that were sent during disaster events. Project will be able to create a machine learning
	pipeline to categorize these events so that the messages can send to an appropriate disaster relief agency. Project will
	include a web app where an emergency worker can input a new message and get classification results in several categories. The
	web app will also display visualizations of the data. 
	
	In ETL pipeline, we will read the dataset, clean the data, and then store it in a SQLite database. In ML pipeline we split the
	data into a training set and a test set.
  
Installation

 	There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run
	with no issues using Python versions 3 and above
  
Project Motivation

  	This study is part of course that are owned by Udacity. This will be the future reference for those who looking for creating
	ETL pipeline and ML pipeline which uses GridSearchCV  with parameters.
 
Folder structure

	- app

		| - template

		| |- master.html  # main page of web app

		| |- go.html  # classification result page of web app

		|- run.py  # Flask file that runs app


	- data

		|- disaster_categories.csv  # data to process 
		
		|- disaster_messages.csv  # data to process

		|- process_data.py

		|- InsertDatabaseName.db   # database to save clean data to


	- models

		|- train_classifier.py

		|- classifier.pkl  # saved model 
	- README.md


How to run
	
	For ETL pipeline
   		 python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
  	For ML pipeline
   		 python train_classifier.py ../data/DisasterResponse.db classifier.pkl

  	For web app
		python run.py
		use env|grep to find the server address
		Go to browser and use
			https://SPACEID-3001.SPACEDOMAIN
      
 Results
 
  	Output is to classify the type of message that system receives
 
 Licensing, Authors, and Acknowledgements
 
  	All kind of Licensing and copy rights for Udacity  

	





	



