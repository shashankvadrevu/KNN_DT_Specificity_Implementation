Assignment-1
============

# K-Nearest Neighbours -- Specificity based model implementation for Scikit Learn

## Requirement

1. Choose the best k by k-fold cross validation

We can use two implementation of SKlearn are available based on the requirements

* [Model Selection class has KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold). This can be used as a generator / iterator 

* [Model Selection using GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn-model-selection-gridsearchcv) 

where, cross validation parameter 

	cv : int, cross-validation generator or an iterable, optional
	
2. shuffle the training data randomly before splitting it into groups to be used for cross validation

* When GridSearchCV is used and cv is passed as int, it does not guarantee shuffle. Therefore, if we plan to use GridSearchCV it is better to use a cross validation generator and use this generator in GridSearchCV. 

3. Training and Testing performance using k-fold cross validation, in terms of precision, recall, accuracy, sensitivity and specificity

* GridSearchCV has model selection based on a parameter called **[Scoring ](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring)** . Arguments that can be passed to [Scoring](https://scikit-learn.org/stable/modules/model_evaluation.html#using-multiple-metric-evaluation) As an iterable of string metrics or As a dict mapping the scorer name to the scoring function. 

## Gap

* The metrics that are currently supported by SKlearn 0.20.2 are listed [here](https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values). This list does not include Specificity. 

* Alternatively if some other metrics are required, it is [suggested](https://scikit-learn.org/stable/modules/model_evaluation.html#using-multiple-metric-evaluation) to generate a confusion matrix and calculate required metrics.

## Current Process

### K-Nearest Neighbours 

1. Pre-Process the data i.e. handle missing values, scale the values if required
2. Split data into Train and Test
3. For each neighbour initialize a KNN class, a cross-validation class
4. Generate cross-validation Split generator
5. For each Split, split train data into cross-validation holdout and rest
6. Build the model on the rest of the data and use cross-validation holdout to predict cross-validation holdout values.
7. Calculate confusion matrix based on actual cross-validation holdout values and predicted cross-validation holdout values.
8. Based on confusion matrix values at each split calculate Recall, Precision, Specificity.
9. Aggregate these values for each split 
10. Average out accuracy measures values for each neighbour

Select a model based on accuracy measures for each node.   

# Installation Procedure:

1. Check the version of pip on the local machine. PIP is already installed if you are using Python 2 >=2.7.9 or Python 3 >=3.4 downloaded from [python.org](python.org). Just make sure to [upgrade pip](https://pip.pypa.io/en/stable/installing/#upgrading-pip). To use pipenv we would require python 3.6 or higher. To check if pip is install in the local machine, in terminal

		pip
		pip --version
		pip install -U pip
		python --version 

2. The project uses [pipenv package](https://pipenv.readthedocs.io/en/latest/) to manage environments. Check if module - pipenv is installed in the local system, in terminal

		python -c"import pipenv"
		
* if you get the following **error** message:	
		
		Traceback (most recent call last):
		File "<string>", line 1, in <module>
		ModuleNotFoundError: No module named 'pipenv' 

* Pipenv can be installed using pip. **Note**:If the local system has two or more python executables make sure to install pipenv in appropriate python 3
	
		pip install pipenv  - or
		pip3 install pipenv


3. Unzip the file.


4. Check if pipfile exits in the synced repo of the local system.

		cat Pipfile
		
	Output msg:
		
		[[source]]
		name = "pypi"
		url = "https://pypi.org/simple"
		verify_ssl = true
		[dev-packages]
		tox = "*"
		[packages]
		[requires]
		python_version = "3.6"

7. Start a virtual environment using pipenv in the project directory

		pipenv shell - or
		sudo pipenv --<<python executable>>
		Example:
		sudo pipenv --python 3.6 

8. Intall dependencies mentioned in the Pipfile. This eliminates the requirement of requierments.txt

		pipenv install

9. To check Questions 2 and 4. Use the following method.
		
		Question 2
		Use the following command.
					python path/of/the/script path/of/the/Folder
					Example:
					python src\assign_1\q4_knn.py F:\University of Waterloo\Winter 2019\CS 680\Assignments\assign_1\knn-dataset\
		

		Question 4
					Use the following command.
					python path/of/the/script path/of/the/file/wdbc-train.csv path/of/the/file/wdbc-test.csv
					Example:
					python src\assign_1\q4_DT.py F:\University of Waterloo\Winter 2019\CS 680\Assignments\assign_1\wdbc-train.csv F:\University of Waterloo\Winter 2019\CS 680\Assignments\assign_1\wdbc-test.csv 



Note
====

This project has been set up using PyScaffold 3.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.

The Questions 2 and 4 were solved collaboratively with Shreesha Addala
