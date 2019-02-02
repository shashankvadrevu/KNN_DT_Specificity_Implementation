========
Assignment-1
========

## Installation Procedure:

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
