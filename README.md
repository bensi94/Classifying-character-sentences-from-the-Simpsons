# Classifying character sentences from the Simpsons
**Authors: Benedikt Óskarsson and Steinar Þorláksson**  
*Project in course Introduction to Machine Learning (T-504-ITML)*

![Simpsons family](https://static.independent.co.uk/s3fs-public/thumbnails/image/2012/10/02/21/pg-28-simpsons.jpg?w968h681)


## About the project 

The goal of the project was to apply theoreticical knowledge about supervised
learning algorithms and the workflow of solving a supervised learning problem
in practice. Using a realistic dataset and design a process of preprocessing the data,
selecting a model and its hyperparameters and impliment a program that automates the process.  

## Research report

The overview of the research and the results can be found in the [Research report](https://github.com/bensi94/Classifying-character-sentences-from-the-Simpsons/blob/master/Report.pdf)

## The Dataset

The dataset is called *The Simpsons by the Data* and contains in four csv files list of all Simponsons episodes, Characters, Locations and most importantly a file that contains all spoken text during each episode (including details about which character said it and where). 
  
 The Dataset is from William Cukierski at [Kaggle.com](https://www.kaggle.com/wcukierski) and is available [here](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data)
 
Datafiles:
  * [simpsons_characters.csv](https://github.com/bensi94/Classifying-character-sentences-from-the-Simpsons/blob/master/simpsons_characters.csv)
  * [simpsons_episodes.csv](https://github.com/bensi94/Classifying-character-sentences-from-the-Simpsons/blob/master/simpsons_episodes.csv)
  * [simpsons_locations.csv](https://github.com/bensi94/Classifying-character-sentences-from-the-Simpsons/blob/master/simpsons_locations.csv)
  * [simpsons_script_lines.csv](https://github.com/bensi94/Classifying-character-sentences-from-the-Simpsons/blob/master/simpsons_script_lines.csv)
 
## Code files

The project is developed with Jupyter Notebooks and Python.   
  
The main notebook file is [Classifying-character-sentences-from-the-Simpsons.ipynb](https://github.com/bensi94/Classifying-character-sentences-from-the-Simpsons/blob/master/Classifying-character-sentences-from-the-Simpsons.ipynb)
in that file we preprocess the data and do the Model selection and search for hyperparameters.  

The second code file is [Best_params_work.ipyn](https://github.com/bensi94/Classifying-character-sentences-from-the-Simpsons/blob/master/Best%20params%20work.ipynb) but in that file we have already used the selection of Models and hyperparamters from the the main file.

## Run the code locally 

Open the file Classifying-character-sentences-from-the-Simpsons.ipynb in jupyter notebook environment. Go to Kernel in the drop down menu, and press Restart and run all.

Running all the code should take from 3 - 7 hours on most computers.
