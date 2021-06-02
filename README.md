# LinearRegression Model Explained
This is a repository containing the explanation for Linear Regression using **Sklearn, pandas, Numpy and Seaborn**. Also performing Exploratory Data Analysis(EDA) and Visualisation. This explaination is divided into following parts and we will look each part in detail:

1. Understand the problem statement, dataset and choose ML model
2. Core Mathematics Concepts
3. Libraries Used
4. Explore the Dataset
5. Perform Visualisations
6. Perform Test_Train dataset split
7. Train the model
8. Perform the predictions
9. Model Metrics and Evaluations

## 1. Understand the problem Statement and the dataset
The data set is of the Housing price along with the various parameters affecting it. The **target variable** to be predicted is a **set of continuous values**; hence **firming our choice** to use the **Linear Regeression model**.

## 2. Core Mathematics Concepts    
**Tricks**  
Linear regression involves **moving a line such that it is the best approximation for a set of points**. The absolute trick and square trick are techniques to move a line closer to a point. Tricks are used for our understanding purposes.
  
**i) Absolute Trick**  
A line with slope w1 and y-intercept w2 would have equation ![1](https://user-images.githubusercontent.com/67451993/120429988-f13f5300-c393-11eb-9870-154ff244324f.PNG). To move the line closer to the point (p,q), the application of the absolute trick involves changing the equation of the line to 
![3](https://user-images.githubusercontent.com/67451993/120430089-1cc23d80-c394-11eb-99ff-7dba84cf81d9.PNG) 
where ![2](https://user-images.githubusercontent.com/67451993/120430166-3499c180-c394-11eb-8664-6e708b3e3ee2.PNG) is the learning rate and is a small number whose sign depends on whether the point is above or below the line.
![4](https://user-images.githubusercontent.com/67451993/120430634-ed600080-c394-11eb-98f3-d3f82e1bba8e.png)

![5](https://user-images.githubusercontent.com/67451993/120430642-f18c1e00-c394-11eb-98bf-190896089f93.png)

**ii) Square Trick**  
This method does not divide the point above or below the line, and also takes the value of q into account. The transformed equation:![9](https://user-images.githubusercontent.com/67451993/120431418-1634c580-c396-11eb-9bb1-28ca3d3a014b.PNG)
 ![6](https://user-images.githubusercontent.com/67451993/120431153-b9390f80-c395-11eb-9af5-e91c6620b622.png)
 
 **Gradient Descent (What actually happens in .fit())**
 Reduce the error by using the gradient descent method. When it reaches the lowest point in the figure, it becomes a better solution to the linear problem.
 ![7](https://user-images.githubusercontent.com/67451993/120431581-509e6280-c396-11eb-974d-e9018ee70f05.png)
 
 ## 3. Libraries Used
The following libraries are used intitally
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```
## 4. Explore the Dataset
We read the dataset into a Pandas dataframe
```python
df=pd.read_csv('/content/housing.csv')
```
The [.head()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html) gives the first 5 rows along with all the columns info for a quick glimpse of dataset
```python
df.head()
```
The [.describe()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html) function gives the description 
```python
df.describe()
```
The [.info()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html) function gives the quick infor on columns, type of data in them and valid entries
```python
df.info()
```
## 5. Perform Visualisations
We use several function from seaborn library to visualize.  
Seaborn is built on MatplotLib library with is built on MATLAB. So people experienced with MATLAB/OCTAVE will find its syntax similar.

[Pairplot](https://seaborn.pydata.org/generated/seaborn.pairplot.html) is quickly used to plot multiple pairwise bivariate distributions
```python
sns.pairplot(df)
```
[Heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html?highlight=heatmap#seaborn.heatmap) gives a overview of how well different features are co-related
```python
sns.heatmap(df.corr(), annot=True)
```
[Jointplot](http://seaborn.pydata.org/generated/seaborn.jointplot.html) gives visualizations with multiple pairwise plots with focus on a single relationship.
```python
sns.jointplot(x='RM',y='MEDV',data=df)
```
[Lmplot](https://seaborn.pydata.org/generated/seaborn.lmplot.html?highlight=lmplot#seaborn.lmplot) gives a Scatter plot with regression line
```python
sns.lmplot(x='LSTAT', y='MEDV',data=df)
sns.lmplot(x='LSTAT', y='RM',data=df)
```

## 6. Perform Test_Train dataset split
We divide the Dataset into 2 parts, Train and test respectively.  
We set test_size as 0.30 of dataset for validation. Random_state is used to ensure split is same everytime we execute the code
```python
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
```
## 7. Train the model
The mathematical concepts we saw above is implemented in single [.fit()](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) statement
```python
from sklearn.linear_model import LinearRegression  #Importing the LinerRegression from sklearn
lm=LinearRegression()                              #Create LinerRegression object so the manupulation later is easy
lm.fit(X_train, y_train)                           #The fit happens here
```
## 8. Perform the predictions
Prediction of the values for testing set and save it in the predictions variable. The .coef_ module is used to get the coefficients(weights) that infuences the values of features
```python
predictions=lm.predict(X_test)
lm.coef_
```
## 9. Model Metrics and Evaluations
The metrics are very important to inspect the accuracy of the model. The metrics are:    
  
**i) Mean Absolute Error (MAE)** : It is the total sum of differences between predicted versus actual value divied by the number of points in dataset. Equation given by:  
![10](https://user-images.githubusercontent.com/67451993/120431949-d1f5f500-c396-11eb-950e-f6639a45f44d.PNG) 
![12](https://user-images.githubusercontent.com/67451993/120431954-d4f0e580-c396-11eb-9f8e-c5adc3c96a97.png)
**ii) Mean Squared Error (MSE)** :  It is the total of average squared difference between the estimated values and the actual values. Equation given by:  
![11](https://user-images.githubusercontent.com/67451993/120431953-d4584f00-c396-11eb-9fdd-b0795c3e4da2.PNG)
![13](https://user-images.githubusercontent.com/67451993/120431955-d4f0e580-c396-11eb-9488-fb65a59561fc.png)
**iii) Sqaure Root of Mean Sqare Error** : Same as Mean Absolute Error, a good measure of accuracy, but only to compare prediction errors of different models or model configurations for a particular variable.
```python
from sklearn import metrics
print(metrics.mean_absolute_error(y_test, predictions))
print(metrics.mean_squared_error(y_test, predictions))
print(np.sqrt(metrics.mean_squared_error(y_test, predictions)))
```
