# Linear-Regression

##### Here,We understand what will in Linear regression model file.
First we import libraries that we will use are NumPy, Pandas, Matplotlib, Seaborn and scikit-learn. Then we Load housing Data from pandas. The pd.read_csv() Method is used for load data set


![1](https://user-images.githubusercontent.com/67451993/118117000-a1aade80-b408-11eb-9a85-52e6aa1eeefd.PNG)

In this Project we use Housing Data those availabe in repository or here some information about data set

# **Boston housing data**
Construct a working model which has the capability of predicting the value of houses. The features ‘RM’, ‘LSTAT’ and ‘PTRATIO’ give us quantitative information about each data point. The Target variable, ‘MEDV’ is the variable we seek to predict.

Note:
1) RM: This is the average number of rooms per dwelling.

2) MEDV: This is the median value of owner-occupied homes in $1000s.

3) LSTAT: This is the percentage lower status of the population.

4) PTRATIO: This is the pupil-teacher ratio by town

After this we check correlation between columns, for that we use seaborn. The sns.Heatmap() used for view correlation.
In this method if you give annot= True means numbers of correlation will be displaied.

![2](https://user-images.githubusercontent.com/67451993/118118142-28ac8680-b40a-11eb-8cef-bea158ea1dc9.PNG)

After this we need to check Relation of columns with each other so here we use seaborn's sns.pairplot() plot to view data.

![3](https://user-images.githubusercontent.com/67451993/118118631-d0c24f80-b40a-11eb-80f2-dfd7cf81108b.PNG)

##Preparing the data for training the model
we concentrate MEDV value of data set
And then Splitting the data into training and testing sets.

![4](https://user-images.githubusercontent.com/67451993/118119171-973e1400-b40b-11eb-804a-fc018475fb4c.PNG)

###Training and testing the model
We use scikit-learn’s LinearRegression to train our model on both the training and test sets.

![5](https://user-images.githubusercontent.com/67451993/118119358-e08e6380-b40b-11eb-9341-7162228ac6d0.PNG)

Final model graph

![6](https://user-images.githubusercontent.com/67451993/118119526-1d5a5a80-b40c-11eb-8964-655cd3b33b91.PNG)
