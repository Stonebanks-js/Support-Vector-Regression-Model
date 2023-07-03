# Support-Vector-Regression-Model
A model that tests dataset based upon the working of Support Vector Regression

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Importing the necessary libraries:

numpy (as np): A library for numerical operations in Python.
matplotlib.pyplot (as plt): A library for creating visualizations in Python.
pandas (as pd): A library for data manipulation and analysis in Python.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Importing the dataset:

The code assumes that there is a file named 'Position_Salaries.csv' at the specified path '/home/aradhya/Documents/Regression Models /Position_Salaries.csv'.
The data is read using the read_csv function from pandas, and the independent variable (X) and dependent variable (y) are extracted from the dataset.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Preprocessing the data:

The dependent variable (y) is reshaped using the reshape method to ensure it is a 2D array.
The StandardScaler from scikit-learn is used to standardize the features (X) and the target variable (y).
The fit_transform method is used to compute the mean and standard deviation of X and y and then perform the standardization.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Training the SVR model:

The SVR class from scikit-learn is used to create a regression object named regressor.
The fit method is called to train the model using the standardized features (X) and target variable (y).

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Predicting and visualizing the results:

The predict method of the regressor object is used to predict the salaries for the provided feature value (6.5) after scaling it using sc_X.transform.
The inverse transformations sc_X.inverse_transform and sc_y.inverse_transform are used to convert the standardized data back to the original scale.
A scatter plot is created using the original feature values and salaries to visualize the actual data points.
The SVR predictions are plotted as a blue line on the scatter plot.
The same plot is shown using plt.show().

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Visualizing the continuous curve:

A continuous range of feature values is created using np.arange to create a smooth curve for visualization.
The feature values are transformed using sc_X.transform to be compatible with the SVR model.
The SVR predictions for the transformed feature values are plotted as a blue line on a scatter plot.
The same plot is shown using plt.show().

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

