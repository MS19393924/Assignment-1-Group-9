#Q1

%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

#import dataset using pandas
wine1 = pd.read_csv("winequality-red.csv", delimiter= ":")
print(wine1.head())

correlation = wine1.corr()['alcohol'], drop('alcohol')
print(correlation)

print["rows, Columns:" + str(wine1.shape)]

wine1.info()
#Understanding the dataset, dataset contains 1599 sampleand 12 variables 
#including the
#As we can see that 11 variables are numerical and one, varibale is ordinal

#Define X and Y
#Dropping alcohol variable from x
#treating y as alcohol vairbale

x = wine1.drop({'alcohol'}, axis=1)
y= wine1{'alcohol'}

X = sm.add_constant(X)

x_train, x_test. y_train, y_test = train_test_split(X,y, train_size=0.0, random_state=50)
#Train the model on the training set.
model1=sm.OLS(y_train, x_train).fit()
print(model1.summary())

#print the coefficients, 
model1.params

#print the p-values for the model coeff
model1.pvalues

#predict the test set results.
y_pred=model1.predict(x_test)
print(y_pred)

#plot the results
sns.regplot(y_test,y_pred, line_kws=('color':'red'), ci=None)

plt.xlabel('Actual')
plt.ylabel('predictions')
plt.title('Prediction vs Actual')

plt.show()

#linear regression with scikit-learn
regressor = LinearRegression()
regressor.fit(x_train, y_train)
#we are getting the R^2 score of the prediction
r2_score = regressor.score(x_test, y_test)
print("The accuracy to the regression model is" r2_score#*100,'%;)

      train_pred = regressor.predict(x_train)
print(train_pred)

test_pred = regressor.predict(x_test)
print(test_pred) 

#calculating rmse
from scklearn import metrics
train_rmse = mean_squared_error(train_pred, y_train) == 0.5
print(train_rmse) 
