{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "f1138db5",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "import pandas as pd\n",
    "import csv\n",
    "from scipy import stats\n",
    "import matplotlib as mpl\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.model_selection import cross_val_score\n",
    "from sklearn.model_selection import train_test_split\n",
    "import statsmodels.api as sm\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.metrics import mean_squared_error\n",
    "import numpy as np\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "e7ce7d68",
   "metadata": {},
   "outputs": [],
   "source": [
    "# import dataset\n",
    "wine=pd.read_csv(\"winequality-red.csv\",delimiter=\";\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "52252bec",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>fixed acidity</th>\n",
       "      <th>volatile acidity</th>\n",
       "      <th>citric acid</th>\n",
       "      <th>residual sugar</th>\n",
       "      <th>chlorides</th>\n",
       "      <th>free sulfur dioxide</th>\n",
       "      <th>total sulfur dioxide</th>\n",
       "      <th>density</th>\n",
       "      <th>pH</th>\n",
       "      <th>sulphates</th>\n",
       "      <th>alcohol</th>\n",
       "      <th>quality</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>7.4</td>\n",
       "      <td>0.70</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1.9</td>\n",
       "      <td>0.076</td>\n",
       "      <td>11.0</td>\n",
       "      <td>34.0</td>\n",
       "      <td>0.9978</td>\n",
       "      <td>3.51</td>\n",
       "      <td>0.56</td>\n",
       "      <td>9.4</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>7.8</td>\n",
       "      <td>0.88</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2.6</td>\n",
       "      <td>0.098</td>\n",
       "      <td>25.0</td>\n",
       "      <td>67.0</td>\n",
       "      <td>0.9968</td>\n",
       "      <td>3.20</td>\n",
       "      <td>0.68</td>\n",
       "      <td>9.8</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>7.8</td>\n",
       "      <td>0.76</td>\n",
       "      <td>0.04</td>\n",
       "      <td>2.3</td>\n",
       "      <td>0.092</td>\n",
       "      <td>15.0</td>\n",
       "      <td>54.0</td>\n",
       "      <td>0.9970</td>\n",
       "      <td>3.26</td>\n",
       "      <td>0.65</td>\n",
       "      <td>9.8</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>11.2</td>\n",
       "      <td>0.28</td>\n",
       "      <td>0.56</td>\n",
       "      <td>1.9</td>\n",
       "      <td>0.075</td>\n",
       "      <td>17.0</td>\n",
       "      <td>60.0</td>\n",
       "      <td>0.9980</td>\n",
       "      <td>3.16</td>\n",
       "      <td>0.58</td>\n",
       "      <td>9.8</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>7.4</td>\n",
       "      <td>0.70</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1.9</td>\n",
       "      <td>0.076</td>\n",
       "      <td>11.0</td>\n",
       "      <td>34.0</td>\n",
       "      <td>0.9978</td>\n",
       "      <td>3.51</td>\n",
       "      <td>0.56</td>\n",
       "      <td>9.4</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \\\n",
       "0            7.4              0.70         0.00             1.9      0.076   \n",
       "1            7.8              0.88         0.00             2.6      0.098   \n",
       "2            7.8              0.76         0.04             2.3      0.092   \n",
       "3           11.2              0.28         0.56             1.9      0.075   \n",
       "4            7.4              0.70         0.00             1.9      0.076   \n",
       "\n",
       "   free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \\\n",
       "0                 11.0                  34.0   0.9978  3.51       0.56   \n",
       "1                 25.0                  67.0   0.9968  3.20       0.68   \n",
       "2                 15.0                  54.0   0.9970  3.26       0.65   \n",
       "3                 17.0                  60.0   0.9980  3.16       0.58   \n",
       "4                 11.0                  34.0   0.9978  3.51       0.56   \n",
       "\n",
       "   alcohol  quality  \n",
       "0      9.4        5  \n",
       "1      9.8        5  \n",
       "2      9.8        5  \n",
       "3      9.8        6  \n",
       "4      9.4        5  "
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "wine.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "91bc1aba",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>fixed acidity</th>\n",
       "      <th>volatile acidity</th>\n",
       "      <th>citric acid</th>\n",
       "      <th>residual sugar</th>\n",
       "      <th>chlorides</th>\n",
       "      <th>free sulfur dioxide</th>\n",
       "      <th>total sulfur dioxide</th>\n",
       "      <th>density</th>\n",
       "      <th>pH</th>\n",
       "      <th>sulphates</th>\n",
       "      <th>alcohol</th>\n",
       "      <th>quality</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>1599.000000</td>\n",
       "      <td>1599.000000</td>\n",
       "      <td>1599.000000</td>\n",
       "      <td>1599.000000</td>\n",
       "      <td>1599.000000</td>\n",
       "      <td>1599.000000</td>\n",
       "      <td>1599.000000</td>\n",
       "      <td>1599.000000</td>\n",
       "      <td>1599.000000</td>\n",
       "      <td>1599.000000</td>\n",
       "      <td>1599.000000</td>\n",
       "      <td>1599.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>8.319637</td>\n",
       "      <td>0.527821</td>\n",
       "      <td>0.270976</td>\n",
       "      <td>2.538806</td>\n",
       "      <td>0.087467</td>\n",
       "      <td>15.874922</td>\n",
       "      <td>46.467792</td>\n",
       "      <td>0.996747</td>\n",
       "      <td>3.311113</td>\n",
       "      <td>0.658149</td>\n",
       "      <td>10.422983</td>\n",
       "      <td>5.636023</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>1.741096</td>\n",
       "      <td>0.179060</td>\n",
       "      <td>0.194801</td>\n",
       "      <td>1.409928</td>\n",
       "      <td>0.047065</td>\n",
       "      <td>10.460157</td>\n",
       "      <td>32.895324</td>\n",
       "      <td>0.001887</td>\n",
       "      <td>0.154386</td>\n",
       "      <td>0.169507</td>\n",
       "      <td>1.065668</td>\n",
       "      <td>0.807569</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>4.600000</td>\n",
       "      <td>0.120000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.900000</td>\n",
       "      <td>0.012000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>0.990070</td>\n",
       "      <td>2.740000</td>\n",
       "      <td>0.330000</td>\n",
       "      <td>8.400000</td>\n",
       "      <td>3.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>7.100000</td>\n",
       "      <td>0.390000</td>\n",
       "      <td>0.090000</td>\n",
       "      <td>1.900000</td>\n",
       "      <td>0.070000</td>\n",
       "      <td>7.000000</td>\n",
       "      <td>22.000000</td>\n",
       "      <td>0.995600</td>\n",
       "      <td>3.210000</td>\n",
       "      <td>0.550000</td>\n",
       "      <td>9.500000</td>\n",
       "      <td>5.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>7.900000</td>\n",
       "      <td>0.520000</td>\n",
       "      <td>0.260000</td>\n",
       "      <td>2.200000</td>\n",
       "      <td>0.079000</td>\n",
       "      <td>14.000000</td>\n",
       "      <td>38.000000</td>\n",
       "      <td>0.996750</td>\n",
       "      <td>3.310000</td>\n",
       "      <td>0.620000</td>\n",
       "      <td>10.200000</td>\n",
       "      <td>6.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>9.200000</td>\n",
       "      <td>0.640000</td>\n",
       "      <td>0.420000</td>\n",
       "      <td>2.600000</td>\n",
       "      <td>0.090000</td>\n",
       "      <td>21.000000</td>\n",
       "      <td>62.000000</td>\n",
       "      <td>0.997835</td>\n",
       "      <td>3.400000</td>\n",
       "      <td>0.730000</td>\n",
       "      <td>11.100000</td>\n",
       "      <td>6.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>15.900000</td>\n",
       "      <td>1.580000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>15.500000</td>\n",
       "      <td>0.611000</td>\n",
       "      <td>72.000000</td>\n",
       "      <td>289.000000</td>\n",
       "      <td>1.003690</td>\n",
       "      <td>4.010000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>14.900000</td>\n",
       "      <td>8.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       fixed acidity  volatile acidity  citric acid  residual sugar  \\\n",
       "count    1599.000000       1599.000000  1599.000000     1599.000000   \n",
       "mean        8.319637          0.527821     0.270976        2.538806   \n",
       "std         1.741096          0.179060     0.194801        1.409928   \n",
       "min         4.600000          0.120000     0.000000        0.900000   \n",
       "25%         7.100000          0.390000     0.090000        1.900000   \n",
       "50%         7.900000          0.520000     0.260000        2.200000   \n",
       "75%         9.200000          0.640000     0.420000        2.600000   \n",
       "max        15.900000          1.580000     1.000000       15.500000   \n",
       "\n",
       "         chlorides  free sulfur dioxide  total sulfur dioxide      density  \\\n",
       "count  1599.000000          1599.000000           1599.000000  1599.000000   \n",
       "mean      0.087467            15.874922             46.467792     0.996747   \n",
       "std       0.047065            10.460157             32.895324     0.001887   \n",
       "min       0.012000             1.000000              6.000000     0.990070   \n",
       "25%       0.070000             7.000000             22.000000     0.995600   \n",
       "50%       0.079000            14.000000             38.000000     0.996750   \n",
       "75%       0.090000            21.000000             62.000000     0.997835   \n",
       "max       0.611000            72.000000            289.000000     1.003690   \n",
       "\n",
       "                pH    sulphates      alcohol      quality  \n",
       "count  1599.000000  1599.000000  1599.000000  1599.000000  \n",
       "mean      3.311113     0.658149    10.422983     5.636023  \n",
       "std       0.154386     0.169507     1.065668     0.807569  \n",
       "min       2.740000     0.330000     8.400000     3.000000  \n",
       "25%       3.210000     0.550000     9.500000     5.000000  \n",
       "50%       3.310000     0.620000    10.200000     6.000000  \n",
       "75%       3.400000     0.730000    11.100000     6.000000  \n",
       "max       4.010000     2.000000    14.900000     8.000000  "
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "wine.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "d0f6f767",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1599, 12)"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "wine.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "df79af41",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "fixed acidity           float64\n",
       "volatile acidity        float64\n",
       "citric acid             float64\n",
       "residual sugar          float64\n",
       "chlorides               float64\n",
       "free sulfur dioxide     float64\n",
       "total sulfur dioxide    float64\n",
       "density                 float64\n",
       "pH                      float64\n",
       "sulphates               float64\n",
       "alcohol                 float64\n",
       "quality                   int64\n",
       "dtype: object"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "wine.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "3876d0d9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1599 entries, 0 to 1598\n",
      "Data columns (total 12 columns):\n",
      " #   Column                Non-Null Count  Dtype  \n",
      "---  ------                --------------  -----  \n",
      " 0   fixed acidity         1599 non-null   float64\n",
      " 1   volatile acidity      1599 non-null   float64\n",
      " 2   citric acid           1599 non-null   float64\n",
      " 3   residual sugar        1599 non-null   float64\n",
      " 4   chlorides             1599 non-null   float64\n",
      " 5   free sulfur dioxide   1599 non-null   float64\n",
      " 6   total sulfur dioxide  1599 non-null   float64\n",
      " 7   density               1599 non-null   float64\n",
      " 8   pH                    1599 non-null   float64\n",
      " 9   sulphates             1599 non-null   float64\n",
      " 10  alcohol               1599 non-null   float64\n",
      " 11  quality               1599 non-null   int64  \n",
      "dtypes: float64(11), int64(1)\n",
      "memory usage: 150.0 KB\n"
     ]
    }
   ],
   "source": [
    "wine.info()\n",
    "#give info about dataset,dataset contains 1599 samples(rows) and 12 variables (columns)\n",
    "# one variable is ordinal and the rest are numerical."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "44a017c6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fixed acidity          -0.061668\n",
      "volatile acidity       -0.202288\n",
      "citric acid             0.109903\n",
      "residual sugar          0.042075\n",
      "chlorides              -0.221141\n",
      "free sulfur dioxide    -0.069408\n",
      "total sulfur dioxide   -0.205654\n",
      "density                -0.496180\n",
      "pH                      0.205633\n",
      "sulphates               0.093595\n",
      "quality                 0.476166\n",
      "Name: alcohol, dtype: float64\n"
     ]
    }
   ],
   "source": [
    "correlation= wine.corr()['alcohol'].drop('alcohol')\n",
    "print(correlation)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "acc3fc23",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                            OLS Regression Results                            \n",
      "==============================================================================\n",
      "Dep. Variable:                alcohol   R-squared:                       0.700\n",
      "Model:                            OLS   Adj. R-squared:                  0.697\n",
      "Method:                 Least Squares   F-statistic:                     251.7\n",
      "Date:                Fri, 27 Aug 2021   Prob (F-statistic):          6.26e-301\n",
      "Time:                        19:31:43   Log-Likelihood:                -1056.3\n",
      "No. Observations:                1199   AIC:                             2137.\n",
      "Df Residuals:                    1187   BIC:                             2198.\n",
      "Df Model:                          11                                         \n",
      "Covariance Type:            nonrobust                                         \n",
      "========================================================================================\n",
      "                           coef    std err          t      P>|t|      [0.025      0.975]\n",
      "----------------------------------------------------------------------------------------\n",
      "const                  569.1049     15.234     37.358      0.000     539.216     598.993\n",
      "fixed acidity            0.5060      0.023     21.778      0.000       0.460       0.552\n",
      "volatile acidity         0.5740      0.128      4.474      0.000       0.322       0.826\n",
      "citric acid              0.7621      0.153      4.975      0.000       0.462       1.063\n",
      "residual sugar           0.2561      0.014     18.203      0.000       0.228       0.284\n",
      "chlorides               -0.9790      0.449     -2.180      0.029      -1.860      -0.098\n",
      "free sulfur dioxide     -0.0049      0.002     -2.125      0.034      -0.009      -0.000\n",
      "total sulfur dioxide    -0.0009      0.001     -1.135      0.257      -0.002       0.001\n",
      "density               -579.4754     15.573    -37.210      0.000    -610.029    -548.922\n",
      "pH                       3.5519      0.170     20.928      0.000       3.219       3.885\n",
      "sulphates                0.9432      0.117      8.082      0.000       0.714       1.172\n",
      "quality                  0.2406      0.026      9.318      0.000       0.190       0.291\n",
      "==============================================================================\n",
      "Omnibus:                       76.588   Durbin-Watson:                   2.036\n",
      "Prob(Omnibus):                  0.000   Jarque-Bera (JB):              132.432\n",
      "Skew:                           0.468   Prob(JB):                     1.75e-29\n",
      "Kurtosis:                       4.332   Cond. No.                     7.66e+04\n",
      "==============================================================================\n",
      "\n",
      "Notes:\n",
      "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
      "[2] The condition number is large, 7.66e+04. This might indicate that there are\n",
      "strong multicollinearity or other numerical problems.\n"
     ]
    }
   ],
   "source": [
    "x= wine.drop(['alcohol'],axis=1)\n",
    "y= wine['alcohol']\n",
    "x= sm.add_constant(x)\n",
    "x_train, x_test, y_train, y_test = train_test_split(x,y)\n",
    "model= sm.OLS(y_train,x_train).fit()\n",
    "print(model.summary())\n",
    "#put x and y\n",
    "#drop alcohol from x\n",
    "# treat y as alcohol variable\n",
    "# split the data in training set and testing set\n",
    "# 80 % trainig set\n",
    "# 20% testing set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "b074e933",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "const                   569.104904\n",
       "fixed acidity             0.506015\n",
       "volatile acidity          0.574047\n",
       "citric acid               0.762111\n",
       "residual sugar            0.256098\n",
       "chlorides                -0.978962\n",
       "free sulfur dioxide      -0.004881\n",
       "total sulfur dioxide     -0.000902\n",
       "density                -579.475362\n",
       "pH                        3.551946\n",
       "sulphates                 0.943186\n",
       "quality                   0.240570\n",
       "dtype: float64"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.params\n",
    "# coefficients"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "6666d4ac",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "const                   1.346688e-202\n",
       "fixed acidity            9.715984e-89\n",
       "volatile acidity         8.404264e-06\n",
       "citric acid              7.495928e-07\n",
       "residual sugar           1.728567e-65\n",
       "chlorides                2.945054e-02\n",
       "free sulfur dioxide      3.375158e-02\n",
       "total sulfur dioxide     2.566439e-01\n",
       "density                 1.691587e-201\n",
       "pH                       4.994911e-83\n",
       "sulphates                1.555274e-15\n",
       "quality                  5.570875e-20\n",
       "dtype: float64"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.pvalues\n",
    "# porint the p value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "03ea028f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "109      9.701079\n",
      "1302    11.509013\n",
      "347     11.897605\n",
      "766      9.472065\n",
      "823      9.954601\n",
      "          ...    \n",
      "1254    10.389986\n",
      "1370     9.926119\n",
      "720      9.041141\n",
      "1068    11.983061\n",
      "475      9.106365\n",
      "Length: 400, dtype: float64\n"
     ]
    }
   ],
   "source": [
    "y_pred=model.predict(x_test)\n",
    "print(y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "1e3bf1e7",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.regplot(y_test,y_pred,line_kws={'color':'orange'},ci=None)\n",
    "plt.xlabel('Actuall')\n",
    "plt.ylabel('predictions')\n",
    "plt.title('prediction vs Actuall')\n",
    "plt.show()\n",
    "# plot the data\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "54c5db26",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "the accuracy of model is 69.9941023904628 %\n"
     ]
    }
   ],
   "source": [
    "regressor= LinearRegression()\n",
    "regressor.fit (x_train,y_train)\n",
    "r2_score = regressor.score(x_train,y_train)\n",
    "print(\"the accuracy of model is\",r2_score*100,'%')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "72f57140",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[11.52890954 10.12521331 11.79767481 ...  9.45732867  9.83594342\n",
      " 11.33454433]\n"
     ]
    }
   ],
   "source": [
    "train_pred = regressor.predict(x_train)\n",
    "print(train_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "53b31f97",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 9.70107875 11.50901266 11.89760531  9.47206506  9.95460126 10.83420978\n",
      "  9.69587705  9.85055765 13.18697087 11.30258374  9.57368579 10.0481382\n",
      " 11.99629768 10.79367822 11.94784529 10.92852368 10.16095885 11.40334268\n",
      "  9.51230753 11.36792629 10.83497023  9.3500056  11.9670464  10.04761234\n",
      "  9.4329684   9.52280715 10.00801218  9.95797486  9.37174193  9.27267166\n",
      " 10.56295816  9.70458431  9.60188637 10.30803001  9.99913243 10.21670909\n",
      " 10.19378499 12.37847609  9.90793122 10.43498623  8.46816153 10.52861165\n",
      " 10.53404722  9.91037119 12.65855946  9.77679644  9.24943136 10.44938781\n",
      " 10.59039426 11.34859313 10.030981   11.72869142  9.77072359  9.98658467\n",
      "  9.72018734  8.90572833 10.94600605 11.82524838 10.69754521  9.95981349\n",
      "  9.36437247 10.31853502 11.97370697  9.60756652 11.93407571  9.70347918\n",
      " 10.88790695 10.48770101  9.80216309 10.15711503 10.61970894  9.91082093\n",
      " 11.47052952 10.45857257 10.29676524 10.0469063   9.53382984  9.79964696\n",
      " 10.25261603  9.81520241 10.51730261  9.71941863  9.78896385 11.25903576\n",
      "  9.79203101 11.91358281 10.53098478 10.10408239 10.73085205 10.33405242\n",
      "  9.42711949  9.41092193 10.06148526  9.25564957 10.10996469 10.46080389\n",
      " 10.31567825 10.82690257 11.5707738   9.92055847  9.26789883 11.11684903\n",
      "  9.80029677  9.61676724 11.99896231 11.90039631  8.86346978 11.03668059\n",
      " 10.94550899  9.73656921 11.05421317 10.47835623  9.94273015 10.88130694\n",
      " 10.36098253 11.78160773  9.45520493  9.51330363 10.65445068  9.82799733\n",
      " 11.0276534  11.8429086  13.11049055 10.11801952 10.45121148  9.81460584\n",
      " 10.9418286   9.40025703 11.0816363   8.80662701 11.67346485 11.44694486\n",
      "  9.88334901 10.17708431  9.71542236  9.44898796  9.85066053  9.51141422\n",
      " 10.88709395 11.50666458  9.43399397  9.79964696  9.98534753  9.74140433\n",
      "  9.99474887 10.95150958  8.99193779 10.7587539   9.8886897  11.5383419\n",
      " 10.73085205 11.44753232  9.54331437  9.85557979 10.13343321  9.36601227\n",
      " 12.11021709 12.53952384  9.76840106 10.49987684 11.6445705  10.44485285\n",
      "  9.5141471  11.34777419 10.35987437 10.01780731 11.14154448 10.80824584\n",
      " 11.32488293 10.0177582  11.30258002 12.28317307  8.9699866  10.11581301\n",
      " 10.17853583 11.13051062 11.67177193 11.56116856  9.47451362 11.71516782\n",
      " 11.06050432  9.98828608  9.81182325 10.88337696 11.17347729  8.96061707\n",
      " 11.41739342 10.69652817 10.23289202  9.44436757 10.65309427 10.05909321\n",
      " 10.6707743  11.03529983 10.65284586 10.26159018  9.71423651 10.36982874\n",
      " 11.79782314 11.1575342   9.70452686  8.85282683 11.05330808  9.19989488\n",
      "  9.44880039  9.90849386 10.65833448 10.08415554 10.71921971  9.64557797\n",
      " 11.89429795 11.46465326  9.65834186  8.80548154  9.1063652  10.70386419\n",
      "  9.60083191 12.78028883  9.53133412 10.04124529 10.40664045 10.74068086\n",
      " 11.91419837 11.69005349 11.66170971 10.0644377   9.38803403  9.89368797\n",
      "  9.65147216  9.44880039  9.78410075 10.74571142  9.69202114 10.35280876\n",
      " 11.51024304 10.4432646   9.05176357  9.28097231 13.3804679   9.83539108\n",
      " 11.45909104  9.86407819 10.29516174 10.18578421  9.70067567 10.94773658\n",
      " 10.99159224 10.43136317  9.66461084  9.67333634 11.54770865 10.95628961\n",
      "  9.99882319 10.07836668 11.14565943 10.13552599 10.28698563 11.27880682\n",
      "  9.640435    9.66932064 10.01673941 10.84709382 10.53450884 11.43558607\n",
      "  9.84360154 10.63618033 10.47083543  9.23615021  9.66285288 10.90703257\n",
      "  9.8584524  10.97980176  9.68475686 10.76531378 10.21864448  9.98477574\n",
      " 10.3357369  10.85038019 10.43059576 10.77444651  9.90396787  9.9032894\n",
      " 10.37373371  9.9074607  10.0089751  10.04269677 12.46018252  9.04850609\n",
      " 10.09057177 10.22689679 10.03260789  9.78478064 12.1370706   9.40368017\n",
      "  9.70635834  8.89409391 10.70386419 11.40793474  9.94324642  9.83745565\n",
      " 10.15256581 12.29351913 11.91054143 10.45352485  9.25096609 10.42202607\n",
      " 11.47522332 11.7233777  10.55766998  9.63651644 11.52890954 10.44419432\n",
      "  9.77728399 10.41674667  9.93290064 11.92795618 10.01786331 10.26492108\n",
      " 10.72041193 10.05428609 11.0376812  10.64431309 10.31784185 10.95628961\n",
      "  9.62779683 10.20243044 11.88803494  9.95253667  9.77072359 10.33571827\n",
      " 10.134625    9.71600436  9.51623246  9.61118673  9.75936272  9.95326886\n",
      " 10.7587539  10.95567969  9.99024178 10.31708696  9.43081503 10.51551118\n",
      " 10.08972052  9.78348498 11.11998822 10.08834038  9.83324088 10.42902943\n",
      " 11.40127223 11.80818364 11.66457013  9.12429219 11.41739342 10.26557736\n",
      " 11.94998928 11.34445622 12.21043525  9.62296732 10.85255648 10.15194694\n",
      " 11.20384169 12.05972732  9.7793945  11.57883013  9.91442352 10.43902471\n",
      " 11.72155392  9.68953882 10.33108721 10.77196674  9.24205507  8.21339584\n",
      " 11.48215567  9.23760318  9.88510937  9.59855604  9.20030729  9.25790796\n",
      " 11.30258374  9.67076078  9.57701837 10.46080389 11.72869142 10.10996469\n",
      " 10.46396318 10.49761749  9.72026697  9.93921289  9.94915738 10.3132972\n",
      " 10.77610348  8.90670697  9.81700976  9.12429219  9.86757256 10.38998561\n",
      "  9.9261194   9.04114147 11.98306116  9.1063652 ]\n"
     ]
    }
   ],
   "source": [
    "test_pred= regressor.predict(x_test)\n",
    "print(test_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "9d7dd16c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.5839584831431236\n"
     ]
    }
   ],
   "source": [
    "train_rmse=mean_squared_error(train_pred,y_train)**0.5\n",
    "print(train_rmse)\n",
    "# calculate rmse"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "638f3507",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
