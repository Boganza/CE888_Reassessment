# import required libraries
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from sklearn.linear_model import LinearRegression

import seaborn as sns

from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt 


from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.preprocessing import PolynomialFeatures

import math
from sklearn.metrics import mean_squared_error, mean_absolute_error


# craetinf output variables 

output1=pd.read_csv('C:/Users/Dr.Farhan/Desktop/CE888/data/Questionnaire_datasetIB.csv',sep = ",", encoding='latin')
output2=pd.read_csv('C:/Users/Dr.Farhan/Desktop/CE888/data/Questionnaire_datasetIA.csv',sep = ",", encoding='latin')

final_score = output1['Total Score original'] + output2['Total Score original']


# creating dataframe of only test group, all even data files are of test group
df_new_tg = set_dataframe(2,61,2)

# creating dataframe of only controle group, all odd data files are of test group
df_new_cg = set_dataframe(1,60,2)

# creating dataframe of both groups
df_new_bg = set_dataframe(1,60,1)


# ploting a heatmap of correlation, we will try to see how input features are rlated to out put feature

corr = df_new_cg.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation matrix heatmap')

plt.show()


# we are ploting liear relation between the features which will help us reduce some linear features and also tells us that out 
# problem is linear or non linear

sns.pairplot(df_new_cg)

# removing some linear features which we dont need

df_new_cg = df_new_cg.drop(columns=['Recording no','Pupil diameter succades','Max Pupil diameter for succades'])
df_new_tg = df_new_tg.drop(columns=['Recording no','Pupil diameter succades','Max Pupil diameter for succades'])
df_new_bg = df_new_bg.drop(columns=['Recording no','Pupil diameter succades','Max Pupil diameter for succades'])


#again ploting the same erlationship

sns.pairplot(df_new)

# this is a function which makes a dataframe for us

def set_dataframe(x,y,z):
    
    # initializing the columns of our data frame

    d = {'Participant no': [], 'Recording no': [], 'Pupil diameter succades': [], 'Pupil diameter for fixations': [],'Max Pupil diameter for succades': [],'Max Pupil diameter for fixations': [],'Succade duration': [],'Fixation duration': [], 'Empythy score': []}

    # initializing our dataframe
    
    df_new = pd.DataFrame(data = d)


    
    for i in range(x,y,z):
        
        # this condition implies that if participant number is less than 10, three zeroes needed in column name otherwise two zeroes
    
        if i <= 9:
            data=pd.read_csv('D:/CE888/raw data/raw_data/Participant00' + str(0) + str(i) + '.tsv',sep='\t')
       
        else:
            data=pd.read_csv('D:/CE888/raw data/raw_data/Participant00' +  str(i) + '.tsv',sep='\t')
        
        # we are selecting just our selected features
        
        data = data[['Recording name','Pupil diameter left','Pupil diameter right','Gaze event duration','Eye movement type']]

        # by this command we will me at the last row of our dataframe and we are calculating the total length afterwards
        
        last_row = data.iloc[-1]
    
        s1=re.sub("Recording","",last_row['Recording name'])
        s1 = int(s1)

        
        data_store = data

        for j in range(1, s1+1):
    
            data = data_store

        # some column have commas which we replace with . to make it numerical values
            def fix_commas(data):
                        for col in data.select_dtypes(include='object'):
                
                            data[col] = data[col].str.replace(',', '.').astype(float)
                        return data
        
            
            # replacing "," with "."
            
            data2 = data[["Pupil diameter left","Pupil diameter right"]]

            data2 = fix_commas(data2)
            

            data[["Pupil diameter left","Pupil diameter right"]] = data2[["Pupil diameter left","Pupil diameter right"]]

            # we are filling the missing values with nearest neighbour values
            
            data.bfill(axis ='rows')

            data2 = data["Eye movement type"]

            # this categorial column has only few NaNs and we replcing them with a neutral input
            
            data["Eye movement type"] = data["Eye movement type"].replace(np.nan, 'EyesNotFound', regex=True)
            
            # putting column back to original dataframe after pre processing

            data[["Pupil diameter left","Pupil diameter right","Gaze event duration"]] = data[["Pupil diameter left","Pupil diameter right","Gaze event duration"]].astype(float)

            # only taking the data for one participant
            
            data = data[data['Recording name']=="Recording"+str(j)+""]
            
            # slplitting it into two columns

            data_succade = data[data['Eye movement type']=="Saccade"]

            data_fix = data[data['Eye movement type']=="Fixation"]
            
            # using feature extration to calculate some new features eg, mean and max

            sd = data_succade['Gaze event duration'].mean()

            sf = data_fix['Gaze event duration'].mean()

            one = data_succade['Pupil diameter left'].mean()

            two = data_succade['Pupil diameter left'].max()

            three = data_fix['Pupil diameter left'].mean()

            four = data_fix['Pupil diameter left'].max()
            
            # adding preperred data to final dataframe

            df_new = df_new.append({'Participant no': i, 'Recording no': j,'Pupil diameter succades': one, 'Pupil diameter for fixations': three,'Max Pupil diameter for succades': two, 'Max Pupil diameter for fixations': four,'Succade duration': sd, 'Fixation duration': sf, 'Empythy score': final_score[i-1]}, ignore_index=True)

    
    return df_new


# in controle group 160th row always giving NaN which in both groups dataframe bocomes 208th row

df_new_cg.drop(160,axis=0,inplace=True)
df_new_bg.drop(208,axis=0,inplace=True)


# this is the function which call our models


def applymodels(df_new,a,b):  
    
    # first we send our data to be noralize to increase calculation speed and accuracy
    df_new = normalize_me(df_new)
    rmse_model = []
    rmse_modelpoly = []
    rmse_modelDT = []
    
    # here I manually implemented grouped cross validation. It runs 10 epoch manually adjusting parameters which change in
    # each epoch and make sure a shift on validation chunk and training data
    
    for i in range(0,10):
        val = i * 6
        train = df_new.loc[(df_new['Participant no'] < (a - val))]
        train2 = df_new.loc[(df_new['Participant no'] > (b - val))]
        comb = [train, train2]
        train = pd.concat(comb)
        test=df_new.loc[(df_new['Participant no'] > (a - val)) & (df_new['Participant no'] < (b - val))]
    

        # calling polinomial model function
        rmse= apply_modelpoly(train,test)
        rmse_modelpoly.append(rmse)
    
        # calling Dt regressor model function
        rmse = apply_modelDT(train,test)
        rmse_modelDT.append(rmse)
        
    
    

    print("Root mean square error for Polynomial Regression is: ",sum(rmse_modelpoly) / len(rmse_modelpoly))
    print("Root mean square error for Decision Tree Regressor is: ",sum(rmse_modelDT) / len(rmse_modelDT))




# its a function for nomalizing our data

def normalize_me(df_new):
    
    # its just avoiding to normalize the participat number which we will not use in model traing but we need its actual value 
    # so that we can train according to to each participant
    
    data2 = df_new[["Pupil diameter for fixations","Max Pupil diameter for fixations","Succade duration","Fixation duration","Empythy score"]]
    result = data2.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))
    df_new[["Pupil diameter for fixations","Max Pupil diameter for fixations","Succade duration","Fixation duration","Empythy score"]] = result[["Pupil diameter for fixations","Max Pupil diameter for fixations","Succade duration","Fixation duration","Empythy score"]]
    return df_new


# function for implementing polinomial regression

def apply_modelpoly(train, test):
    
    # doing test and train split
    
    features = ["Pupil diameter for fixations","Max Pupil diameter for fixations","Succade duration","Fixation duration"]
    label = 'Empythy score'
    X_train = train[features]
    X_test = test[features]
    Y_train = train[label]
    Y_test = test[label].copy()
       
    # implementing polinomial regression with degree 3    
     
    poly = PolynomialFeatures(degree=3)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.fit_transform(X_test)
 
    poly.fit(X_poly_train, Y_train)
    lin2 = LinearRegression()
    lin2.fit(X_poly_train, Y_train)
    Y_pred = lin2.predict(X_poly_test)
    
    # calculating root mean square error
    
    r_sq = mean_squared_error(Y_test, Y_pred, squared=False)
    rmse = math.sqrt(r_sq)
    
  
    return rmse
   


# function for implementing decision tree regressor

def apply_modelDT(train, test):
    
    # doing test and train split
    
    features = ["Pupil diameter for fixations","Max Pupil diameter for fixations","Succade duration","Fixation duration"]
    label = 'Empythy score'
    X_train = train[features]
    X_test = test[features]
    Y_train = train[label]
    Y_test = test[label]
    
    # implementing polinomial regression with degree 3    
    
    model = DecisionTreeRegressor(random_state = 10)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    
    # calculating root mean square error
    
    r_sq = mean_squared_error(Y_test, Y_pred, squared=False)
    rmse = math.sqrt(r_sq)
    
    

    
    return rmse
# function for implementing decision tree regressor

def apply_modelDT(train, test):
    
    # doing test and train split
    
    features = ["Pupil diameter for fixations","Max Pupil diameter for fixations","Succade duration","Fixation duration"]
    label = 'Empythy score'
    X_train = train[features]
    X_test = test[features]
    Y_train = train[label]
    Y_test = test[label]
    
    # implementing polinomial regression with degree 3    
    
    model = DecisionTreeRegressor(random_state = 10)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    
    # calculating root mean square error
    
    r_sq = mean_squared_error(Y_test, Y_pred, squared=False)
    rmse = math.sqrt(r_sq)
    
    

    
    return rmse




# calling method for only controle group
applymodels(df_new_cg,54,60)


#calling method for only test group
applymodels(df_new_tg,55,61)


#calling method for both groups together
applymodels(df_new_bg,54,61)








