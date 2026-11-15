#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data=pd.read_csv(r"C:\Users\anagha\Downloads\Telecom+Customer+Churn\telecom_customer_churn.csv")
data.head()



# In[3]:


#Data exploration

data.drop("Customer ID" , axis='columns',inplace= True)
data.dtypes


# In[4]:


data.isnull().sum()


# In[5]:


data.rename(columns=lambda x: x.strip().replace(' ', '_'), inplace=True)


# In[6]:


from statistics import mode

m0=print("Multiple_lines",mode(data['Multiple_Lines']))
m1=print("Internet_type:",mode(data['Internet_Type']))
m2=print("online_Security:",mode(data['Online_Security']))
m3=print("Online_ Backup:", mode(data['Online_Backup']))
m4=print("Device protection plan:", mode(data["Device_Protection_Plan"]))
m5=print("Premium Tech Support:", mode(data["Premium_Tech_Support"]))
m6=print('Streaming TV:', mode(data["Streaming_TV"]))
m7=print('Streaming Movies',mode(data['Streaming_Movies']))
m8=print('Streaming Music', mode(data['Streaming_Music']))
m9=print('Unlimited_Data',mode(data['Unlimited_Data']))


# In[7]:


def unique_values_names(df):
    for column in df:
        if df[column].dtype=='object':
            print(f'{column}:{df[column].unique()}')
            
unique_values_names(data)


# In[8]:


data.drop("Churn_Category", axis='columns',inplace= True)
data.drop("Churn_Reason", axis='columns',inplace= True)


# In[9]:


classifications=data['Customer_Status'].unique()
print(classifications)


# In[10]:


data.columns


# In[11]:


X=data[data.Customer_Status=='Stayed']
Y=data[data.Customer_Status=='churned']
Z=data[data.Customer_Status=='Joined']


# In[12]:


Stayed_count=(data['Customer_Status']=='Stayed').sum()
churned_count=(data['Customer_Status']=='Churned').sum()
joined_count=(data['Customer_Status']=='Joined').sum()
print('Stayed_count',Stayed_count)
print("Churned_count",churned_count)
print("joined_count",joined_count)


# In[13]:


import seaborn as sns

sns.countplot(x = 'Customer_Status',data = data)


# In[14]:


data["Avg_Monthly_Long_Distance_Charges"].fillna(25.42, inplace = True)
data['Multiple_Lines'].fillna('No',inplace=True)
data['Internet_Type'].fillna('Fiber Optic',inplace=True)
data['Avg_Monthly_GB_Download'].fillna(26.18,inplace=True)
data['Online_Security'].fillna('No',inplace=True)
data['Online_Backup'].fillna('No',inplace=True)
data['Device_Protection_Plan'].fillna('No',inplace=True)
data['Premium_Tech_Support'].fillna('No',inplace=True)
data['Streaming_TV'].fillna('No',inplace=True)
data['Streaming_Movies'].fillna('No',inplace=True)
data['Streaming_Music'].fillna('No',inplace=True)
data['Unlimited_Data'].fillna('Yes',inplace=True)


# In[15]:


data.shape


# In[89]:


#Feature selection for numerical variables
data.drop("Gender", axis='columns',inplace=True)
data.drop("Married", axis='columns',inplace=True)
data.drop("City", axis='columns',inplace=True)
data.drop("Offer", axis='columns',inplace=True)
data.drop("Phone_Service", axis='columns',inplace=True)
data.drop("Multiple_Lines", axis='columns',inplace=True)
data.drop("Internet_Service", axis='columns',inplace=True)
data.drop("Internet_Type", axis='columns',inplace=True)
data.drop("Online_Security", axis='columns',inplace=True)
data.drop("Online_Backup", axis='columns',inplace=True)
data.drop("Device_Protection_Plan", axis='columns',inplace=True)
data.drop("Premium_Tech_Support", axis='columns',inplace=True)
data.drop("Streaming_TV", axis='columns',inplace=True)
data.drop("Streaming_Movies", axis='columns',inplace=True)
data.drop("Streaming_Music", axis='columns',inplace=True)
data.drop("Unlimited_Data", axis='columns',inplace=True)
data.drop("Contract", axis='columns',inplace=True)
data.drop("Paperless_Billing", axis='columns',inplace=True)
data.drop("Payment_Method", axis='columns',inplace=True)



#Feature Selection

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


X = data.iloc[:,:-1]
Y = data.iloc[:,-1]

#Create an SelectKBest object to select features with two best ANOVA F-Values */
fvalue_selector = SelectKBest(f_classif, k=10)

#Apply the SelectKBest object to the features and target */
X_kbest = fvalue_selector.fit_transform(X, Y)


# In[103]:


# Get a boolean mask indicating which features were selected
selected_features_mask = fvalue_selector.get_support()

# Use the mask to filter the column names of the original DataFrame 'X'
selected_feature_names = X.columns[selected_features_mask]

# Print the names of the selected features
print("Selected Feature Names:")
print(selected_feature_names)


# In[91]:


data.dtypes


# In[104]:


data.columns


# In[61]:


#feature selection for categorical variables
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder

# Select the categorical column to encode
categorical_columns = ['Gender', 'Married', 'Offer', 'Phone_Service', 'Multiple_Lines', 
                    'Internet_Service', 'Internet_Type', 'Online_Security', 'Online_Backup', 
                    'Device_Protection_Plan', 'Premium_Tech_Support', 'Streaming_TV', 'Streaming_Movies', 
                    'Streaming_Music', 'Unlimited_Data', 'Contract', 'Paperless_Billing', 'Payment_Method']
  

label_encoders = {}
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col]
y_encoded = label_encoder.fit_transform(y)

# Separate the features and the target variable
X = data[categorical_columns]  # Features
y =data['Customer_Status']

# Perform the chi-squared test for feature selection
k = 5  # You can change this value to select the desired number of top features
chi2_selector = SelectKBest(chi2, k=10)
X_new = chi2_selector.fit_transform(X, y)

# Access the selected features
selected_features = X.columns[chi2_selector.get_support()]

print("Selected features:")
print(selected_features)


# In[17]:


#Churned_numerical_Var
import seaborn as sns
num=['Age', 'Number_of_Dependents', 'Number_of_Referrals',
       'Tenure_in_Months', 'Avg_Monthly_GB_Download', 'Monthly_Charge',
       'Total_Charges', 'Total_Refunds', 'Total_Long_Distance_Charges',
       'Total_Revenue']

df_churned = data.loc[data['Customer_Status'] == "Churned"]
fig = plt.figure(figsize = (25, 25))

i = 1
for x in num:
    plt.subplot(4, 3, i)
    ax = sns.histplot(x = df_churned[x], color = '#E05252')
    ax.set(xlabel = None, ylabel = None)
    plt.title(str(x), loc = 'center')
    plt.xticks(rotation = 20, fontsize = 10)
    i += 1


# In[20]:


#Stayed_Numerical_Var
num=['Age', 'Number_of_Dependents', 'Number_of_Referrals',
       'Tenure_in_Months', 'Avg_Monthly_GB_Download', 'Monthly_Charge',
       'Total_Charges', 'Total_Refunds', 'Total_Long_Distance_Charges',
       'Total_Revenue']

df_stayed= data.loc[data['Customer_Status'] == "Stayed"]
fig = plt.figure(figsize = (25, 25))

i = 1
for x in num:
    plt.subplot(4, 3, i)
    ax = sns.histplot(x = df_stayed[x], color = '#E05252')
    ax.set(xlabel = None, ylabel = None)
    plt.title(str(x), loc = 'center')
    plt.xticks(rotation = 20, fontsize = 10)
    i += 1


# In[18]:


#Churned_categorical variable
cat=['Married', 'Offer', 'Multiple_Lines', 'Online_Security',
       'Online_Backup', 'Device_Protection_Plan', 'Premium_Tech_Support',
       'Contract', 'Paperless_Billing', 'Payment_Method']
fig = plt.figure(figsize = (25, 25))
i=1
for x in cat:
    plt.subplot(5, 5, i)
    ax = df_churned.groupby([x]).size().plot(kind = 'pie', autopct = '%.2f%%', colors = ['#FF5959', '#676FA3','#CDDEFF', '#EEF2FF'])
    ax.set(xlabel = None, ylabel = None)
    plt.title(str(x), loc='center')
    i+=1
plt.show()


# In[19]:


#Stayed_categorical variable
cat=['Married', 'Offer', 'Multiple_Lines', 'Online_Security',
       'Online_Backup', 'Device_Protection_Plan', 'Premium_Tech_Support',
       'Contract', 'Paperless_Billing', 'Payment_Method']
fig = plt.figure(figsize = (25, 25))
i=1
for x in cat:
    plt.subplot(5, 5, i)
    ax = df_stayed.groupby([x]).size().plot(kind = 'pie', autopct = '%.2f%%', colors = ['#FF5959', '#676FA3','#CDDEFF', '#EEF2FF'])
    ax.set(xlabel = None, ylabel = None)
    plt.title(str(x), loc='center')
    i+=1
plt.show()


# In[16]:


#After feature selection
cols_to_drop=['Gender','City','Zip_Code','Latitude','Longitude','Phone_Service','Avg_Monthly_Long_Distance_Charges',
           'Internet_Service','Internet_Type','Streaming_TV', 'Streaming_Movies','Streaming_Music', 'Unlimited_Data',
          'Total_Extra_Data_Charges','Offer']

data_1 = data.drop(columns=cols_to_drop)

data_1.head()


# In[17]:


def unique_values_names(df):
    for column in df:
        if df[column].dtype=='object':
            print(f'{column}:{df[column].unique()}')
            
unique_values_names(data_1)


# In[18]:


#hot encoding
yes_no_cols=['Married','Multiple_Lines','Online_Security','Online_Backup','Device_Protection_Plan','Premium_Tech_Support',
           'Paperless_Billing']
for col in yes_no_cols:
    data_1[col].replace({'Yes':1,'No':0}, inplace=True)


# In[19]:


for col in yes_no_cols:
    print(f'{col}: {data_1[col].unique()}')


# In[20]:


data_2=pd.get_dummies(data=data_1,columns=['Contract','Payment_Method'])
data_2.columns


# In[21]:


data_2['Customer_Status'].replace({'Stayed':1,'Joined':1,'Churned':0}, inplace=True)


# In[22]:


#Feature scaling
cols_to_scale= ['Age', 'Number_of_Dependents', 'Number_of_Referrals','Tenure_in_Months', 'Avg_Monthly_GB_Download',
                'Monthly_Charge','Total_Charges', 'Total_Refunds', 'Total_Long_Distance_Charges','Total_Revenue']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_2[cols_to_scale] = scaler.fit_transform(data_2[cols_to_scale])
data_2.head()


# In[23]:


data_2.dtypes


#  Select the categorical column to Label encode
# from sklearn.preprocessing import LabelEncoder
# 
# categorical_columns = [ 'Married','Offer' 'Multiple_Lines', 'Online_Security',
#        'Online_Backup', 'Device_Protection_Plan', 'Premium_Tech_Support',
#        'Contract', 'Paperless_Billing', 'Payment_Method','Customer_Status']
#   
# 
# label_encoders = {}
# for col in categorical_columns:
#     label_encoders[col] = LabelEncoder()
#     data_1[col] = label_encoders[col].fit_transform(data_1[col])
#     
# 
# 
# 
# data_1.head()
# 
# 

# In[24]:


data_2.shape


# ## Data Modelling

# In[25]:


data_2.columns


# In[26]:


features=['Age', 'Married', 'Number_of_Dependents', 'Number_of_Referrals',
       'Tenure_in_Months', 'Multiple_Lines', 'Avg_Monthly_GB_Download',
       'Online_Security', 'Online_Backup', 'Device_Protection_Plan',
       'Premium_Tech_Support', 'Paperless_Billing', 'Monthly_Charge',
       'Total_Charges', 'Total_Refunds', 'Total_Long_Distance_Charges',
       'Total_Revenue', 'Contract_Month-to-Month',
       'Contract_One Year', 'Contract_Two Year',
       'Payment_Method_Bank Withdrawal', 'Payment_Method_Credit Card',
       'Payment_Method_Mailed Check']
X = data_2[features]  
Y =data_2['Customer_Status']

# Splitting X and Y
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42, stratify = data_2['Customer_Status'] )


# In[27]:


# Checking dimensions
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)


# In[28]:


len(X_train.columns)


# #### KNN

# In[29]:


# K nearest neighbors Algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 83, metric = 'minkowski', p = 2)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)

Y_pred_knn


# In[30]:


# Support Vector Classifier Algorithm
from sklearn.svm import SVC
svc = SVC(kernel = 'linear', random_state = 42)
svc.fit(X_train, Y_train)


# In[31]:


# Naive Bayes Algorithm
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, Y_train)


# In[32]:


# Decision tree Algorithm
from sklearn.tree import DecisionTreeClassifier
dectree = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
dectree.fit(X_train, Y_train)


# In[33]:


# Random forest Algorithm
from sklearn.ensemble import RandomForestClassifier
ranfor = RandomForestClassifier(n_estimators = 11, criterion = 'entropy', random_state = 42)
ranfor.fit(X_train, Y_train)


# In[34]:


# Making predictions on test dataset
Y_pred_knn = knn.predict(X_test)
Y_pred_svc = svc.predict(X_test)
Y_pred_nb = nb.predict(X_test)
Y_pred_dectree = dectree.predict(X_test)
Y_pred_ranfor = ranfor.predict(X_test)


# In[35]:


# Evaluating using accuracy_score metric
from sklearn.metrics import accuracy_score

accuracy_knn = accuracy_score(Y_test, Y_pred_knn)
accuracy_svc = accuracy_score(Y_test, Y_pred_svc)
accuracy_nb = accuracy_score(Y_test, Y_pred_nb)
accuracy_dectree = accuracy_score(Y_test, Y_pred_dectree)
accuracy_ranfor = accuracy_score(Y_test, Y_pred_ranfor)


# In[36]:


# Accuracy on test set

print("K Nearest neighbors: " + str(accuracy_knn * 100))
print("Support Vector Classifier: " + str(accuracy_svc * 100))
print("Naive Bayes: " + str(accuracy_nb * 100))
print("Decision tree: " + str(accuracy_dectree * 100))
print("Random Forest: " + str(accuracy_ranfor * 100))


# In[ ]:


#From the above comparison, we can observe that Support vector classifier gets the highest accuracy of 85.52 %


# In[38]:


# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred_svc)
cm


# In[39]:


sns.heatmap(pd.DataFrame(cm), annot=True)


# In[39]:


#Classification report
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred_svc))


# In[41]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Nadam

# Define the input shape
input_shape = (23,) 

# Create a sequential model
model = Sequential()

# The first dense layer 
model.add(Dense(16, input_shape=input_shape, kernel_initializer='uniform', activation='relu'))

# Dropout layer with a probability of 0.1
model.add(Dropout(0.1))

# The second dense layer 
model.add(Dense(16, kernel_initializer='uniform', activation='relu'))

# Dropout layer with a probability of 0.1
model.add(Dropout(0.1))


# Final output layer with a single neuron and sigmoid activation
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# Using the Nadam optimizer
model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])


model.summary()


# Splitting the data
model.fit(X_train, Y_train, epochs=100, batch_size=10, validation_data=(X_test, Y_test))


# For testing, set the dropout probability to 0 (ignored during testing)
model.layers[-1].rate = 0

# For prediction, 

# y_pred = model.predict(X_test)


# In[89]:


y_pred = model.predict(X_test)
y_pred


# In[46]:


#Adding one more layer and the optimizer is adam
import tensorflow as tf
from tensorflow import keras

model=keras.Sequential([
    keras.layers.Dense(18,input_shape=(23,),activation='ReLU'),
    keras.layers.Dense(15,activation='ReLU'),
    keras.layers.Dense(12,activation='ReLU'),
    keras.layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train,Y_train, epochs=100)


# In[47]:


# optimizer is adam
model=keras.Sequential([
    keras.layers.Dense(18,input_shape=(23,),activation='ReLU'),
    keras.layers.Dense(15,activation='ReLU'),
    keras.layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train,Y_train, epochs=100)


# In[41]:


#Using nadam optimizer and one more layer - our model

import tensorflow as tf
from tensorflow import keras


model=keras.Sequential([
    keras.layers.Dense(18,input_shape=(23,),activation='ReLU'),
    keras.layers.Dense(15,activation='ReLU'),
    keras.layers.Dense(12,activation='ReLU'),
    keras.layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer='nadam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train,Y_train, epochs=100)


# In[49]:


#Using nadam optimizer 
model=keras.Sequential([
    keras.layers.Dense(18,input_shape=(23,),activation='ReLU'),
    keras.layers.Dense(15,activation='ReLU'),
    keras.layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer='nadam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train,Y_train, epochs=100)


# In[42]:


y_pred = model.predict(X_test)
y_pred[:5]


# In[43]:


Y_pred=[]
for element in y_pred:
    if element >0.5:
        Y_pred.append(1)
    else:
        Y_pred.append(0)
        
Y_pred[:5]


# In[44]:


from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(Y_test,Y_pred))


# In[45]:


import seaborn as sn
cm=tf.math.confusion_matrix(labels=Y_test,predictions=Y_pred)

plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True,fmt='d')
plt.xlabel('predicted')
plt.ylabel('Truth')


# In[ ]:




