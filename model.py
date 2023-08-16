# Income prediction model
import pandas as pd
import numpy as np
import pickle


df=pd.read_csv("airline_passenger_satisfaction.csv")

# Drop unnecessary column

df.drop('Unnamed: 0',axis=1,inplace=True)

# Imputation of missing values by linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Split the data into independent and dependent variables
X1 = df[['arrival_delay_in_minutes']]
y1 = df[['departure_delay_in_minutes']]

# Create an imputer object
imputer = SimpleImputer()

# Fit the imputer on the independent variables
imputer.fit(X1)

# Transform the independent variables
X1 = imputer.transform(X1)

# Create a linear regression object
lr = LinearRegression()

# Fit the model on the imputed data
lr.fit(X1, y1)

#  Predict missing values
predicted_values = lr.predict(X1)

# Impute the missing values
df.loc[df['arrival_delay_in_minutes'].isnull().index, 'arrival_delay_in_minutes'] = predicted_values

# Handling outliers using Flooring And Capping Method:

Q1 = df['flight_distance'].quantile(0.25)
Q3 = df['flight_distance'].quantile(0.75)
IQR = Q3 - Q1
whisker_width = 1.5
lower_whisker = Q1 -(whisker_width*IQR)
upper_whisker = Q3 + (whisker_width*IQR)
df['flight_distance']=np.where(df['flight_distance']>upper_whisker,upper_whisker,
                               np.where(df['flight_distance']<lower_whisker,lower_whisker,
                                        df['flight_distance']))

# Handling outliers using Flooring And Capping Method:

Q1 = df['departure_delay_in_minutes'].quantile(0.25)
Q3 = df['departure_delay_in_minutes'].quantile(0.75)
IQR = Q3 - Q1
whisker_width = 1.5
lower_whisker = Q1 -(whisker_width*IQR)
upper_whisker = Q3 + (whisker_width*IQR)
df['departure_delay_in_minutes']=np.where(df['departure_delay_in_minutes']>upper_whisker,upper_whisker,
                                 np.where(df['departure_delay_in_minutes']<lower_whisker,lower_whisker,
                                        df['departure_delay_in_minutes']))

# Handling outliers using Flooring And Capping Method:
Q1 = df['arrival_delay_in_minutes'].quantile(0.25)
Q3 = df['arrival_delay_in_minutes'].quantile(0.75)
IQR = Q3 - Q1
whisker_width = 1.5
lower_whisker = Q1 -(whisker_width*IQR)
upper_whisker = Q3 + (whisker_width*IQR)
df['arrival_delay_in_minutes']=np.where(df['arrival_delay_in_minutes']>upper_whisker,upper_whisker,
                               np.where(df['arrival_delay_in_minutes']<lower_whisker,lower_whisker,
                                        df['arrival_delay_in_minutes']))


# Encoding the categorical data in columns Gender and customer type by One hot encoding

from sklearn.preprocessing import OneHotEncoder
one_hot = OneHotEncoder()
transformed_data = one_hot.fit_transform(df[['Gender','customer_type']].values).toarray()
transformed_data = pd.DataFrame(transformed_data)

# Import label encoding library
from sklearn.preprocessing import LabelEncoder

# Function to apply label encoding
def encode_labels(data):
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
  
    return data
  
# Applying function in whole column of data
df = encode_labels(df)

# Creating a new feature 'total_delay'
df['total_delay_avg'] = (df['departure_delay_in_minutes'] + df['arrival_delay_in_minutes'])/2
df.drop(['departure_delay_in_minutes', 'arrival_delay_in_minutes'], axis=1, inplace=True)

# Calculate the overall facility satisfaction by taking the average of relevant columns
internet_facility_cols = ['inflight_wifi_service', 'online_boarding']  # Add other relevant columns here
df['internet_facility'] = df[internet_facility_cols].mean(axis=1)

# Drop the individual facility satisfaction columns since they are combined into 'overall_facility_satisfaction'
df.drop(internet_facility_cols, axis=1, inplace=True)

# Calculate the overall convenience rating
convenience_cols = ['departure_arrival_time_convenient', 'gate_location', 'ease_of_online_booking', 'checkin_service']
df['overall_convenience_rating'] = df[convenience_cols].mean(axis=1)

# Calculate the overall service quality rating
service_quality_cols = ['inflight_service', 'food_and_drink','cleanliness']
df['overall_service_quality_rating'] = df[service_quality_cols].mean(axis=1)

# Drop the individual convenience, service quality, and cleanliness rating columns
df.drop(convenience_cols + service_quality_cols, axis=1, inplace=True)

# Calculate the overall facility satisfaction by taking the average of relevant columns
overall_experience_cols = ['seat_comfort', 'inflight_entertainment', 'onboard_service', 'leg_room_service', 'baggage_handling']  # Add other relevant columns here
df['overall_experience'] = df[overall_experience_cols].mean(axis=1)

# Drop the individual facility satisfaction columns since they are combined into 'overall_facility_satisfaction'
df.drop(overall_experience_cols, axis=1, inplace=True)

# Create new feature total travel class
df['total_travel_class'] = df['type_of_travel'] + df['customer_class']

# Drop individual column

df.drop(['type_of_travel', 'customer_class'], axis=1, inplace=True)

# Crage new feature Age groups (Under20, 20s, 30s, 40s, 50s,60s,70s, and 80s)

age_group = []

for i in df['age']:
    if i > 0 and i < 20:
        i = 'Under 20'
        age_group.append(i)
   
    elif i >= 20 and i < 30:
        i = '20s'
        age_group.append(i)
    
    elif i >= 30 and i < 40:
        i = '30s'
        age_group.append(i)
        
    elif i >= 40 and i < 50:
        i = '40s'
        age_group.append(i)
        
    elif i >= 50 and i < 60:
        i = '50s'
        age_group.append(i)
    
    elif i >= 60 and i < 70:
        i = '60s'
        age_group.append(i)
        
    elif i >= 70 and i < 80:
        i = '70s'
        age_group.append(i)
        
    elif i >= 80 and i < 90:
        i = '80s'
        age_group.append(i)
        
df['Age_Group'] = age_group

# Drop the original 'age' column
df.drop('age', axis=1, inplace=True)

# Create new feature distance ranges

def categorize_distance(distance):
    if distance < 500:
        return 'Short Haul'
    elif distance < 1000:
        return 'Medium Haul'
    else:
        return 'Long Haul'

df['Distance_Category'] = df['flight_distance'].apply(categorize_distance)

# Drop the original 'flight_distance' column
df.drop('flight_distance', axis=1, inplace=True)

# Import label encoding library
from sklearn.preprocessing import LabelEncoder

# Function to apply label encoding
def encode_labels(data):
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
  
    return data
  
# Applying function in whole column of data
df = encode_labels(df)

# Import standardscaler() library
from sklearn.preprocessing import StandardScaler

# Separate the target variable ('satisfaction') from the features
X = df.drop('satisfaction', axis=1)
y = df['satisfaction']

# Get the column names of the numeric features
numeric_features = X.select_dtypes(include='number').columns

# Create the StandardScaler object
scaler = StandardScaler()

# Standardize the numeric features
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Now the numeric features in X are standardized with mean=0 and std=1

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


from sklearn.linear_model import LogisticRegression
# Using the best parameters of LogisticRegression for building the final model

classifier = LogisticRegression(C=0.5, solver='sag')
classifier.fit(X_train, y_train)


import pickle
#Saving the model to disk
pickle.dump(classifier,open('model.pkl','wb') )
#pickle for converting to byte stream, serialising 
#and deserializing

