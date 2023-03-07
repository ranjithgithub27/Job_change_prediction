# -*- coding: utf-8 -*-
"""
19,159 datas for train
2,130 datas for test

@author: DELL
"""

import pandas as pd1
import pandas as pd2

df1 = pd1.read_csv("aug_train.csv")
df2 = pd2.read_csv("aug_test.csv")

#To know basic information
'''print(df1.head())
print(df2.head())
print(df1.info())
print(df2.info())'''
#print(df1.duplicated().sum())
#print(df1.isnull().sum())


# Filling null values and replacing char to int
df1['gender'] = df1['gender'].fillna("Male")
df1['enrolled_university'] = df1['enrolled_university'].fillna("no_enrollment")
df1['education_level'] = df1['education_level'].fillna("Graduate")
df1['major_discipline'] = df1['major_discipline'].fillna("STEM")
df1['experience'] = df1['experience'].fillna("0")
df1['company_type'] = df1['company_type'].fillna("Pvt Ltd")
df1['last_new_job'] = df1['last_new_job'].fillna("never")

df1.loc[df1['gender']=='Male','gender'] = 1
df1.loc[df1['gender']=='Female','gender']= 0
df1.loc[df1['gender']=='Other','gender'] = 2
df1.loc[df1['last_new_job'] == 'never','last_new_job'] = 0
df1.loc[df1['last_new_job'] == '>4','last_new_job'] = 5
df1.loc[df1['experience']== '>20','experience'] = 21
df1.loc[df1['experience']== '<1','experience'] = 0.6

#Assigning dataset for training
x = df1.drop(['city','company_size','target'],axis=1)
x1 = df1['target']


# Filling null values and replacing char to int 
df2['gender'] = df2['gender'].fillna("Male")
df2['enrolled_university'] = df2['enrolled_university'].fillna("no_enrollment")
df2['education_level'] = df2['education_level'].fillna("Graduate")
df2['major_discipline'] = df2['major_discipline'].fillna("STEM")
df2['experience'] = df2['experience'].fillna("0")
df2['company_type'] = df2['company_type'].fillna("Pvt Ltd")
df2['last_new_job'] = df2['last_new_job'].fillna("never")

df2.loc[df2['gender']=='Male','gender'] = 1
df2.loc[df2['gender']=='Female','gender']= 0
df2.loc[df2['gender']=='Other','gender'] = 2
df2.loc[df2['last_new_job'] == 'never','last_new_job'] = 0
df2.loc[df2['last_new_job'] == '>4','last_new_job'] = 5
df2.loc[df2['experience']== '>20','experience'] = 21
df2.loc[df2['experience']== '<1','experience'] = 0.6

#Assigning dataset for testing
y = df2.drop(['city','company_size'] ,axis=1)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#label encoding for trained data
x.relevent_experience = le.fit_transform(x.relevent_experience)
x.enrolled_university = le.fit_transform(x.enrolled_university)
x.education_level = le.fit_transform(x.education_level)
x.company_type = le.fit_transform(x.company_type)
x.major_discipline = le.fit_transform(x.major_discipline)

#label encoding for test data
y.relevent_experience = le.fit_transform(y.relevent_experience)
y.enrolled_university = le.fit_transform(y.enrolled_university)
y.education_level = le.fit_transform(y.education_level)
y.company_type = le.fit_transform(y.company_type)
y.major_discipline = le.fit_transform(y.major_discipline)


from sklearn.neighbors import KNeighborsClassifier
alg = KNeighborsClassifier(n_neighbors=3)
alg.fit(x, x1)
ypred = alg.predict(y)
print(ypred)









