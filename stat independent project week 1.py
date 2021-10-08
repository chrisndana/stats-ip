#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Defining question

# In[ ]:


# question
how to determine the top 5 kind of jobs in which people who have a bank account have been doing or employed
# metic of sucesss
Theta will help us in  measuring certain aspects of our oparation to achieve success


# # Data reading

# In[2]:


df=pd.read_csv("Financial Dataset - 1.csv")
df.head()


# In[3]:


df.columns


# In[4]:


# changing column names
df1=df.rename(columns={'Has a Bank account':'Has_a_Bank_account', 'Type of Location':'Type_of_Location','Cell Phone Access':'Cell_Phone_Access', 'Respondent Age':'Respondent_Age', 'The relathip with head':'The_relatioship_with_head','Level of Educuation':'Level_of_Educuation','Type of Job':'Type_of_Job'})
df1.head()


# In[5]:


df2=df1.drop(columns='uniqueid')
df2.head()


# In[6]:


df2.isnull().sum()


# In[7]:


df3=df2.dropna()
df3.head()


# In[8]:


df3.isnull().sum()


# In[9]:


df3.duplicated().sum()


# In[10]:


df4=df3.drop_duplicates()
df4.head()


# In[11]:


df4.duplicated().sum()


# In[12]:


df5=df4.drop(columns='year')
df5.head()


# # Detecting and removing outliers

# In[13]:


# ploting boxplot to identify outliers
fig,((ax1,ax2))=plt.subplots(2,1,figsize=(10,7))
fig.suptitle('Boxplots')
sns.boxplot(df4['household_size'],ax=ax1)
sns.boxplot(df4['Respondent_Age'],ax=ax2)


# In[14]:


# dropping outliers using the upper and lower limit method
upper_limit=df4.household_size.mean()+3*df4.household_size.std()
upper_limit


# In[15]:


lower_limit=df4.household_size.mean()-3*df4.household_size.std()
lower_limit


# In[16]:


df4[(df4.household_size>upper_limit)|(df4.household_size<lower_limit)]


# In[17]:


df6=df4[(df4.household_size<upper_limit)&(df4.household_size>lower_limit)]
df6.head()


# # Univariate analysis
# 

# In[18]:


df6.describe()


# In[19]:


df6.skew()


# In[20]:


sns.countplot(x='Has_a_Bank_account',data=df6)
plt.title('The number of people who have bank account in the three countries')
plt.ylabel('value count')
plt.xlabel('Have a bank account or not')


# # Bivariate analysis

# In[21]:


sns.pairplot(df6)


# In[22]:


corelation=df6.corr()
sns.heatmap(corelation,xticklabels=corelation.columns,yticklabels=corelation.columns,annot=True)


# In[23]:


df7=df6.drop(columns='year')
df7.head()


# # Perfoming principal component analysis

# In[24]:


from numpy import asarray
from sklearn.preprocessing import OrdinalEncoder
# data=asarray([['red'],['blue'],['green']])
encoder=OrdinalEncoder()
df8=encoder.fit_transform(df7)
df8


# In[25]:


df9=pd.DataFrame(df8)
df9.head(20)


# In[26]:


df10=df9.rename(columns={0:'country', 1:'Has_a_Bank_account', 2:'Type_of_Location',3:'Cell_Phone_Access', 4:'household_size', 5:'Respondent_Age',6:'gender_of_respondent', 7:'The_relatioship_with_head',8:'marital_status',9:'Level_of_Educuation', 10:'Type_of_Job'})
df10


# In[27]:


x=df10.drop(['Has_a_Bank_account','gender_of_respondent','The_relatioship_with_head','marital_status','country','Type_of_Location','Cell_Phone_Access'],axis='columns')
y=df10['Has_a_Bank_account']


# In[28]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=0)


# In[29]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[30]:


from sklearn.decomposition import PCA
pca=PCA()
x_train=pca.fit_transform(x_train)
x_test=pca.transform(x_test)


# In[31]:


explained_variance=pca.explained_variance_ratio_
explained_variance


# In[32]:


from sklearn.decomposition import KernelPCA
kpca=KernelPCA(n_components=3,kernel='rbf')
x_train=kpca.fit_transform(x_train)
x_test=kpca.transform(x_test)


# In[33]:


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(max_depth=2,random_state=0)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)


# In[34]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
print('Accuracy' , accuracy_score(y_test,y_pred))


# In[37]:


# testing the accuracy using the 2 principal components
from sklearn.decomposition import KernelPCA
kpca=KernelPCA(n_components=2,kernel='rbf')
x_train=kpca.fit_transform(x_train)
x_test=kpca.transform(x_test)


# In[41]:


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(max_depth=2,random_state=0)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)


# In[43]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
print('Accuracy',accuracy_score(y_test,y_pred))


# # Implementing solution

# In[61]:


df12=df6.drop(['year','Type_of_Location','Cell_Phone_Access','household_size','Respondent_Age','gender_of_respondent','The_relatioship_with_head','marital_status','Level_of_Educuation'],axis='columns')
df12.head()


# In[62]:


df12.groupby(['Has_a_Bank_account','Type_of_Job']).count()
# from the below analysis its clear that the leading people in the 3 countris who dont have bank account are the informally employed
# the country whose citizens are self employed  are the oone who have opened bank account mostly


# # Recomendation and documantation

# From the various analysis done before its clear that the type of job affect whether the person will have a bank or not have a bank account
# Also inorder to determine whether a person has a Bank  or not we can use 3 components since the first three components (pc1=34.71%,pc2=28.61% and pc3=21.66%) which totals to approximately 84.98% of prediction of people who have bank account or not

# # Follow up questions

# Did i have the right data

# Yes i had the right data to answer the question that i had formulated earleir

# Did i have the right Question:
# Yes i had the right question since i was able to perfom analysis and get a solution to it at the end of the project

# In[ ]:




