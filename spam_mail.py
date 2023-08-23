# spammail_predicition
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
# loading the data from csv file to a pandas dataframe
raw_mail_data=pd.read_csv('/content/mail_data.csv')
# replace the null values with a null string
mail_data=raw_mail_data.where((pd.notnull(raw_mail_data)),'')
# label spam mail as 1; ham mail as 0;
mail_data.loc[mail_data['Category']=='spam','Category',]=1
mail_data.loc[mail_data['Category']=='ham','Category',]=0
# seperating data as labels and text
X=mail_data['Message']
Y=mail_data['Category']
# convert test data to feature vectors that can be used as input to the logistic regression

feature_extraction=TfidfVectorizer(min_df=1,stop_words='english',lowercase='True')

X_train_features=feature_extraction.fit_transform(X_train)
X_test_features=feature_extraction.transform(X_test)

# convert Y_train and Y_test values as integers
Y_train=Y_train.astype('int')
Y_test=Y_test.astype('int')
model=LogisticRegression()
# training the Logistic regression model with training data
model.fit(X_train_features,Y_train)
prediction_on_test_data=model.predict(X_test_features)
accuracy_on_test_data=accuracy_score(Y_test,prediction_on_test_data)
from sklearn.naive_bayes import MultinomialNB
NB_classifier=MultinomialNB()
NB_classifier.fit(X_train_features,Y_train)
y_predict_test=NB_classifier.predict(X_test_features)
accuracy_on_test_data=accuracy_score(Y_test,y_predict_test)
print('Accuracy on test data:',accuracy_on_test_data)

# sample case 
input_mail=["I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times."]

#convert test to feature vectors
input_data_features=feature_extraction.transform(input_mail)

#making prediction

prediction=model.predict(input_data_features)
print(prediction)

if (prediction[0]==0):
  print('Ham mail')
else:
  print('Spam mail')
