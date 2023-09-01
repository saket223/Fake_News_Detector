import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')

# printing stopwords in English
print(stopwords.words('english'))

# Data-Preprocessing
# loading the dataset
news_dataset = pd.read_csv('fake_or_real_news.csv')

# print the first 5 rows of the dataset
news_dataset.head()

# counting the number of missing valuesin the dataset
news_dataset.isnull().sum()


# replacing the null values with empty strings
news_dataset = news_dataset.fillna('')

#merging the author name and news title
news_dataset['content'] = news_dataset['text']

print(news_dataset['content'])

# separating the data and the label
X = news_dataset.drop(columns=['label'],axis=1)
Y = news_dataset['label']

# Stemming: It is the process of reducing a word to its root word.(remove␣suffix and prefix)
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)

print(news_dataset['content'])

#separating the data and label
X = news_dataset['content'].values
Y = news_dataset['label'].values

# converting the textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

# Splitting the dataset to training & test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,stratify=Y, random_state=42)

clf_dt=DecisionTreeClassifier()
clf_dt=clf_dt.fit(X_train, Y_train)

predictions_dt = clf_dt.predict(X_test)
test_data_accuracy_dt = accuracy_score(predictions_dt, Y_test)
test_data_accuracy_dt = round(test_data_accuracy_dt, 3)
print('Accuracy score of the test data : ', test_data_accuracy_dt)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print(classification_report(Y_test, predictions_dt))
mat=confusion_matrix(Y_test, predictions_dt)
print(confusion_matrix(Y_test, predictions_dt))

# Making a Predictive System
X_new = X_test[3]
prediction = clf_dt.predict(X_new)
print(prediction)
if (prediction[0]=='REAL'):
    print('The news is Real')
else:
    print('The news is Fake')   


classes = ['Real','Fake']

cm_df = pd.DataFrame(mat, index = classes,columns = classes)
plt.figure(figsize = (5,5))
sns.set(font_scale=1.2)
sns.heatmap(cm_df, annot = True,cbar=False,linewidth=2,fmt='d',cmap="GnBu")
plt.title('Confusion Matrix for Decision Tree Classifier')
plt.ylabel('Original Values')
plt.xlabel('Predicted Values')
plt.savefig('Cm_dt.png')
plt.show()

import joblib
joblib.dump(clf_dt, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')





