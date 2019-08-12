import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# cleaning texts
import nltk
import re
nltk.download('stopwords')

corpus = []

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# creating the corpus by pre-processing the data
for i in range(0, len(dataset['Review'])):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(
        stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
# create your sparse matrix with all words from reviews
cv = CountVectorizer(max_features=1500)

# create your independent variable matrix
X = cv.fit_transform(corpus).toarray()
# create your dependent variable matrix
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

#compare different classification models
metrics = [
        'Accuracy',
        'Precision',
        'Recall',
        'F1 Score'
]
classifierArray = [
    'GaussianNB',
    'Bernoulli_NB',
    'Mutilnomial_NB',
    'DECISION_TREE',
    'RANDOM_FOREST',
    'Sigmoid_SVM',
    'Gaussian_SVM'
]

comparison_matrix = {
        'GaussianNB':{
            'Accuracy':[],
            'Precision':[],
            'Recall':[],
            'F1 Score':[]

        },
        'Bernoulli_NB':{
            'Accuracy':[],
            'Precision':[],
            'Recall':[],
            'F1 Score':[]

        },
        'Mutilnomial_NB':{
            'Accuracy':[],
            'Precision':[],
            'Recall':[],
            'F1 Score':[]

        },
        'DECISION_TREE':{
            'Accuracy':[],
            'Precision':[],
            'Recall':[],
            'F1 Score':[]

        },
        'RANDOM_FOREST':{
            'Accuracy':[],
            'Precision':[],
            'Recall':[],
            'F1 Score':[]

        },
        'Sigmoid_SVM':{
            'Accuracy':[],
            'Precision':[],
            'Recall':[],
            'F1 Score':[]

        },
        'Gaussian_SVM':{
            'Accuracy':[],
            'Precision':[],
            'Recall':[],
            'F1 Score':[]

        }
}


def classifierFunc():
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    tp, tn, fp, fn = cm[1,1], cm[0,0], cm[0,1], cm[1,0]

    Accuracy = round((tp+tn)/(tp+tn+fp+fn)*100, 1)
    Precision = round(tp / (tp + fp)*100,1)
    Recall = round(tp / (tp + fn)*100,1)
    F1_Score = round(2 * Precision * Recall / (Precision + Recall),2)

    comparison_matrix[index][metrics[0]].append(Accuracy)
    comparison_matrix[index][metrics[1]].append(Precision)
    comparison_matrix[index][metrics[2]].append(Recall)
    comparison_matrix[index][metrics[3]].append(F1_Score)

def scaler():
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    global X_train 
    X_train = sc.fit_transform(X_train)
    global X_test 
    X_test = sc.transform(X_test)

for index in classifierArray:
    if (index == 'GaussianNB'):
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifierFunc()

    elif (index == 'Bernoulli_NB'):
        from sklearn.naive_bayes import BernoulliNB
        classifier = BernoulliNB()
        classifierFunc()

    elif (index == 'Mutilnomial_NB'):
        from sklearn.naive_bayes import MultinomialNB
        classifier = MultinomialNB()
        classifierFunc()

    elif (index == 'DECISION_TREE'):
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
        classifierFunc()
        
    elif (index == 'RANDOM_FOREST'):
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators=300, criterion='entropy',random_state=0)
        classifierFunc()

    elif (index == 'Sigmoid_SVM'):
        from sklearn.svm import SVC
        classifier = SVC(kernel='sigmoid', random_state=0)
        scaler()
        classifierFunc()

    elif (index == 'Gaussian_SVM'):
        from sklearn.svm import SVC
        classifier = SVC(kernel='rbf', random_state=0)
        scaler()
        classifierFunc()

#Create dataframe
dataframe = pd.DataFrame(comparison_matrix)
#dataframe.to_csv('Model_performance.csv')