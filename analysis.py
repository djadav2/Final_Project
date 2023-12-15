# utilities
import re
import numpy as np
import pandas as pd
import string
# plotting
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# nltk
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
# sklearn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
             's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']
STOPWORDS = set(stopwordlist)

def main():

    DATASET_COLUMNS=['target','ids','date','flag','user','text']
    DATASET_ENCODING = "ISO-8859-1"
    df = pd.read_csv('training.csv', encoding=DATASET_ENCODING, names=DATASET_COLUMNS)

    df.head()

    # Plotting the distribution for dataset.
    ax = df.groupby('target').count().plot(kind='bar', title='Distribution of data',legend=False)
    ax.set_xticklabels(['Negative','Positive'], rotation=0)
    # Storing data in lists.
    text, sentiment = list(df['text']), list(df['target'])
    sns.countplot(x='target', data=df)
    plt.savefig("data_distribution.png")
    plt.show(block=False)

    # Preprocessing Data
    data=df[['text','target']]
    data['target'] = data['target'].replace(4,1)
    data_p = data[data['target'] == 1]
    data_n = data[data['target'] == 0]

    data_pos = data_p.iloc[:int(20000)]
    data_neg = data_n.iloc[:int(20000)]

    # Combining Dataset
    dataset = pd.concat([data_pos, data_neg])

    dataset['text']=dataset['text'].str.lower()
    dataset['text'].tail()

    # Cleaning Stopwords
    dataset['text'] = dataset['text'].apply(lambda text: cleaning_stopwords(text))
    dataset['text'].head()

    english_punctuations = string.punctuation
    punctuations_list = english_punctuations

    # Removing punctuation
    dataset['text']= dataset['text'].apply(lambda x: cleaning_punctuations(x, punctuations_list))
    dataset['text'].tail()

    # Removing repeat characters
    dataset['text'] = dataset['text'].apply(lambda x: cleaning_repeating_char(x))
    dataset['text'].tail()

    # Removing URLs
    dataset['text'] = dataset['text'].apply(lambda x: cleaning_URLs(x))
    dataset['text'].tail()

    # Cleaning numbers
    dataset['text'] = dataset['text'].apply(lambda x: cleaning_numbers(x))
    dataset['text'].tail()

    # Tokenization of tweet text
    tokenizer = RegexpTokenizer(r'w+')
    dataset['text'] = dataset['text'].apply(tokenizer.tokenize)
    dataset['text'].head()

    # Stemming
    st = nltk.PorterStemmer()
    dataset['text']= dataset['text'].apply(lambda x: stemming_on_text(x, st))
    dataset['text'].head()

    # Lemmatizer
    lm = nltk.WordNetLemmatizer()
    dataset['text'] = dataset['text'].apply(lambda x: lemmatizer_on_text(x, lm))
    dataset['text'].head()

    X=data.text
    y=data.target

    # Separating the 95% data for training data and 5% for testing data
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.05, random_state =26105111)
    vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
    vectoriser.fit(X_train)
    print('No. of feature_words: ', len(vectoriser.get_feature_names_out()))

    # Transform the Vectorizer
    X_train = vectoriser.transform(X_train)
    X_test  = vectoriser.transform(X_test)

    # Evaluate Model-1 (Naive Bayes)
    BNBmodel = BernoulliNB()
    BNBmodel.fit(X_train, y_train)
    model_Evaluate(BNBmodel, X_test, y_test)
    y_pred1 = BNBmodel.predict(X_test)

    # Plot the ROC-AUC Curve for model-1
    fpr, tpr, thresholds = roc_curve(y_test, y_pred1)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC CURVE')
    plt.legend(loc="lower right")
    plt.savefig("naivebayes.png")
    plt.show(block=False)

    # Evaluate Model-2 (Support Vector Machine)
    SVCmodel = LinearSVC()
    SVCmodel.fit(X_train, y_train)
    model_Evaluate(SVCmodel, X_test, y_test)
    y_pred2 = SVCmodel.predict(X_test)

    # Plot the ROC-AUC Curve for model-2
    fpr, tpr, thresholds = roc_curve(y_test, y_pred2)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC CURVE')
    plt.legend(loc="lower right")
    plt.savefig("svm.png")
    plt.show(block=False)

    # Evaulate model-3 (Logistic Regression)
    LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
    LRmodel.fit(X_train, y_train)
    model_Evaluate(LRmodel, X_test, y_test)
    y_pred3 = LRmodel.predict(X_test)

    # Plot the ROC-AUC Curve for model-3
    fpr, tpr, thresholds = roc_curve(y_test, y_pred3)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC CURVE')
    plt.legend(loc="lower right")
    plt.savefig("logistic_regression.png")
    plt.show(block=False)

def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

def cleaning_punctuations(text, punctuations_list):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)

def cleaning_repeating_char(text):
    return re.sub(r'(.)1+', r'1', text)

def cleaning_URLs(data):
    return re.sub('((www.[^s]+)|(https?://[^s]+))',' ',data)

def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)

def stemming_on_text(data, st):
    text = [st.stem(word) for word in data]
    return data

def lemmatizer_on_text(data, lm):
    text = [lm.lemmatize(word) for word in data]
    return data

def model_Evaluate(model, X_test, y_test):
    # Predict values for Test dataset
    y_pred = model.predict(X_test)
    # Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
    xticklabels = categories, yticklabels = categories)
    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)

if __name__ == "__main__":
    main()
