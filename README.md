# Final Project

This project aims to analyze Twitter speech analysis using machine learning algorithms by implementing an NLP Twitter speech analysis model that helps to overcome the challenges of sentiment classification of speech. Creating this application is important for companies as they greatly benefit by analyzing the sentiment of customer feedback to identify areas where they need to improve their products or services, help companies monitor their brand reputation online and quickly respond to negative comments or reviews, help political campaigns understand public opinion and tailor their messaging accordingly, help organizations monitor social media and news outlets for negative sentiment and respond appropriately, help marketers understand consumer behavior, and provide many other benefits.

The sample use case provided in the code above performs analysis of the Sentiment140 dataset by developing a machine learning pipeline involving the use of three classifiers (Logistic Regression, Bernoulli Naive Bayes, and SVM) along with using Term Frequency-Inverse Document Frequency (TF-IDF). The performance of these classifiers is then evaluated using accuracy and F1 Scores.

## Dependencies

### Utility
1. Numpy
2. Pandas

### Plotting
1. Seaborn
2. Matplotlib

### NLTK
1. WordNetLemmatizer
2. RegexpTokenizer

### Sklearn
1. BernoulliNB
2. LinearSVC
3. LogisticRegression
4. TfidfVectorizer

## Explanation of Code

The section below details the functionality and usage of the codebase.

### Implementation Details

The codebase does the following (in order):
1. Import Necessary Dependencies
2. Read and Load the Dataset
3. Data Preprocessing - Cleaning stopwords, Removing Punctutation, Removing repeat characters, Removing URLs, Cleaning numbers, Tokenization of text, Stemming the data (reducing words to their base form), and Lemmatizing the data (Grouping words by lemma).
4. Distributing Test and Training Data - 5% test, 95% training data
5. Transforming Dataset using TF-IDF Vectorization - TF-IDF transforms the text into meaningful representation of integers or numbers which is then used to fit machine learning algorithms for predictions.
6. Builds each model
7. Model Evaluation
   
## Runtime

Firstly, the dataset must be loaded into the working directory at the `provided_file` path. In the case above, the dataset was downloaded from [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140) and imported into the local working directory as `training.csv.`

From there, simply running the command `python3 analysis.py` will launch the application and train the seperate classifiers on the dataset provided in `provided_file`.

The images created by the application show visualizations of the code at work.

1. data_distribution.png - Visualizes the differences in positive and negative tweets analyzed in the dataset.
2. naivebayes.png - Visualizes the ROC curve for the Naive Bayes classifier.
3. svm.png - Visuzlies the ROC curve for the Linear SVM classifier.
4. logistic_regression.png - Visualizes the ROC curve for the Logistic Regression Model.

The terminal also outputs the accuracy of each classifier using 5% of the training data as the test data.

## Video

Link to the video where I explain the application: [Google Drive](https://drive.google.com/file/d/1RoYd4eHlA09rqIZBaO6T--TsPnMzer8r/view?usp=sharing)

