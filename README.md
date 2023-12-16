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

## Runtime

Simply running the command `python3 analysis.py` will launch the application and train the seperate classifiers on the dataset provided in `provided_file`.

The images created by the application show visualizations of the code at work.

1. data_distribution.png - Visualizes the differences in positive and negative tweets analyzed in the dataset.
2. naivebayes.png - Visualizes the ROC curve for the Naive Bayes classifier.
3. svm.png - Visuzlies the ROC curve for the Linear SVM classifier.
4. logistic_regression.png - Visualizes the ROC curve for the Logistic Regression Model.

The terminal also outputs the accuracy of each classifier using 5% of the training data as the test data.

