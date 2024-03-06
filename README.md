# Project : Language Detection 
This is my first project in Data Science. The goal was to create a Machine Learning model using NLP to detect the language input. I also created a small web site and deployed it via pythonanywhere.com so everyone can acces it for free as it's a small app.

# App preview 
You can acces to the site here :  [MySite](https://cebsmind.pythonanywhere.com/) 

![preview1](https://github.com/cebsmind/Language_detection/assets/154905924/f61eb00b-c4f3-4c40-8cb0-0de163ed191b)

# Data Set Information
The data set come from HuggingFace [ACCES HERE](https://huggingface.co/datasets/papluca/language-identification#additional-information)
- The Language Identification dataset is a collection of 90k samples consisting of text passages and corresponding language label. This dataset was created by collecting data from 3 sources: Multilingual Amazon Reviews Corpus, XNLI, and STSb Multi MT.
- The Language Identification dataset contains text in 20 languages, which are:
`arabic (ar), bulgarian (bg), german (de), modern greek (el), english (en), spanish (es), french (fr), hindi (hi), italian (it), japanese (ja), dutch (nl), polish (pl), portuguese (pt), russian (ru), swahili (sw), thai (th), turkish (tr), urdu (ur), vietnamese (vi), and chinese (zh)`

# How TF-IDF works ?

![image](https://github.com/cebsmind/Language_detection/assets/154905924/03b4dae6-bfd7-4746-bafc-63ee78c70b3f)


### What is TF-IDF?
TF-IDF (term frequency-inverse document frequency) is a statistical measure that evaluates how relevant a word is to a document in a collection of documents.

This is done by multiplying two metrics: how many times a word appears in a document (**TF**), and the inverse document frequency of the word across a set of documents (**IDF**)

It has many uses, most importantly in automated text analysis, and is very useful for scoring words in machine learning algorithms for Natural Language Processing (NLP).

TF-IDF was invented for document search and information retrieval. It works by increasing proportionally to the number of times a word appears in a document, but is offset by the number of documents that contain the word. So, words that are common in every document, such as **this**, **what**, **and if**, rank low even though they may appear many times, since they don’t mean much to that document in particular.

However, if the word **Bug** appears many times in a document, while not appearing many times in others, it probably means that it’s very relevant.

# How to build a model ?
## Pre process 
First thing we need to do is to process our data so it can be understand by our computer, because computer can't really read text, only numbers.
### Ovieview of our data set : 

![image](https://github.com/cebsmind/Language_detection/assets/154905924/03872b82-dda0-4bb9-bbd5-d41b1ee192ef)

For each sentence we have the language label. 
- First I decided to map every language label to the language name, this can be done like this : 
```python
# Define the mapping dictionary
language_mapping = {
    'ar': 'Arabic',
    'bg': 'Bulgarian',
    'de': 'German',
    'el': 'Modern Greek',
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'hi': 'Hindi',
    'it': 'Italian',
    'ja': 'Japanese',
    'nl': 'Dutch',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'sw': 'Swahili',
    'th': 'Thai',
    'tr': 'Turkish',
    'ur': 'Urdu',
    'vi': 'Vietnamese',
    'zh': 'Chinese'
}

# Map the labels to language names
df_train['labels'] = df_train['labels'].map(language_mapping)
df_val['labels'] = df_val['labels'].map(language_mapping)
df_test['labels'] = df_test['labels'].map(language_mapping)
```

- Next we need to labelize every language to a number from 1-20 (we have 20 languages)
```python
from sklearn.preprocessing import LabelEncoder
# Initialize the label encoder
label_encoder = LabelEncoder()

# Fit and transform the labels in y_train
y_train_encoded = label_encoder.fit_transform(y_train).ravel()

# Transform the labels in y_val and y_test using the same encoder
y_val_encoded = label_encoder.transform(y_val).ravel()
y_test_encoded = label_encoder.transform(y_test).ravel()
```

- Create a function to clean text

```python
def clean_text(text):
    # Lowercasing
    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
    text = re.sub(r'\[', ' ', text)
    return text
```
In this code we decide to remove punctuation and numbers it's irelevant for our language detection, we only need words.

- Vectorize our text data with TF-IDF vectorizer & save it

```python
# Créez une instance de TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit et transformez le jeu de données d'entraînement
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transformez les ensembles de validation et de test en utilisant le même vectoriseur
X_val_tfidf = tfidf_vectorizer.transform(X_val)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Sauvegarder le vectoriseur dans un fichier pour application flask
joblib.dump(tfidf_vectorizer, 'model/TF-IDF.pkl')
```

# Modelisation
For text classification, Naives Bayes Classifier is a really good algorithms

### Overview 
The Naïve Bayes classifier is a supervised machine learning algorithm that is used for classification tasks such as text classification. They use principles of probability to perform classification tasks.
Naïve Bayes is part of a family of generative learning algorithms, meaning that it seeks to model the distribution of inputs of a given class or category. Unlike discriminative classifiers, like logistic regression, it does not learn which features are most important to differentiate between classes.

We can chose the best hyperparameters for this model using a grid search, and evaluate our model on the validation set using accuracy, recall & f1 score. This can be done easily :
```python
# Define the hyperparameter grid
param_grid = {
    'alpha': [0.1, 0.5, 1.0],  # Add more values as needed
    'fit_prior': [True, False],
    # Add more hyperparameters and their possible values
}

# Create the Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()

# Create the GridSearchCV object
grid_search = GridSearchCV(nb_classifier, param_grid, cv=5, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X_train_tfidf, y_train)

# Print the best parameters found by the grid search
print("Best Parameters:", grid_search.best_params_)

# Get the best model from the grid search
best_nb_classifier = grid_search.best_estimator_

# Make predictions on the validation set using the best model
y_val_pred = best_nb_classifier.predict(X_val_tfidf)

# Display the classification report
report = classification_report(y_val, y_val_pred)
print(report)

# You can also access other information from the grid search, such as the best cross-validated score
print("Best Cross-Validated Accuracy:", grid_search.best_score_)
```

We get :

![image](https://github.com/cebsmind/Language_detection/assets/154905924/2e6545e2-9aa1-41f4-8f14-9f0a5f3c47f8)

We can see that our model perform pretty good with a simple approach. However we see that some languages performs poorly. Like `Chinese`, `Japanese` and `Arabic`. This can be explained by the complexity of theses languages, wheras latin language are less complex. So our method has his limits, but it's still a good start.

## Evaluate on test sets
We can now evaluate our model on the test set (a set that our model never seen) 
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Utiliser le même label encoder pour inverser la transformation
y_test_original = label_encoder.inverse_transform(y_test_encoded).ravel()

# Faire des prédictions sur l'ensemble de test
y_test_pred = best_nb_classifier.predict(X_test_tfidf)

# Calculer la matrice de confusion
conf_matrix = confusion_matrix(y_test_original, y_test_pred)

# Afficher la matrice de confusion sous forme de heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=language_classes, yticklabels=language_classes)
plt.title("Matrice de Confusion")
plt.xlabel("Valeurs Prédites")
plt.ylabel("Valeurs Réelles")
plt.show()
```

We get :

![image](https://github.com/cebsmind/Language_detection/assets/154905924/51c065c8-86c6-4f7c-89de-6d4886e613ad)

Not bad ! We see that our model struggled with `Chinese` (he often predict them `Portugese`) and also with `Spanish` he also predict them `Portuguese`
But overall we have a good accuracy.

## Single text predictions
If we want to use our model to predict words we want, we can do it like this : 
```python
#function to predict language
def predict_language(new_text):
    # Prétraiter le nouveau texte de la même manière que vos données d'entraînement
    cleaned_text = clean_text(new_text)

    # Vectoriser le texte avec le même vectoriseur BoW que celui utilisé lors de l'entraînement
    text_vectorized = tfidf_vectorizer.transform([cleaned_text])

    # Faire la prédiction avec le modèle entraîné
    predicted_language = best_nb_classifier.predict(text_vectorized)

    return predicted_language[0]

# Listes de textes de test pour chaque langue
test_texts = {
    'Arabic': 'مرحبًا بك في عالم البرمجة',
    'Bulgarian': 'Здравейте, свят!',
    'German': 'Hallo Welt!',
    'Modern Greek': 'Καλημέρα κόσμε!',
    'English': 'Hello, world!',
    'Spanish': '¡Hola mundo!',
    'French': 'Bonjour tout le monde !',
    'Hindi': 'नमस्ते दुनिया!',
    'Italian': 'Ciao mondo!',
    'Japanese': 'こんにちは、世界！',
    'Dutch': 'Hallo wereld!',
    'Polish': 'Witaj świecie!',
    'Portuguese': 'Olá mundo!',
    'Russian': 'Привет, мир!',
    'Swahili': 'Habari dunia!',
    'Thai': 'สวัสดี, โลก!',
    'Turkish': 'Merhaba dünya!',
    'Urdu': 'ہیلو دنیا!',
    'Vietnamese': 'Chào thế giới!',
    'Chinese': '你好，世界！'
}

true_labels = list(test_texts.keys())
predicted_labels = [predict_language(text) for text in test_texts.values()]

accuracy = accuracy_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels, average='weighted')  # utilisez 'weighted' si les classes ne sont pas équilibrées
precision = precision_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')

print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1 Score:", f1)

```
We get :
- Accuracy: 0.85
- Recall: 0.85
- Precision: 0.775
- F1 Score: 0.8

Some improvement can be done, but this project goal is to start with a basic approach and even though is really simple it achieve good accuracy to predict the language.

### Save model
```python
# Save the best_nb_classifier model to a file
joblib.dump(best_nb_classifier, 'model/LanguageDetection.joblib')
```

# Set up flask app
Set up your folders like this : 
```plaintext
flask-app/
│
├── model/
│   ├── LanguageDetection.joblib
│   └── TF-IDF.pkl
│
├── static/
│   │ 
│   ├── css/
│   │   ├── main.css
│   │   
│   └── js/
│       └── backgroundAnimation.js
│
├── templates/
│      ├── about_me.html
│      └── base.html
│      └── index.html
│
├── main.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
```
#### Install dependencies 
- `python -m venv env`
- `.env/Scripts/activate`
- `pip install -r requirements.txt`
#### Run in terminal 
- `python main.py`
- open http://127.0.0.1:5000/

As you can see, I also created a Dockerfile. Why ? 
The concept of Docker for Data Science is to help developers develop and ship their code easily, in the form of containers. These containers can be deployed anywhere, making the process of setting up a project much simpler, ensuring consistency and reproducibility across different environments.

# Conclusion 
This first project was a first step for using NLP techniques and Machine Learning to predict language detection. We saw that even though it's a simple method it works pretty well. Of course improvement can be done, but the goal was to initiate with Data Science techniques, and model deployment. 
