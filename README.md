# Project : Language Detection 
This marks my inaugural venture into the realm of Data Science. The primary objective was to craft a Machine Learning model utilizing Natural Language Processing (NLP) for the purpose of language detection. In tandem with this, a compact website was developed and subsequently deployed on pythonanywhere.com, ensuring free accessibility to all users given the modest scale of the application.

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
The initial step in constructing our model involves processing the data to make it understandable to our computer. Computers inherently struggle with interpreting raw text, necessitating conversion into a numerical format.

### Ovieview of our data set : 

![image](https://github.com/cebsmind/Language_detection/assets/154905924/03872b82-dda0-4bb9-bbd5-d41b1ee192ef)

Each sentence in our dataset is accompanied by a language label.

The first preprocessing task involves mapping each language label to its respective language name. This can be achieved as follows:

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

Subsequently, we proceed to label each language with a corresponding numerical identifier ranging from 1 to 20, as there are a total of 20 languages in our dataset.

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

Develop a function to cleanse the text data

```python
def clean_text(text):
    # Lowercasing
    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
    text = re.sub(r'\[', ' ', text)
    return text
```
In the following code, we opt to eliminate punctuation and numbers as they are deemed irrelevant for our language detection task. Our focus lies solely on words.

Subsequently, we proceed to vectorize our text data using the TF-IDF vectorizer and save the resulting vectors.

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

# Model Building
In the realm of text classification, the Naive Bayes Classifier emerges as a highly effective algorithm.

### Overview 
The Naïve Bayes classifier is a supervised machine learning algorithm that is used for classification tasks such as text classification. They use principles of probability to perform classification tasks.
Naïve Bayes is part of a family of generative learning algorithms, meaning that it seeks to model the distribution of inputs of a given class or category. Unlike discriminative classifiers, like logistic regression, it does not learn which features are most important to differentiate between classes.

We can select the optimal hyperparameters for our model through a grid search and evaluate its performance on the validation set using metrics such as accuracy, recall, and F1 score. This process can be effortlessly executed as follows:

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

"We observe that our model performs admirably well with a straightforward approach. However, there are certain languages, such as **Chinese**, **Japanese**, and **Arabic**, that exhibit lower performance. This can be attributed to the inherent complexity of these languages, in contrast to the relatively simpler structure of Latin languages. Thus, our method has its limitations, yet it marks a promising beginning

## Evaluate on test sets
Our next step involves evaluating the model on the test set, a dataset that the model has never encountered during training.

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

Not bad! It's evident that our model encounters challenges, particularly with Chinese (frequently misclassified as Portuguese) and Spanish (occasionally misclassified as Portuguese). Despite these challenges, our overall accuracy remains commendable.

## Single text predictions
For individual text predictions, we can employ the following approach:"
 
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

Although there is room for improvement, the primary objective of this project was to initiate with a basic approach. Despite its simplicity, the model has demonstrated commendable accuracy in predicting language.

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

As illustrated, I've also included a Dockerfile. Why?

- The incorporation of Docker in Data Science serves to facilitate developers in crafting and delivering their code seamlessly, encapsulated within containers. These containers are highly versatile, allowing deployment across various environments and streamlining project setup. This fosters consistency and reproducibility.

# Conclusion
This inaugural project served as a foundational step into the realm of Natural Language Processing (NLP) and Machine Learning for language detection. Despite its simplicity, the method employed has demonstrated effectiveness. While there is room for enhancement, the primary goal was to initiate the exploration of Data Science techniques and model deployment.
