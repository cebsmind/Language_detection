from flask import Flask, render_template, request
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer

import os

app = Flask(__name__)

# Chemins vers les fichiers (utilisation de chemins relatifs)
model_path = os.path.join(os.path.dirname(__file__), 'model', 'LanguageDetection.joblib')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'model', 'TF-IDF.pkl')

# Importer le modèle
language_model = joblib.load(model_path)

# Charger le vectoriseur à partir du fichier
loaded_vectorizer = joblib.load(vectorizer_path)

# Définir les fonctions de pre process
def clean_text(text):
    # Lowercasing
    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
    text = re.sub(r'\[', ' ', text)
    return text

# Importation de CountVectorizer
# Initialiser le vectoriseur TF-IDF
vectorizer = TfidfVectorizer()

# History to store sentences and their predicted languages
sentence_history = []

@app.route('/')
def home():
    return render_template('index.html', sentence_history=sentence_history)

@app.route('/about_me')
def about_me():
    # page "About Me"
    return render_template('about_me.html')

@app.route('/detect_language', methods=['POST'])
def detect_language():
    global sentence_history  # Add this line to make the variable global

    if request.method == 'POST':
        text_to_detect = request.form['text_input']
        text_to_detect_cleaned = clean_text(text_to_detect)

        # Utiliser le vectoriseur chargé pour transformer le texte
        text_vectorized = loaded_vectorizer.transform([text_to_detect_cleaned])
        # Utilisation du modèle pour prédire la langue
        detected_language = language_model.predict(text_vectorized)[0]

        # Clear the history before adding the new entry
        sentence_history = []

        # Add the input sentence and its prediction to the history
        sentence_history.append({'input_sentence': text_to_detect, 'predicted_language': detected_language})

        return render_template('index.html', result=detected_language, sentence_history=sentence_history)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)