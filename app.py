from flask import Flask, render_template, request
import joblib
import re
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Chemins vers les fichiers
model_path = r'C:\Users\Gatsu\Jupyter\DL\LanguageDetection\LanguageDetection.joblib'  # Modele
vectorizer_path = r'C:\Users\Gatsu\Jupyter\DL\LanguageDetection\count_vectorizer.pkl'  # Vectorizer

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
# Initialiser le vectoriseur BoW
vectorizer = CountVectorizer()

@app.route('/')
def home():
    return render_template('index.html')  # Rendre le modèle 'index.html' qui étend 'base.html'

@app.route('/detect_language', methods=['POST'])
def detect_language():
    if request.method == 'POST':
        text_to_detect = request.form['text_input']
        text_to_detect = clean_text(text_to_detect)

        # Utiliser le vectoriseur chargé pour transformer le texte
        text_vectorized = loaded_vectorizer.transform([text_to_detect])
        # Utilisation du modèle pour prédire la langue
        detected_language = language_model.predict(text_vectorized)[0]
        return render_template('index.html', result=detected_language)

if __name__ == '__main__':
    app.run(debug=True)