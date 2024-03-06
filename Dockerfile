# Utilisez une image Python officielle comme image de base
FROM python:3.10

# Définissez le répertoire de travail à /app
WORKDIR /app

# Copiez les fichiers nécessaires dans le conteneur
COPY . .

# Installez les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Commande à exécuter lorsque le conteneur démarre
CMD ["python", "main.py"]