document.addEventListener("DOMContentLoaded", function () {
    // Sélectionnez le corps du document
    const body = document.body;

    // Initialisez la position et la vitesse du défilement
    let position = 0;
    const speed = 15; // Ajustez la vitesse du défilement selon vos préférences

    // Fonction pour animer le fond
    function animateBackground() {
        // Mettez à jour la position en fonction de la vitesse
        position += speed;

        // Appliquez la nouvelle position au fond du corps
        body.style.backgroundPosition = position + "px 0";

        // Demandez une nouvelle animation
        requestAnimationFrame(animateBackground);
    }

    // Lancez l'animation lors du chargement du contenu DOM
    animateBackground();
});