function clearHistory() {
    console.log("Clearing history and redirecting to home");
    sessionStorage.clear();
    window.location.href = "/";
}