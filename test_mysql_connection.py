import mysql.connector

try:
    conn = mysql.connector.connect(
        host="127.0.0.1",  # utilise "127.0.0.1" au lieu de "localhost"
        user="root",
        password="",       # vide si tu n'as pas défini de mot de passe
        database="ccc"
    )
    print("✅ Connexion réussie !")
    conn.close()
except Exception as e:
    print(f"❌ Erreur de connexion : {e}")
