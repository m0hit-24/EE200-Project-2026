from fingerprint.engine import build_database, save_database

if __name__ == "__main__":
    db = build_database("data/songs")
    save_database(db, "database.pkl")
    print(f"Indexed database saved with {len(db)} unique hash keys.")
