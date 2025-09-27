import sqlite3
import hashlib

def create_users_table():
    """Creates the users table in the database if it doesn't exist."""
    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT,
            role TEXT
        )
    ''')
    conn.commit()
    conn.close()

def add_user(username, password, role):
    """Adds a new user to the database with a hashed password."""
    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    try:
        c.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)', (username, hashed_password, role))
        conn.commit()
    except sqlite3.IntegrityError:
        print(f"User '{username}' already exists.")
    conn.close()

def authenticate(username, password):
    """Authenticates a user and returns their role if successful, otherwise None."""
    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    c.execute('SELECT role FROM users WHERE username = ? AND password = ?', (username, hashed_password))
    result = c.fetchone()
    conn.close()
    if result:
        return result[0]
    return None

def get_all_users():
    """Returns a list of all users and their roles."""
    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    c.execute('SELECT username, role FROM users')
    users = c.fetchall()
    conn.close()
    return users

def reset_password(username, new_password):
    """Resets the password for a given user."""
    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
    c.execute('UPDATE users SET password = ? WHERE username = ?', (hashed_password, username))
    conn.commit()
    conn.close()
