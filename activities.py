import mysql.connector
import hashlib  # For SHA-256 encryption


def connect_to_database():
    """Establish a connection to the MySQL database."""
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="healthcare_analytics"
    )


def hash_password(password):
    """Hashes a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(username, email, role, password, confirm_password):
    """Register a new user and store their hashed password."""
    if password != confirm_password:
        return {"message": "Passwords do not match!", "status": "error"}

    hashed_password = hash_password(password)  # Hash the password

    conn = connect_to_database()
    cursor = conn.cursor()
    try:
        query = "INSERT INTO users (username, password, email, role) VALUES (%s, %s, %s, %s)"
        cursor.execute(query, (username, hashed_password, email, role))
        conn.commit()
        return {"message": "Registration successful!", "status": "success"}
    except mysql.connector.Error as err:
        return {"message": f"Error: {err}", "status": "error"}
    finally:
        cursor.close()
        conn.close()


def login_user(username, password):
    """Authenticate a user by checking their hashed password."""
    hashed_password = hash_password(password)  # Hash the entered password

    conn = connect_to_database()
    cursor = conn.cursor()
    try:
        query = "SELECT username, role FROM users WHERE username = %s AND password = %s"
        cursor.execute(query, (username, hashed_password))
        user = cursor.fetchone()

        if user:
            return {"message": "Login successful!", "status": "success", "username": user[0], "role": user[1]}
        else:
            return {"message": "Invalid username or password!", "status": "error"}
    finally:
        cursor.close()
        conn.close()


def check_admin_exists():
    """Check if an Admin account already exists."""
    conn = connect_to_database()
    cursor = conn.cursor()
    try:
        query = "SELECT COUNT(*) FROM users WHERE role = 'Admin'"
        cursor.execute(query)
        result = cursor.fetchone()
        return result[0] > 0  # Returns True if an Admin exists
    finally:
        cursor.close()
        conn.close()
