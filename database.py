import sqlite3
from datetime import datetime

def create_attendance_table():
    conn = sqlite3.connect('data/attendance.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            username TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def mark_attendance(username):
    conn = sqlite3.connect('data/attendance.db')
    c = conn.cursor()
    c.execute('INSERT INTO attendance (username, timestamp) VALUES (?, ?)', (username, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_attendance(username=None):
    conn = sqlite3.connect('data/attendance.db')
    c = conn.cursor()
    if username:
        c.execute('SELECT * FROM attendance WHERE username=?', (username,))
    else:
        c.execute('SELECT * FROM attendance')
    rows = c.fetchall()
    conn.close()
    return rows

def get_attendance_for_date(target_date):
    """
    Retrieves attendance records for a specific date.
    The date should be in the format 'YYYY-MM-DD'.
    """
    conn = sqlite3.connect('data/attendance.db')
    c = conn.cursor()
    c.execute('SELECT * FROM attendance WHERE timestamp LIKE ?', (f'{target_date}%',))
    rows = c.fetchall()
    conn.close()
    return rows
