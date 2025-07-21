import cv2
import face_recognition
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import time

class AttendanceDatabase:
    def __init__(self, db_path='attendance.db'):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()
    
    def create_tables(self):
        cursor = self.conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS People (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            face_encoding BLOB,
            registration_date TEXT
        )''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS Sessions (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            subject TEXT NOT NULL,
            start_time TEXT,
            duration INTEGER DEFAULT 10
        )''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS Attendance (
            id INTEGER PRIMARY KEY,
            person_id INTEGER,
            session_id INTEGER,
            date TEXT,
            time TEXT,
            status TEXT,
            FOREIGN KEY (person_id) REFERENCES People (id),
            FOREIGN KEY (session_id) REFERENCES Sessions (id)
        )''')
        
        self.conn.commit()  
    
    def register_person(self, name, face_encoding):
        cursor = self.conn.cursor()
        registration_date = datetime.now().strftime("%Y-%m-%d")
        face_encoding = face_encoding.tobytes() if face_encoding is not None else None
        
        cursor.execute(
            "INSERT INTO People (name, face_encoding, registration_date) VALUES (?, ?, ?)",
            (name, face_encoding, registration_date)
        )
        
        self.conn.commit()
    
    def create_session(self, name, subject):
        cursor = self.conn.cursor()
        start_time = datetime.now().strftime("%H:%M:%S")  # Current time
        cursor.execute(
            "INSERT INTO Sessions (name, subject, start_time, duration) VALUES (?, ?, ?, ?)",
            (name, subject, start_time, 10)  # 10-minute duration
        )
        self.conn.commit()
        print(f" Session '{name}' for subject '{subject}' created at {start_time}.")

    def mark_attendance(self, person_id, session_name, subject):
        cursor = self.conn.cursor()
        date = datetime.now().strftime("%Y-%m-%d")
        time_now = datetime.now().strftime("%H:%M:%S")

        # Get session start time & duration
        cursor.execute(
            "SELECT id, start_time, duration FROM Sessions WHERE name = ? AND subject = ?",
            (session_name, subject)
        )
        session = cursor.fetchone()

        if not session:
            print(f" No session found for {session_name} in subject {subject}.")
            return False

        session_id, start_time, duration = session
        session_start = datetime.strptime(start_time, "%H:%M:%S")
        current_time = datetime.strptime(time_now, "%H:%M:%S")

        # Check if within allowed duration
        if (current_time - session_start).total_seconds() > duration * 60:
            print(" Attendance time is over!")
            return False

        # Prevent duplicate attendance
        cursor.execute(
            "SELECT id FROM Attendance WHERE person_id = ? AND session_id = ? AND date = ?",
            (person_id, session_id, date)
        )
        if cursor.fetchone() is None:
            cursor.execute(
                "INSERT INTO Attendance (person_id, session_id, date, time, status) VALUES (?, ?, ?, ?, 'Present')",
                (person_id, session_id, date, time_now)
            )
            self.conn.commit()
            return True
        return False

    def get_attendance_report(self):
        df = pd.read_sql_query('''
        SELECT p.name, s.name AS session, s.subject, a.date, a.time, a.status
        FROM Attendance a
        JOIN People p ON a.person_id = p.id
        JOIN Sessions s ON a.session_id = s.id
        ORDER BY a.date DESC, a.time DESC
        ''', self.conn)
        print(df)
        return df

class AttendanceSystem:
    def __init__(self, db_path='attendance.db'):
        self.db = AttendanceDatabase(db_path)
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
    
    def load_known_faces(self):
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT id, name, face_encoding FROM People")
        
        for person_id, name, encoding in cursor.fetchall():
            if encoding is not None:
                self.known_face_encodings.append(np.frombuffer(encoding, dtype=np.float64))
                self.known_face_names.append((person_id, name))
    
    def register_new_person(self, name):
        cap = cv2.VideoCapture(0)
        encodings = []
        print(f" Capturing images for {name}. Look in different directions.")
        
        for _ in range(40):
            ret, frame = cap.read()
            if not ret:
                continue
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            if face_encodings:
                encodings.append(face_encodings[0])
            
            cv2.imshow("Registering", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if encodings:
            avg_encoding = np.mean(encodings, axis=0)
            self.db.register_person(name, avg_encoding)
            self.known_face_encodings.append(avg_encoding)
            print(f" {name} registered successfully!")
        else:
            print(" No face detected. Try again!")
    
    def take_attendance(self, session_name, subject):
        cap = cv2.VideoCapture(0)
        print(" Taking attendance. Press 'q' to stop.")
        
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT id FROM People")
        registered_ids = {row[0] for row in cursor.fetchall()}  # All registered person IDs
        marked_people = set()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            for face_encoding in face_encodings:
                distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                
                if len(distances) > 0:
                    best_match = np.argmin(distances)
                    if distances[best_match] < 0.6:
                        person_id, name = self.known_face_names[best_match]
                        
                        if person_id not in marked_people:
                            if self.db.mark_attendance(person_id, session_name, subject):
                                marked_people.add(person_id)
                                print(f" {name} marked present.")
                            else:
                                print(f" {name} already marked or too late.")
            
            cv2.imshow("Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = AttendanceSystem()
    
    action = input("Enter 'r' to register, 's' to create a session, 'a' to take attendance, or 'p' to print report: ").strip().lower()
    
    if action == 'r':
        name = input("Enter name: ").strip()
        system.register_new_person(name)
    
    elif action == 's':
        session = input("Enter session name: ").strip()
        subject = input("Enter subject name: ").strip()
        system.db.create_session(session, subject)

    elif action == 'a':
        session = input("Enter session name: ").strip()
        subject = input("Enter subject name: ").strip()
        system.take_attendance(session, subject)
    
    elif action == 'p':
        system.db.get_attendance_report()
