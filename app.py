import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import pandas as pd
from datetime import date
import time
import hashlib
import sqlite3

# --- Import local modules with error handling ---
try:
    from auth import authenticate, create_users_table, add_user, get_all_users, reset_password
    from facenet_utils import get_embedding, save_embedding, recognize_face, delete_embedding
    from yolov8_utils import detect_faces
    from database import create_attendance_table, mark_attendance, get_attendance, get_attendance_for_date
except ImportError as e:
    st.error("A critical component failed to load. This might be due to a missing file or a dependency issue.")
    st.error(f"Error: {e}")
    st.info("Please ensure all files are in the correct location and try again.")
    st.stop()


# --- Initial Setup ---
# Ensure required directories and tables exist
os.makedirs('data/embeddings', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
create_users_table()
create_attendance_table()

# Create a default admin user if one doesn't exist
def create_default_admin():
    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ?', ('admin',))
    if c.fetchone() is None:
        # Hashing the password to match the auth.py's add_user function
        hashed_password = hashlib.sha256('admin'.encode()).hexdigest()
        add_user('admin', 'admin', 'Admin')
    conn.close()

# Create default admin user on app startup
create_default_admin()


# Streamlit UI
st.set_page_config(page_title="Face Recognition Attendance System", layout="wide")
st.title("Face Recognition Attendance System")

# Initialize session state for login and webcam
if 'role' not in st.session_state:
    st.session_state['role'] = None
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'camera_running' not in st.session_state:
    st.session_state['camera_running'] = False
if 'present_this_session' not in st.session_state:
    st.session_state['present_this_session'] = []
if 'last_frame_time' not in st.session_state:
    st.session_state['last_frame_time'] = 0


def Admin_Dashboard():
    st.header("Admin Dashboard")
    st.write("Manage users and enroll student faces.")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Add User", "Enroll Student (Upload Photo)", "Enroll Student (Webcam)", "Delete Student Face", "Reset User Password", "View All Attendance"])

    with tab1:
        st.subheader("Add New User")
        with st.form("add_user_form"):
            new_username = st.text_input("New Username", key="new_user")
            new_password = st.text_input("New Password", type="password", key="new_pass")
            new_role = st.selectbox("Role", ["Teacher", "Student"], key="new_role")
            submitted = st.form_submit_button("Add User")
            if submitted and new_username and new_password:
                add_user(new_username, new_password, new_role)
                st.success(f"Added {new_role}: {new_username}")

    with tab2:
        st.subheader("Enroll Student Face (Upload Photo)")
        st.write("Upload a student's photo to enroll their face into the system.")
        student_users = [user[0] for user in get_all_users() if user[1] == 'Student']
        if not student_users:
            st.info("No students registered yet. Please add a new user with the 'Student' role first.")
        else:
            student_to_enroll = st.selectbox("Select Student", student_users, key="upload_select")
            uploaded_file = st.file_uploader(f"Choose image for {student_to_enroll}", type=["jpg", "jpeg", "png"], key="uploader")
            if uploaded_file and student_to_enroll:
                img = Image.open(uploaded_file).convert('RGB')
                st.image(img, caption="Uploaded Image", width=300)
                st.write("Processing...")
                
                img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                faces_with_coords = detect_faces(img_cv2)
                
                if not faces_with_coords:
                    st.error("No face detected in the uploaded image. Please try another photo.")
                else:
                    try:
                        face_img, _ = faces_with_coords[0]
                        embedding = get_embedding(Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)))
                        save_embedding(student_to_enroll, embedding)
                        st.success(f"Face enrolled successfully for {student_to_enroll}!")
                    except Exception as e:
                        st.error(f"Failed to process image: {e}")

    with tab3:
        st.subheader("Enroll Student (Webcam)")
        st.write("Use your webcam to capture a photo of an existing student for enrollment.")
        
        student_users = [user[0] for user in get_all_users() if user[1] == 'Student']
        student_name_to_enroll = None
        if not student_users:
            st.info("No students registered yet. Please add a new user with the 'Student' role first.")
        else:
            student_name_to_enroll = st.selectbox("Select Student", student_users, key="webcam_select")
        
        start_webcam_button = st.button("Start Webcam", key="start_webcam_enroll")
        stop_webcam_button = st.button("Stop Webcam", key="stop_webcam_enroll")
        capture_photo_button = st.button("Capture Photo", key="capture_photo")
        
        FRAME_WINDOW = st.image([])
        
        if start_webcam_button:
            st.session_state['camera_running'] = True
        if stop_webcam_button:
            st.session_state['camera_running'] = False
        
        if st.session_state['camera_running'] and student_name_to_enroll:
            camera = cv2.VideoCapture(0)
            while st.session_state['camera_running']:
                ret, frame = camera.read()
                if not ret:
                    st.error("Failed to capture video.")
                    st.session_state['camera_running'] = False
                    break
                
                display_frame = frame.copy()
                faces_with_coords = detect_faces(display_frame)
                
                for _, (x1, y1, x2, y2) in faces_with_coords:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(display_frame, "Ready to Capture", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

                FRAME_WINDOW.image(display_frame, channels="BGR", use_container_width=True)
                
                if capture_photo_button:
                    st.write("Processing captured photo...")
                    if not faces_with_coords:
                        st.error("No face detected in the captured photo. Please try again.")
                    else:
                        try:
                            face_img, _ = faces_with_coords[0]
                            embedding = get_embedding(Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)))
                            save_embedding(student_name_to_enroll, embedding)
                            st.success(f"Face enrolled successfully for {student_name_to_enroll}!")
                            st.session_state['camera_running'] = False
                        except Exception as e:
                            st.error(f"Failed to process image: {e}")
                    break
            camera.release()
            st.session_state['camera_running'] = False

    with tab4:
        st.subheader("Delete Student Face")
        st.write("Remove a student's face data from the system.")
        student_users = [user[0] for user in get_all_users() if user[1] == 'Student']
        student_to_delete = st.selectbox("Select Student to Delete Face Data", student_users)
        if st.button(f"Delete Face Data for {student_to_delete}", key="delete_face"):
            delete_embedding(student_to_delete)
            st.success(f"Face data for {student_to_delete} deleted successfully!")
    
    with tab5:
        st.subheader("Reset User Password")
        st.write("Reset the password for any user.")
        all_users = [user[0] for user in get_all_users()]
        user_to_reset = st.selectbox("Select User to Reset Password", all_users)
        new_password = st.text_input(f"New password for {user_to_reset}", type="password", key="reset_pass")
        if st.button("Reset Password", key="reset_user_pass"):
            reset_password(user_to_reset, new_password)
            st.success(f"Password for {user_to_reset} reset successfully!")

    with tab6:
        st.subheader("All Attendance Records")
        records = get_attendance()
        if records:
            df = pd.DataFrame(records, columns=['Username', 'Timestamp'])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No attendance records yet.")

def Teacher_Dashboard():
    st.header("Teacher Dashboard")
    st.write("Mark attendance and view student records.")

    tab1, tab2 = st.tabs(["Mark Attendance", "View Attendance Records"])

    with tab1:
        st.subheader("Real-time Attendance")
        st.write("Click 'Start Real-time Attendance' to begin. The system will automatically detect and mark present students in the live feed.")
        
        start_button = st.button("Start Real-time Attendance", key="start_rt_attendance")
        stop_button = st.button("Stop Attendance", key="stop_rt_attendance")

        if start_button:
            st.session_state['camera_running'] = True
        if stop_button:
            st.session_state['camera_running'] = False

        frame_placeholder = st.empty()
        
        if st.session_state['camera_running']:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Error: Could not open webcam.")
                st.session_state['camera_running'] = False
            else:
                while st.session_state['camera_running']:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture image from webcam.")
                        break

                    faces_with_coords = detect_faces(frame)
                    display_frame = frame.copy()
                    
                    for face_img, (x1, y1, x2, y2) in faces_with_coords:
                        try:
                            img_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                            embedding = get_embedding(img_pil)
                            
                            # Use the updated recognize_face function which has a consistent threshold
                            user, score = recognize_face(embedding)

                            if user and score > 0.7:
                                today_present_records = get_attendance_for_date(str(date.today()))
                                today_present_usernames = [rec[0] for rec in today_present_records]
                                
                                # Mark attendance only if they aren't already marked for today
                                if user not in today_present_usernames:
                                    mark_attendance(user)
                                    st.success(f"Attendance marked for {user}!")
                                
                                # Add to the session list to display
                                if user not in st.session_state['present_this_session']:
                                    st.session_state['present_this_session'].append(user)

                                # Draw bounding box and text
                                text = f"{user}: {score:.2f}"
                                color = (0, 255, 0)
                            else:
                                text = "Unknown"
                                color = (0, 0, 255)

                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(display_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                        except Exception as e:
                            st.error(f"Error during face processing: {e}")
                    
                    frame_placeholder.image(display_frame, channels="BGR", use_container_width=True)

                cap.release()
                cv2.destroyAllWindows()

        st.subheader("Students Marked Present in This Session")
        if st.session_state['present_this_session']:
            for student in st.session_state['present_this_session']:
                st.success(student)
        else:
            st.info("No students marked present yet.")


    with tab2:
        st.subheader("Current Attendance Records")
        
        all_students = [user[0] for user in get_all_users() if user[1] == 'Student']
        today_present = [rec[0] for rec in get_attendance_for_date(str(date.today()))]
        present_count = len(today_present)
        absent_count = len(all_students) - present_count
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Students", len(all_students))
        col2.metric("Present Today", present_count)
        col3.metric("Absent Today", absent_count)

        st.markdown("---")
        st.subheader("List of all Students")
        st.dataframe(pd.DataFrame(all_students, columns=['Students']), use_container_width=True)
        
        st.markdown("---")
        st.subheader("Detailed Attendance Records")
        records = get_attendance()
        if records:
            df = pd.DataFrame(records, columns=['Username', 'Timestamp'])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No attendance records yet.")

def Student_Dashboard():
    st.header("Student Panel")
    st.write("View your face registration and attendance history.")

    tab1, tab2 = st.tabs(["Register My Face", "View My Attendance"])
    
    with tab1:
        st.subheader("Register Your Face")
        st.write(f"Hello, **{st.session_state['username']}**! Please upload a clear photo of your face to register it for attendance.")
        uploaded_file = st.file_uploader("Choose a face image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            try:
                img = Image.open(uploaded_file).convert('RGB')
                st.image(img, caption="Uploaded Image", width=300)
                st.write("Processing...")
                
                img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                faces_with_coords = detect_faces(img_cv2)
                
                if not faces_with_coords:
                    st.error("No face detected in the uploaded image. Please try another photo.")
                else:
                    face_img, _ = faces_with_coords[0]
                    embedding = get_embedding(Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)))
                    save_embedding(st.session_state['username'], embedding)
                    st.success("Face enrolled successfully! You are now ready to be marked present by a teacher.")
            except Exception as e:
                st.error(f"Failed to process image: {e}")

    with tab2:
        st.subheader("My Attendance History")
        
        # Dashboard for today's status
        col1, col2 = st.columns(2)
        
        records_today = get_attendance_for_date(str(date.today()))
        present_today = any(rec[0] == st.session_state['username'] for rec in records_today)
        
        with col1:
            st.metric("Status Today", "Present" if present_today else "Absent")

        records_total = get_attendance(st.session_state['username'])
        with col2:
            st.metric("Total Present Days", len(records_total))

        st.markdown("---")
        st.subheader("Detailed Attendance Records")
        
        if records_total:
            df = pd.DataFrame(records_total, columns=['Username', 'Timestamp'])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No attendance records found for you.")

# --- Main App Logic ---
# Login section in the sidebar
st.sidebar.title("Login")

# Add a helpful hint for the user
st.sidebar.info("Default Admin: **admin** / **admin**")

if st.session_state['role'] is None:
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        role = authenticate(username, password)
        if role:
            st.session_state['role'] = role
            st.session_state['username'] = username
            st.success(f"Logged in as {role}")
            st.rerun()
        else:
            st.error("Invalid credentials")
else:
    st.sidebar.title("User Info")
    st.sidebar.write(f"Logged in as: **{st.session_state['username']}**")
    st.sidebar.write(f"Role: **{st.session_state['role']}**")
    if st.sidebar.button("Logout"):
        st.session_state['role'] = None
        st.session_state['username'] = None
        st.session_state['camera_running'] = False
        st.session_state['present_this_session'] = []
        st.rerun()

    role = st.session_state['role']
    
    if role == "Admin":
        Admin_Dashboard()
    elif role == "Teacher":
        Teacher_Dashboard()
    elif role == "Student":
        Student_Dashboard()
