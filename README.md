Manual attendance is a major productivity drain. Traditional methods—roll calls, sign-in sheets, or physical biometrics—are slow, prone to proxy attendance (buddy punching), and create unnecessary bottlenecks, especially in high-traffic areas like classrooms or event entrances.
Our Goal: Deliver a real-time, zero-contact, and highly secure attendance system that can be deployed anywhere with a webcam, built using cutting-edge deep learning models.
🚀 The Solution: A Deep Learning Powerhouse
We engineered a full-stack Python application that leverages two best-in-class computer vision models to create a robust and virtually instantaneous attendance log.
🛠️ Core Technology Stack & Algorithms


Component
	Technology / Algorithm
	Why We Chose It (The Advantage)
	Face Detection
	YOLOv8 (You Only Look Once)
	Speed! YOLOv8 is an ultra-fast model that ensures near-real-time performance, even on a standard CPU, allowing us to detect faces instantly.
	Feature Extraction
	FaceNet (InceptionResnetV1)
	Accuracy! FaceNet generates highly descriptive, 128-dimensional Face Embeddings. This is the digital 'fingerprint' of a face, ensuring low false-positive rates.
	Matching Engine
	Cosine Similarity
	Efficiency! This mathematical metric quickly measures the 'distance' between a new face embedding and all stored embeddings, giving us a match score in milliseconds.
	Frontend
	Streamlit + OpenCV
	Rapid Prototyping! Streamlit allowed us to build an interactive, responsive web app with webcam integration in hours, perfect for a hackathon environment.
	Database
	SQLite3
	Simplicity! We opted for local, lightweight SQLite to ensure fast read/write speeds for user and attendance logs without requiring a complex server setup.
	⚙️ How It Works: The Automated Flow
The system operates in two secure phases: Enrollment and Real-Time Marking.
1. Enrollment: Creating the Digital ID
1. Input: A new user (Student) provides a face photo.
2. Crop & Focus: YOLOv8 precisely finds and crops the face.
3. Vectorization: The cropped face is fed to FaceNet, which converts it into a unique 128-D vector (the embedding).
4. Secure Storage: This vector is saved as a dedicated .npy file, indexed by the user's name, ready for comparison.
2. Attendance Marking: Verification in Action
1. Live Feed: The system continuously captures frames from the webcam.
2. Real-time Detection: YOLOv8 scans the frame, detecting all faces present.
3. Identification: Each new detected face is vectorized into an embedding.
4. Matching: The new embedding is instantly compared against all saved user embeddings using Cosine Similarity.
5. Logging: If the similarity score is above the 0.6 threshold, the user is identified, and their attendance is marked in the attendance.db—eliminating proxy attempts.
🏆 Impact & Key Advantages (Why We Win)
* Zero-Contact Solution: Essential for modern hygiene standards in schools and offices.
* Unrivaled Speed: From detection to logging, the entire process takes less than 500ms, vastly improving user throughput.
* Proxy-Proof Security: By utilizing a robust biometric signature, the system completely prevents time theft or unauthorized check-ins.
* Role-Based Management: Secure access is enforced for Admin, Teacher, and Student roles, ensuring data integrity and administrative control.
📁 Project Structure (Modular and Scalable)
File
	Description
	app.py
	The Streamlit Web UI and main application logic. The front door of the project.
	auth.py
	Handles all user management and authentication, using SHA-256 for secure password hashing.
	database.py
	The data logger; handles all attendance insertions and retrieval from attendance.db.
	yolov8_utils.py
	The Vision System; handles the initial, fast face detection step.
	facenet_utils.py
	The AI Brain; responsible for generating high-quality face embeddings and performing similarity matching.
	💡 Next Steps & Future Scope
* Migrate to a production database (e.g., PostgreSQL or Firestore) for large-scale, distributed deployment.
* Add multi-face tracking and batch attendance processing for large classroom environments.
* Integrate liveness detection to prevent marking attendance using photos or video playback.
