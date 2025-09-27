import numpy as np
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import torch
import os
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Set up logging for better debugging
logging.basicConfig(level=logging.INFO)

# Set up device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load FaceNet model
try:
    logging.info(f"Using device: {device}")
    logging.info("Attempting to load FaceNet model...")
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    logging.info("FaceNet model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load FaceNet model: {e}")
    raise ImportError("Could not load FaceNet model. Check your internet connection or `facenet-pytorch` installation.")


def get_embedding(img: Image.Image):
    """Generates a 128-d embedding for a given face image."""
    img = img.resize((160, 160))
    img = np.array(img).astype('float32') / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = facenet(img).cpu().numpy()
    return embedding[0]

def save_embedding(username, embedding):
    """Saves a user's face embedding to a .npy file."""
    if not os.path.exists('data/embeddings'):
        os.makedirs('data/embeddings')
    np.save(f"data/embeddings/{username}.npy", embedding)

def recognize_face(embedding, threshold=0.7):
    """
    Compares a new embedding to all stored embeddings to find a match.
    Returns the username and confidence score of the best match.
    """
    best_match = None
    best_score = 0
    if not os.path.exists('data/embeddings'):
        return None, 0
    
    # Iterate through all saved embeddings
    for fname in os.listdir('data/embeddings'):
        if fname.endswith('.npy'):
            username = os.path.splitext(fname)[0]
            db_embedding = np.load(f"data/embeddings/{fname}")
            
            # Calculate cosine similarity
            score = cosine_similarity([embedding], [db_embedding])[0][0]
            
            if score > best_score:
                best_score = score
                best_match = username
    
    # Return match only if it meets the threshold
    if best_score > threshold:
        return best_match, best_score
    else:
        return None, best_score

def delete_embedding(username):
    """Deletes the embedding file for a specific user."""
    embedding_path = f"data/embeddings/{username}.npy"
    if os.path.exists(embedding_path):
        os.remove(embedding_path)
        return True
    return False
