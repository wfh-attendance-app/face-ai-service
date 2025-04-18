import face_recognition
import faiss
import numpy as np
import os

class FaceStore:
    def __init__(self, reference_dir="reference"):
        self.reference_dir = reference_dir
        self.user_ids = []
        self.embeddings = []
        self.index = None
        self._load_faces()

    def _load_faces(self):
        for filename in os.listdir(self.reference_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(filename)[0]
                path = os.path.join(self.reference_dir, filename)
                image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    self.user_ids.append(name)
                    self.embeddings.append(encodings[0])

        if self.embeddings:
            emb_array = np.array(self.embeddings).astype('float32')
            self.index = faiss.IndexFlatL2(emb_array.shape[1])
            self.index.add(emb_array)

    def match_face(self, query_encoding, threshold=0.6):
        if not self.index:
            return None

        query = np.array([query_encoding]).astype('float32')
        distances, indices = self.index.search(query, 1)
        best_distance = distances[0][0]
        best_index = indices[0][0]

        if best_distance < threshold:
            return self.user_ids[best_index], best_distance
        else:
            return None
