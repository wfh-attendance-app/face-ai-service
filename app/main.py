from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from face_store import FaceStore
import face_recognition
import io, os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = FaceStore()

@app.post("/identify")
async def identify_face(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = face_recognition.load_image_file(io.BytesIO(image_bytes))
    locations = face_recognition.face_locations(image)
    if not locations:
        return JSONResponse(content={"error": "no face detected"}, status_code=400)

    encoding = face_recognition.face_encodings(image, locations)[0]
    result = store.match_face(encoding)

    if result:
        user_id, distance = result
        return {"match": True, "user_id": user_id, "distance": float(distance)}
    else:
        return {"match": False}

@app.post("/verify")  # 1:1 match
async def verify_face(file: UploadFile = File(...), user_id: str = Form(...)):
    image = face_recognition.load_image_file(io.BytesIO(await file.read()))
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        raise HTTPException(status_code=400, detail="no face found in uploaded image.")

    uploaded_encoding = encodings[0]

    reference_path = f"reference/{user_id}.jpg"
    if not os.path.exists(reference_path):
        raise HTTPException(status_code=404, detail="reference image not found.")

    reference_image = face_recognition.load_image_file(reference_path)
    reference_encodings = face_recognition.face_encodings(reference_image)

    if not reference_encodings:
        raise HTTPException(status_code=400, detail="no face found in reference image.")

    result = face_recognition.compare_faces([reference_encodings[0]], uploaded_encoding)[0]
    return {"match": bool(result)}
