from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import face_recognition
import io
from face_store import FaceStore

app = FastAPI()
store = FaceStore()

@app.post("/identify")
async def identify_face(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = face_recognition.load_image_file(io.BytesIO(image_bytes))
    locations = face_recognition.face_locations(image)
    if not locations:
        return JSONResponse(content={"error": "No face detected"}, status_code=400)

    encoding = face_recognition.face_encodings(image, locations)[0]
    result = store.match_face(encoding)

    if result:
        user_id, distance = result
        return {"match": True, "user_id": user_id, "distance": float(distance)}
    else:
        return {"match": False}
