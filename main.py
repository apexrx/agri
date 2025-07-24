refrom fastapi import FastAPI, File, UploadFile, Form
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter
import joblib
from pydantic import BaseModel


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the routers for each version
v1_router = APIRouter(prefix="/v1")
v2_router = APIRouter(prefix="/v2")
v3_router = APIRouter(prefix="/v3")
v4_router = APIRouter(prefix="/v4")
v5_router = APIRouter(prefix="/v5")

FERT_SCALER = joblib.load("../training/scaler.joblib")
CROP_SCALER = joblib.load("../training/CropScaler.joblib")

# Load models
POTATO_MODEL = tf.keras.models.load_model("../training/model_1.keras")
BELL_PEPPER_MODEL = tf.keras.models.load_model("../training/model_2.keras")
FERTILIZER_MODEL = joblib.load("../training/xgboost_model.joblib")
CROP_MODEL = tf.keras.models.load_model("../training/model_3.keras")
WHEAT_MODEL = tf.keras.models.load_model("../training/model_4.keras")

# Define class names
POTATO_CLASS_NAMES = ["Early blight", "Late blight", "Healthy"]
BELL_PEPPER_CLASS_NAMES = ["Bacterial Spot", "Healthy"]
WHEAT_CLASS_NAMES = ["Brown Rust", "Healthy", "Loose Smut", "Septoria", "Yellow Rust"]

CROP_ENCODING = {
    0: 'Cotton', 1: 'Ginger', 2: 'Gram', 3: 'Grapes', 4: 'Groundnut', 5: 'Jowar',
    6: 'Maize', 7: 'Masoor', 8: 'Moong', 9: 'Rice', 10: 'Soybean', 11: 'Sugarcane',
    12: 'Tur', 13: 'Turmeric', 14: 'Urad', 15: 'Wheat'
}
SOIL_ENCODING = {
    0: 'Black', 1: 'Dark Brown', 2: 'Light Brown', 3: 'Medium Brown', 4: 'Red', 5: 'Reddish Brown'
}
FERTILIZER_ENCODING = {
    0: '10:10:10 NPK', 1: '10:26:26 NPK', 2: '12:32:16 NPK', 3: '13:32:26 NPK',
    4: '18:46:00 NPK', 5: '19:19:19 NPK', 6: '20:20:20 NPK', 7: '50:26:26 NPK',
    8: 'Ammonium Sulphate', 9: 'Chilated Micronutrient', 10: 'DAP', 11: 'Ferrous Sulphate',
    12: 'Hydrated Lime', 13: 'MOP', 14: 'Magnesium Sulphate', 15: 'SSP', 16: 'Sulphur',
    17: 'Urea', 18: 'White Potash'
}

CROP_DECODING = {v: k for k, v in CROP_ENCODING.items()}
SOIL_DECODING = {v: k for k, v in SOIL_ENCODING.items()}
FERTILIZER_DECODING = {v: k for k, v in FERTILIZER_ENCODING.items()}

CROP_CLASS_MAPPING = {
    0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea', 4: 'coconut', 5: 'coffee', 6: 'cotton',
    7: 'grapes', 8: 'jute', 9: 'kidneybeans', 10: 'lentil', 11: 'maize', 12: 'mango', 13: 'mothbeans',
    14: 'mungbean', 15: 'muskmelon', 16: 'orange', 17: 'papaya', 18: 'pigeonpeas', 19: 'pomegranate',
    20: 'rice', 21: 'watermelon'
}


@v1_router.get("/ping")
async def ping():
    return "Hello World from Potato Disease Model!"

@v2_router.get("/ping")
async def ping():
    return "Hello World from Bell Pepper Disease Model!"

@v3_router.get("/ping")
async def ping():
    return "Hello World from Fertilizer Recommendation Model!"

@v4_router.get("/ping")
async def ping():
    return "Hello World from Crop Recommendation Model!"

@v5_router.get("/ping")
async def ping():
    return "Hello World from Wheat Disease Model!"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@v1_router.post("/predict")
async def predict_potato_disease(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    prediction = POTATO_MODEL.predict(img_batch)

    predicted_class = POTATO_CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])

    return {
        "class": predicted_class,
        "confidence": f"{confidence * 100:.2f}%"
    }

@v2_router.post("/predict")
async def predict_bell_pepper_disease(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    prediction = BELL_PEPPER_MODEL.predict(img_batch)

    predicted_class = BELL_PEPPER_CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])

    return {
        "class": predicted_class,
        "confidence": f"{confidence * 100:.2f}%"
    }

def read_file_as_image2(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@v5_router.post("/predict")
async def predict_wheat_disease(
    file: UploadFile = File(...)
):
    image = read_file_as_image2(await file.read())

    print(f"Image shape before prediction: {image.shape}")
    print(f"Sample pixel values: {image[0, 0, 0, :]}")

    prediction = WHEAT_MODEL.predict(image)
    predicted_class = WHEAT_CLASS_NAMES[np.argmax(prediction[0])]
    confidence = round(100 * np.max(prediction[0]), 2)

    return {
        "class": predicted_class,
        "confidence": f"{confidence:.2f}%"
    }


class FertilizerInput(BaseModel):
    soil: str
    nitrogen: float
    phosphorus: float
    potassium: float
    pH: float
    rainfall: float
    temperature: float
    crop: str

@v3_router.post("/predict")
async def predict_fertilizer_recommendation(
    soil: str = Form(...),
    nitrogen: float = Form(...),
    phosphorus: float = Form(...),
    potassium: float = Form(...),
    pH: float = Form(...),
    rainfall: float = Form(...),
    temperature: float = Form(...),
    crop: str = Form(...)
):
    soil_encoded = SOIL_DECODING.get(soil, -1)
    crop_encoded = CROP_DECODING.get(crop, -1)

    if soil_encoded == -1 or crop_encoded == -1:
        return {"error": "Invalid soil or crop value"}

    input_features = np.array([
        soil_encoded, nitrogen, phosphorus, potassium, pH, rainfall, temperature, crop_encoded
    ]).reshape(1, -1)

    input_features_scaled = FERT_SCALER.transform(input_features)

    prediction = FERTILIZER_MODEL.predict(input_features_scaled)
    predicted_fertilizer = FERTILIZER_ENCODING.get(prediction[0], "Unknown")

    return {
        "recommended_fertilizer": predicted_fertilizer
    }

@v4_router.post("/predict")
async def predict_crop_class(
    nitrogen: float = Form(...),
    phosphorus: float = Form(...),
    potassium: float = Form(...),
    temperature: float = Form(...),
    humidity: float = Form(...),
    pH: float = Form(...),
    rainfall: float = Form(...)
):
    input_features = np.array([
        nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall
    ]).reshape(1, -1)

    input_features_scaled = CROP_SCALER.transform(input_features)

    prediction = CROP_MODEL.predict(input_features_scaled)
    predicted_class_index = np.argmax(prediction[0])
    predicted_class = CROP_CLASS_MAPPING.get(predicted_class_index, "Unknown")

    confidence = np.max(prediction[0]) * 100  # Convert to percentage

    return {
        "class": predicted_class,
        "confidence": f"{confidence:.2f}%"
    }

# Include the routers in the main app
app.include_router(v1_router)
app.include_router(v2_router)
app.include_router(v3_router)
app.include_router(v4_router)
app.include_router(v5_router)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
