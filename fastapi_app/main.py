from fastapi import FastAPI
from fastapi_app.routers import ocr


app = FastAPI()

app.include_router(ocr.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to OCR API's"}