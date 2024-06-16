from fastapi import APIRouter, File, UploadFile
from PIL import Image
from fastapi_app.services.ocr_service import (
    load_det_cached,
    load_rec_cached,
    load_layout_cached,
    load_order_cached,
    text_detection,
    layout_detection,
    order_detection,
    ocr,
    get_page_image,
    page_count
)
import io

router = APIRouter()

det_model, det_processor = load_det_cached()
rec_model, rec_processor = load_rec_cached()
layout_model, layout_processor = load_layout_cached()
order_model, order_processor = load_order_cached()

@router.post("/text-detection/")
async def text_detection_endpoint(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    det_img, pred = text_detection(img, det_model, det_processor)
    return {"detected_text": pred.model_dump()}

@router.post("/ocr/")
async def ocr_endpoint(file: UploadFile = File(...), langs: list[str]):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    rec_img, pred = ocr(img, langs, det_model, det_processor, rec_model, rec_processor)
    return {"ocr_result": pred.model_dump()}

@router.post("/layout-detection/")
async def layout_detection_endpoint(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    layout_img, pred = layout_detection(img, layout_model, layout_processor)
    return {"layout_result": pred.model_dump()}

@router.post("/order-detection/")
async def order_detection_endpoint(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    order_img, pred = order_detection(img, order_model, order_processor)
    return {"order_result": pred.model_dump()}