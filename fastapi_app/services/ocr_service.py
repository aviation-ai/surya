# fastapi_app/services/ocr_service.py
import os
import argparse
from PIL import Image
from typing import List
import io
import pypdfium2

from surya.detection import batch_text_detection
from surya.layout import batch_layout_detection
from surya.model.detection.segformer import load_model, load_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from surya.model.ordering.processor import load_processor as load_order_processor
from surya.model.ordering.model import load_model as load_order_model
from surya.ordering import batch_ordering
from surya.postprocessing.heatmap import draw_polys_on_image
from surya.ocr import run_ocr
from surya.postprocessing.text import draw_text_on_image
from surya.languages import CODE_TO_LANGUAGE
from surya.input.langs import replace_lang_with_code
from surya.schema import OCRResult, TextDetectionResult, LayoutResult, OrderResult
from surya.settings import settings

args = argparse.Namespace(math=False)

def load_det_cached():
    checkpoint = settings.DETECTOR_MATH_MODEL_CHECKPOINT if args.math else settings.DETECTOR_MODEL_CHECKPOINT
    return load_model(checkpoint=checkpoint), load_processor(checkpoint=checkpoint)

def load_rec_cached():
    return load_rec_model(), load_rec_processor()

def load_layout_cached():
    return load_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT), load_processor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)

def load_order_cached():
    return load_order_model(), load_order_processor()

def text_detection(img, det_model, det_processor) -> (Image.Image, TextDetectionResult):
    pred = batch_text_detection([img], det_model, det_processor)[0]
    polygons = [p.polygon for p in pred.bboxes]
    det_img = draw_polys_on_image(polygons, img.copy())
    return det_img, pred

def layout_detection(img, layout_model, layout_processor) -> (Image.Image, LayoutResult):
    _, det_pred = text_detection(img, layout_model, layout_processor)
    pred = batch_layout_detection([img], layout_model, layout_processor, [det_pred])[0]
    polygons = [p.polygon for p in pred.bboxes]
    labels = [p.label for p in pred.bboxes]
    layout_img = draw_polys_on_image(polygons, img.copy(), labels=labels)
    return layout_img, pred

def order_detection(img, order_model, order_processor) -> (Image.Image, OrderResult):
    _, layout_pred = layout_detection(img, order_model, order_processor)
    bboxes = [l.bbox for l in layout_pred.bboxes]
    pred = batch_ordering([img], [bboxes], order_model, order_processor)[0]
    polys = [l.polygon for l in pred.bboxes]
    positions = [str(l.position) for l in pred.bboxes]
    order_img = draw_polys_on_image(polys, img.copy(), labels=positions, label_font_size=20)
    return order_img, pred

def ocr(img, langs: List[str], det_model, det_processor, rec_model, rec_processor) -> (Image.Image, OCRResult):
    replace_lang_with_code(langs)
    img_pred = run_ocr([img], [langs], det_model, det_processor, rec_model, rec_processor)[0]
    bboxes = [l.bbox for l in img_pred.text_lines]
    text = [l.text for l in img_pred.text_lines]
    rec_img = draw_text_on_image(bboxes, text, img.size, langs, has_math="_math" in langs)
    return rec_img, img_pred

def open_pdf(pdf_file):
    stream = io.BytesIO(pdf_file)
    return pypdfium2.PdfDocument(stream)

def get_page_image(pdf_file, page_num, dpi=96):
    doc = open_pdf(pdf_file)
    renderer = doc.render(
        pypdfium2.PdfBitmap.to_pil,
        page_indices=[page_num - 1],
        scale=dpi / 72,
    )
    png = list(renderer)[0]
    png_image = png.convert("RGB")
    return png_image

def page_count(pdf_file):
    doc = open_pdf(pdf_file)
    return len(doc)
