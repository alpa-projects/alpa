import json
from typing import Union

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from llm_serving.service.constants import (
    NUM_BEAMS, NUM_RETURN_SEQ, ALPA_SERVE_URL, USE_RECAPTCHA)
from llm_serving.service.recaptcha import load_recaptcha

app = FastAPI()

app.mount("/static", StaticFiles(directory="service/static"), name="static")
templates = Jinja2Templates(directory="service/static")

if NUM_BEAMS > 1: # beam search is on, disable sampling
    sampling_css = "display:none"
else:
    sampling_css = ""


recaptcha = load_recaptcha(USE_RECAPTCHA)


@app.get("/")
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "num_return_sequences": NUM_RETURN_SEQ,
        "sampling_css": sampling_css,
        "recaptcha": recaptcha.get_code(),
        "alpa_serve_url": ALPA_SERVE_URL,
    })
