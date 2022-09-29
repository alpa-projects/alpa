import json
from typing import Union

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from llm_serving.service.constants import (
    NUM_BEAMS, NUM_RETURN_SEQ, USE_RECAPTCHA, KEYS_FILENAME, ALPA_SERVE_URL)
from llm_serving.service.recaptcha import ReCaptcha

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

if NUM_BEAMS > 1: # beam search is on, disable sampling
    sampling_css = "display:none"
else:
    sampling_css = ""

if USE_RECAPTCHA:
    keys = json.load(open(KEYS_FILENAME, "r"))
    recaptcha = ReCaptcha(site_key=keys["RECAPTCHA_SITE_KEY"],
                          secret_key=keys["RECAPTCHA_SECRET_KEY"])
else:
    recaptcha = None

@app.get("/")
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "num_return_sequences": NUM_RETURN_SEQ,
        "sampling_css": sampling_css,
        "recaptcha": recaptcha.get_code() if recaptcha else "",
        "alpa_serve_url": ALPA_SERVE_URL,
    })
