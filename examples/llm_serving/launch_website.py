import json
import logging
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


def log_scope(request):
    scope = request.scope
    del scope["app"]
    del scope["fastapi_astack"]
    del scope["router"]
    del scope["endpoint"]
    del scope["route"]
    scope["tstamp"] = time.time()
    logging.info(scope)
    return scope


##### Redirect Begin #####
import asyncio
import pickle
import time

from alpa.serve.http_util import HTTPRequestWrapper, make_error_response, RelayException
import ray
from starlette.responses import JSONResponse
ray.init(address="auto", namespace="alpa_serve")

manager = None

async def connect_manager():
    global manager
    while True:
        if manager is None:
            try:
                manager = ray.get_actor("mesh_group_manager_0")
            except ValueError:
                manager = None
        await asyncio.sleep(1)

asyncio.get_event_loop().create_task(connect_manager())

async def redirect(request):
    global manager

    body = await request.body()
    scope = log_scope(request)
    request = pickle.dumps(HTTPRequestWrapper(scope, body))
    try:
        ret = await manager.handle_request.remote("default", request)
    except ray.exceptions.RayActorError:
        manager = None
    if isinstance(ret, RelayException):
        ret = make_error_response(ret)
        ret = JSONResponse(ret, status_code=400)
    return ret


@app.post("/completions")
async def completions(request: Request):
    return await redirect(request)


@app.post("/logprobs")
async def logprobs(request: Request):
    return await redirect(request)


@app.post("/call")
async def logprobs(request: Request):
    return await redirect(request)

##### Redirect End #####

@app.get("/")
async def homepage(request: Request):
    for x in request.scope['headers']:
        if x[0] == b"user-agent" and b"UptimeRobot" not in x[1]:
            log_scope(request)
            break
    return templates.TemplateResponse("index.html", {
        "request": request,
        "num_return_sequences": NUM_RETURN_SEQ,
        "sampling_css": sampling_css,
        "recaptcha": recaptcha.get_code(),
        "alpa_serve_url": ALPA_SERVE_URL,
    })
