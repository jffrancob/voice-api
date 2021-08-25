from fastapi import FastAPI

import speech_recognition as sr
from os import path
import yaml

import logging
import logging.config

import asyncio
import functools
import traceback


with open("config.yaml") as file_stream:
    config = yaml.full_load(file_stream)

logging.config.dictConfig(config.get("logging"))
logger = logging.getLogger()
logger.debug("starting config...")


app = FastAPI()


loop = asyncio.get_running_loop()

# recognize speech using Google Cloud Speech
file = open('/GOOGLE_CLOUD_SPEECH_CREDENTIALS', mode='r')
GOOGLE_CLOUD_SPEECH_CREDENTIALS = file.read()
file.close()


@app.post("/recognize")
async def recognize(file_path: str):
    audio_file = path.join("/sounds", file_path)

    try:
        logger.debug(f"Executing Recognition to file: {file_path}")
        r = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)
            data = {"audio_data": audio,
                    "language": 'es_CO',
                    "credentials_json": GOOGLE_CLOUD_SPEECH_CREDENTIALS
                    }
            result = await loop.run_in_executor(None, functools.partial(r.recognize_google_cloud,
                                                                        **data))
            result_text = result.strip(" .")
            logger.debug(f"Recognition in {file_path}. result: {result_text}")
            return {"text": result_text}

    except Exception as e:
        logger.error(traceback.format_exc())
        logger.debug("Could not request results from Google Cloud Speech service; {0}".format(e))
