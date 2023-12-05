import asyncio
import functools
import logging
import logging.config
import os
import sys
from datetime import datetime
from typing import List, Optional

import openai
import yaml
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing_extensions import Annotated


class AudioFile(BaseModel):
    file_path: str
    file_format: str = "wav"
    model: str


with open("/etc/config.yaml") as file_stream:
    config = yaml.full_load(file_stream)

logging.config.dictConfig(config.get("logging"))
logger = logging.getLogger()
logger.debug("starting config...")

try:
    openai.api_key = os.environ["OPENAI_API_KEY"]
except KeyError:
    logger.error("Please set the environment variable OPENAI_API_KEY")
    sys.exit(1)

app = FastAPI()
loop = asyncio.get_running_loop()

system_prompt1 = """You will be provided with two variables separated by commas.
The first one is a specific date and the second one is a mesure of time.
Your task is to provide the final date in format YYYY/MM/DD.
If it is not possible to calculate the date, the response is null"""

system_prompt2 = """The user has been asked if they are a certain person.
Your task is to determine if the answer is positive or negative .
If it is not posible to determine then the answer is Null.
Limit the answer to [True, False, Null]."""


def whisper_recognize(audio_path):
    audio_file = open(audio_path, "rb")
    transcript = openai.Audio.translate(
        "whisper-1",
        audio_file,
        prompt="The input audio is in spanish. Translate into english",
    )
    return transcript


@app.post("/recognize")
async def recognize(audio: AudioFile):
    audio_path = os.path.join("/sounds", f"{audio.file_path}.{audio.file_format}")

    try:
        logger.debug(f"Executing Recognition to file {audio} with model {audio.dic()}")

        result = await loop.run_in_executor(
            None, functools.partial(whisper_recognize, audio_path)
        )

        logger.debug(f"Speech Recognition result: {result}")
        audio_transcript = result.get("text")

        if audio_transcript:
            model = audio.model
            if audio.model == "date-parser":
                model = "ft:gpt-3.5-turbo-0613:personal::80yeg8Ta"
                system_prompt = system_prompt1
                today = datetime.now().strftime("%Y/%m/%d")
                temperature = 0.0
                content = f"{today}, {audio_transcript}"
            elif audio.model == "user-correspond":
                model = "ft:gpt-3.5-turbo-0613:personal::83QYn3GH"
                system_prompt = system_prompt2
                temperature = 1
                content = audio_transcript

            result = await loop.run_in_executor(
                None,
                functools.partial(
                    correct_transcript, model, temperature, system_prompt, content
                ),
            )
            logger.debug(f"Transcript correction result: {result}")
            result = {"text": result}

        return result

    except Exception as error:
        logger.debug(f"Could not request results from whisper for model: {model}; {error}")
        return error

@app.post("/files/")
async def create_file(file: Annotated[bytes, File(description="A file read as bytes")]):
    return {"file_size": len(file)}


@app.post("/uploadfile/")
async def create_upload_file(
    file: Annotated[UploadFile, File(description="A file read as UploadFile")],
):
    content = await file.read()
    return {"filename": file.filename,
            "content_type": file.content_type,
            "size": len(content),
            "content": content
            }


def correct_transcript(model, temperature, system_prompt, content):
    response = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
    )
    logger.debug(f"Transcript correction response: {response}")
    return response["choices"][0]["message"]["content"]
