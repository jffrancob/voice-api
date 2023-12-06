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
from fastapi import FastAPI
from pydantic import BaseModel


class AudioFile(BaseModel):
    file_path: str
    file_format: Optional[str] = "wav"
    model: str
    language: str = "English"


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

logger.debug("OpenAPI initialized")

app = FastAPI()
loop = asyncio.get_running_loop()

system_prompts = {
    "date-parser": """You will be provided with two variables separated by commas.
The first one is a specific date and the second one is a mesure of time.
Your task is to provide the final date in format YYYY/MM/DD.
If it is not possible to calculate the date, the response is null""",
    "user-correspond": """The user has been asked if they are a certain person.
Your task is to determine if the answer is positive or negative .
If it is not posible to determine then the answer is Null.
Limit the answer to [True, False, Null].""",
}


def whisper_recognize(audio_path: str, language: str):
    audio_file = open(audio_path, "rb")
    if language.lower() != "english":
        transcript = openai.Audio.translate(
            "whisper-1",
            audio_file,
            prompt=f"The input audio is in {language}. Translate into english",
        )
    else:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)

    return transcript


@app.post("/recognize")
async def recognize(audio: AudioFile):
    logger.debug(f"Received request to recognize audio: {audio}")
    audio_path = os.path.join("/sounds", f"{audio.file_path}.{audio.file_format}")

    logger.debug(f"Audio path: {audio_path}")
    try:
        logger.debug(
            f"Executing Recognition to file {audio_path} with model {audio.model} and language {audio.language}"
        )

        try:
            result = await loop.run_in_executor(
                None,
                functools.partial(
                    whisper_recognize, audio_path=audio_path, language=audio.language
                ),
            )
        except Exception as error:
            logger.debug(f"Could not request results from openAI translate: {error}")
            return error

        logger.debug(f"Speech Recognition result: {result}")
        audio_transcript = result.get("text")

        if audio_transcript:
            model = audio.model
            system_prompt = system_prompts.get(audio.model)
            if audio.model == "date-parser":
                model = "ft:gpt-3.5-turbo-0613:personal::80yeg8Ta"
                today = datetime.now().strftime("%Y/%m/%d")
                temperature = 0.0
                content = f"{today}, {audio_transcript}"
            elif audio.model == "user-correspond":
                model = "ft:gpt-3.5-turbo-0613:personal::83QYn3GH"
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
        logger.debug(
            f"Could not request results from whisper for model: {model}; {error}"
        )
        return error


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
