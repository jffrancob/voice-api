import asyncio
import functools
import logging
import logging.config
import os
import sys
from typing import List, Optional

import azure.cognitiveservices.speech as speechsdk
import yaml
from fastapi import FastAPI, Query

with open("/etc/config.yaml") as file_stream:
    config = yaml.full_load(file_stream)

logging.config.dictConfig(config.get("logging"))
logger = logging.getLogger()
logger.debug("starting config...")

try:
    os.environ["AZURE_API_TOKEN"]
except KeyError:
    logger.error("Please set the environment variable AZURE_API_TOKEN")
    sys.exit(1)

app = FastAPI()
loop = asyncio.get_running_loop()

api_token = os.environ.get("AZURE_API_TOKEN")
api_region = os.environ.get("API_REGION", "eastus")
DEFAULT_LANG = os.environ.get("DEFAULT_LANG", "es-MX")


def azure_recognize(speech_recognizer):
    result = speech_recognizer.recognize_once_async().get()
    return result


@app.post("/recognize")
async def recognize(
    file_path: str,
    phrase: Optional[List[str]] = Query(None),
    language: Optional[str] = DEFAULT_LANG,
):
    audio_file = os.path.join("/sounds", file_path)

    try:
        logger.debug(
            f"Executing Recognition to file: {audio_file} and phrase list {phrase}"
        )

        speech_config = speechsdk.SpeechConfig(
            subscription=api_token, region=api_region
        )
        speech_config.speech_recognition_language = language
        audio_input = speechsdk.AudioConfig(filename=audio_file)
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_input
        )

        if phrase:
            phrase_list_grammar = speechsdk.PhraseListGrammar.from_recognizer(
                speech_recognizer
            )
            for sentence in phrase:
                phrase_list_grammar.addPhrase(sentence)

        result = await loop.run_in_executor(
            None, functools.partial(azure_recognize, speech_recognizer)
        )
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            result_text = result.text.strip(" .")
            logger.debug(f"Recognition in {file_path}. result: {result_text}")
            return {"text": result_text}
        elif result.reason == speechsdk.ResultReason.NoMatch:
            logger.error(f"No speech could be recognized: {result.no_match_details}")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            logger.error(f"Speech Recognition canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                logger.error(f"Error details: {cancellation_details.error_details}")
        else:
            logger.error(f"Speech Recognition result: {result.reason}")

        return None

    except Exception as e:
        logger.debug(
            "Could not request results from Google Cloud Speech service; {0}".format(e)
        )
