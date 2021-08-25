from typing import Optional
from fastapi import FastAPI

from functools import partial

import aiohttp
import asyncio

import os
import sys
import sox
import yaml
import hashlib

import logging
import logging.config


with open("config.yaml") as file_stream:
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

base_url = "https://eastus.tts.speech.microsoft.com/cognitiveservices"
api_token = os.environ.get("AZURE_API_TOKEN")
output_format = "raw-24khz-16bit-mono-pcm"
provider = "azure"
ssml_string = """
<speak version="1.0"
       xmlns="https://www.w3.org/2001/10/synthesis"
       xmlns:mstts="http://www.w3.org/2001/mstts"
       xml:lang="{language}">
  <voice name="{voice}">
    <prosody rate="{rate}" pitch="{pitch}">
      {lexicon_tag}
      {text}
    </prosody>
  </voice>
</speak>
"""


@app.get("/voicelist")
async def voicelisst():
    headers = {"Ocp-Apim-Subscription-Key": api_token}
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(f"{base_url}/voices/list") as result:
            text = await result.json()
            return text


@app.post("/synthesize")
async def synthesize(text: str,
                     voice: Optional[str] = "es-MX-JorgeNeural",
                     rate: Optional[str] = "0%",
                     pitch: Optional[str] = "0%",
                     lexicon: Optional[str] = None,
                     ):
    headers = {"Ocp-Apim-Subscription-Key": api_token,
               "X-Microsoft-OutputFormat": output_format,
               "Content-Type": "application/ssml+xml"
               }
    lexicon_tag = f'<lexicon uri="{lexicon}"/>' if lexicon else ""

    data = ssml_string.format(text=text,
                              voice=voice,
                              language=voice[0:5],
                              rate=rate,
                              pitch=pitch,
                              lexicon_tag=lexicon_tag
                              )
    logger.debug(data)
    # Define file name and path
    filename = hashlib.md5(text.encode()).hexdigest()
    file_dir = f"/sounds/{provider}/{voice}"
    file_path = f"{file_dir}/{filename}"

    ensure_dir(file_dir)
    output_filepath = f"{file_path}.sln24"
    if not os.path.exists(output_filepath) or os.path.getsize(output_filepath) <= 0:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(f"{base_url}/v1", data=data) as result:
                with open(output_filepath, 'wb') as output_file:
                    while True:
                        chunk = await result.content.read(1024)
                        if not chunk:
                            break
                        output_file.write(chunk)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None,
                                        partial(sox_converter, file_path, 'alaw', 'al', 8000)
                                        )

    return {"sound_path": f"{provider}/{voice}/{filename}"}


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def sox_converter(file_path, exten, file_type, rate=16000, bits=16, channels=1):
    input_filepath = f'{file_path}.sln24'
    output_filepath = f'{file_path}.{exten}'

    if not os.path.exists(output_filepath) or os.path.getsize(output_filepath) <= 0:
        tfm = sox.Transformer()
        tfm.silence(location=1)
        tfm.silence(location=-1)
        tfm.pad(0.05, 0.05)

        tfm.set_input_format(file_type='sln', rate=24000, bits=16, channels=1)
        tfm.set_output_format(file_type, rate, bits, channels)

        result = tfm.build_file(input_filepath=input_filepath, output_filepath=output_filepath)
    else:
        result = None

    return result
