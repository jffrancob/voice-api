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

DEFAULT_LANG = os.environ.get("DEFAULT_LANG", "es-MX-JorgeNeural")

app = FastAPI()

base_url = "https://eastus.tts.speech.microsoft.com/cognitiveservices"
api_token = os.environ.get("AZURE_API_TOKEN")
output_format = "raw-24khz-16bit-mono-pcm"
provider = "azure"
cache_format = "sln24"
ssml_string = """
<speak version="1.0"
       xmlns="https://www.w3.org/2001/10/synthesis"
       xmlns:mstts="http://www.w3.org/2001/mstts"
       xml:lang="{language}">
  <voice name="{voice}">
      {text}
  </voice>
</speak>
"""


loop = asyncio.get_event_loop()


@app.get("/voicelist")
async def voicelisst():
    headers = {"Ocp-Apim-Subscription-Key": api_token}
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(f"{base_url}/voices/list") as result:
            text = await result.json()
            return text


@app.post("/synthesize")
async def synthesize(
    text: str,
    voice: Optional[str] = DEFAULT_LANG,
    exten: Optional[str] = "alaw",
    file_type: Optional[str] = "al",
    rate: Optional[str] = 8000,
):
    headers = {
        "Ocp-Apim-Subscription-Key": api_token,
        "X-Microsoft-OutputFormat": output_format,
        "Content-Type": "application/ssml+xml",
    }

    # SSML to send
    data = ssml_string.format(text=text, voice=voice, language=voice[0:5])
    logger.debug(data)

    # Define file name and path
    md5_data = hashlib.md5(data.encode()).hexdigest()
    d1, d2, filename = md5_data[0:2], md5_data[2:4], md5_data
    dir_schema = f"{provider}/{voice}/{d1}/{d2}"
    file_dir = f"/sounds/{dir_schema}"
    file_path = f"{file_dir}/{filename}"

    # We need to ensure the defined directory
    ensure_dir(file_dir)

    # Store the content in plain text
    with open(f"{file_path}.txt", "w") as content:
        content.write(data)
        content.close()

    # This block performs the synthesize using Azure.
    # It's done using the best quality.
    # After that the final format is obtained using SOX
    cache_filepath = f"{file_path}.{cache_format}"
    if not os.path.exists(cache_filepath) or os.path.getsize(cache_filepath) <= 0:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(f"{base_url}/v1", data=data) as result:
                with open(cache_filepath, "wb") as output_file:
                    while True:
                        chunk = await result.content.read(1024)
                        if not chunk:
                            break
                        output_file.write(chunk)

    if not os.path.exists(cache_filepath) or os.path.getsize(cache_filepath) <= 0:
        result = "azure_error"
    else:
        # The SOX converter is called in async mode
        result = await loop.run_in_executor(
            None, partial(sox_converter, file_path, exten, file_type, rate)
        )

    # The final result is returned
    return {"sound_path": f"{dir_schema}/{filename}", "exten": exten, "result": result}


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def sox_converter(file_path, exten, file_type, rate=16000, bits=16, channels=1):
    input_filepath = f"{file_path}.{cache_format}"
    output_filepath = f"{file_path}.{exten}"

    if os.path.exists(output_filepath) and os.path.getsize(output_filepath) > 0:
        return "cached"
    else:
        tfm = sox.Transformer()
        tfm.silence(location=1)
        tfm.silence(location=-1)
        tfm.pad(0.05, 0.05)

        tfm.set_input_format(file_type="sln", rate=24000, bits=16, channels=1)
        tfm.set_output_format(file_type, rate, bits, channels)

        result = tfm.build_file(input_filepath=input_filepath, output_filepath=output_filepath)

        return "success" if result else "error"
