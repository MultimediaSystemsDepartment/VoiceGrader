# CAISE-ASR

Source code for a web service that performs: 
* speech-to-text conversion on audio files, using _WhisperX_,
* live microphone audio speech-to-text conversion, using _Whisper Live_,
* audio quality assessment - MOS (using _WhiSQA_) and SNR.

The service was created for the CAISE project: [caise.pl](https://caise.pl).

Maintainer: Grzegorz Szwoch, Katedra Systemów Multimedialnych PG, grzszwoc@pg.edu.pl.


## Repository structure

- `/app/`: web application written with FastAPI.
- `/asr_processor/`: the audio processing module.
- `/charts/asr-caise/`: Helm chart for the service.
- `/charts/argo.yml`: service configuration file for Argo.
- `/data/`: placeholder directory for models and other data.
- `/kubernetes/`: configuration file for deployment with Kubernetes.
- `/util/`: helper scripts, mainly for testing.
- `/whisper_live/`: WhisperLive module code.
- `/Dockerfile`: configuration file for building the Docker image.
- `/asrdock`: script for handling Docker images.
- `/main.py`: main script for the application.
- `/requirements.txt`: list of the required Python modules.
- `/start_gui`: script for starting the application.

## Installation

### Requirements

* A computing platform with a CUDA-compatible device.
* Linux operating system.
* Python (tested with version 3.13).

Microphone access in the web browser requires that the service is run on a server
that can be accessed via secure connection (HTTPS). An user must accept the browser
permission to access the microphone.


### Standalone mode

Create Python virtual environment.
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start the service with the `start_gui` script, or the command in the virtual environment:
```
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 60
```


### Docker

Use the provided `Dockerfile` file to build and run the container.

You can use the provided `asrdock` script to perform the typical operations.
Build the container with `./asrdock build`, then run it with `./asrdock run`
in the standalone mode or with `./asrdock daemon` in the daemon mode.


### Kubernetes
Use the configuration file in `kubernetes/asr_workflow.yaml`. Adjust the configuration
to your platform, build the image and push it to the repository, then deploy the service 
with `kubectl`. See the `kubernetes/deploy` script for an example of service deployment
on the CAISE platform.


### Argo

The provided `.gitlab-ci.yml` file builds the image and the Helm chart, and pushes them
to the repository. Helm chart is in `charts/asr-caise`. Adjust the configuration for
your platform.
In Argo, add Helm repository, then create the application.
A configuration file used on the CAISE platform is in `charts/argo.yml`.


## User interface

Graphical user interface to the service is available at the main endpoint (`/`). For the CAISE
platform, the address is:

[https://asr-caise.ingress.caise.k8s.apl.task.gda.pl](https://asr-caise.ingress.caise.k8s.apl.task.gda.pl)

The interface is accessible in English and in Polish. The language version is adjusted 
to the preferred browser language, it can also be forced by adding `?lang=en` or `?lang=pl`
to the address.


## Processing endpoint (/process)

Audio file processing is available through REST API interface. The endpoint is `/process`.
For the CAISE platform:

[https://asr-caise.ingress.caise.k8s.apl.task.gda.pl/process](https://asr-caise.ingress.caise.k8s.apl.task.gda.pl/process)

A POST request must be sent to the endpoint, with the following data.
* _file_ (binary): audio file as binary data; accepted formats: WAV, MP3, FLAC. Required.
* _language_ (string): speech recognition language - a two-letter code ('en', 'pl', etc.).
Optional, automatic detection is performed if the code is not provided.
* _ref_text_ (string): a reference (ground truth) text to compare with the recognized text. 
Optional.

The processing results are returned as a JSON document.

A script that performs audio file processing is available at `util/process.py`.

Example request using Python _requests_ module:
```
with open('test.wav', 'rb') as fin:
    resp = requests.post(
        'https://asr-caise.ingress.caise.k8s.apl.task.gda.pl/process',
        files={'file': fin},
        data={
            'ref_text': 'Testing speech recognition',
            'language': 'en',
        },
        timeout=10,
    )
```

Server response:
```
{
  "snr": {
    "snr": 49.25248506758115,
    "peak_snr": 72.36391500694675,
    "level_speech": -23.112298571761762,
    "level_noise": -72.3647836393429,
    "perc_speech": 0.9777454637415439,
    "perc_noise": 0.02225453625845609
  },
  "qual": {
    "mos": 1.341814547777176,
    "noisiness": 1.604602038860321,
    "discontinuity": 1.9564339518547058,
    "coloration": 2.1745124459266663,
    "loudness": 2.1396666765213013
  },
  "asr": {
    "recognized": "Testing speech recognition.",
    "segments": [
      {
        "text": "Testing speech recognition.",
        "start": 0.031,
        "end": 2.495,
        "avg_logprob": -0.10493607683615251
      }
    ],
    "language": "pl",
    "avg_logprob": -0.10493607683615251,
    "wer": 0.0,
    "cer": 0.0
  },
  "processing_time": 0.261723468080163,
  "audio_time": 2.9656875,
  "usage": {
    "ts": 1782987994.322451,
    "gpu_mean": 35.63614461940792,
    "gpu_max": 35.63614461940792,
    "mem_mean": 5777.0,
    "mem_max": 5777,
    "mem_total": 22144
  }
}
```

## Other endpoints

### Performance metrics (/perf)

Performance metrics - GPU and VRAM mean nad maximum usage, may be obtained by sending
a GET request to the `/perf` endpoint. The data is updated every 60 seconds.
The request has no parameters, the response is in JSON format.

Example response:
```
{
  "ts": "2026-07-03T09:24:48.926130",
  "gpu_mean": 35.63614461940792,
  "gpu_max": 35.63614461940792,
  "mem_mean": 5777.0,
  "mem_max": 5777,
  "pod": "asr-caise-deploy-6745dcbdd4-6jpw8"
}
```

### Service diagnostics (/status)

The current state of the service may be obtained by sending a GET request with no parameters
to the `/status` endpoint. The response is in JSON format. The main purpose is checking if the
service is active, by verifying that the server returned response code 200.

Example response:
```
{
  "ts": "2026-07-03 09:33:06.312023",
  "workflow": {
    "uptime": "1 day, 22:02:12.060225",
    "snr": "Running",
    "qual": "Running",
    "asr": "Running"
  },
  "monitor": {
    "active": true,
    "mimir_thread_running": true,
    "mimir_last_status": 200
  },
  "pod": "asr-caise-deploy-6745dcbdd4-6jpw8"
}
```
