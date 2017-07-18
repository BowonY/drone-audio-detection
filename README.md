# Real time UAV sound detection
## Credits
https://github.com/NeilYager/LittleSleeper

## Requirements
* Python 3.x

* Python speech features
`pip install python_speech_features`

* Pyaudio
Install Portaudio
`brew install portaudio`
Create/add this:
``` bash
[build_ext]
include_dirs=/usr/local/Cellar/portaudio/19.6/include/
library_dirs=/usr/local/Cellar/portaudio/19.6/lib/
```
Run this:
`pip install --allow-external pyaudio --allow-unverified pyaudio pyaudio`

* tornado
`pip install tornado`

* mongodb
`brew install mongodb`

## How to run
* `python audio_server.py`
* `python web_server.py`

## How to access
* Type in localhost:8090/ in browser
