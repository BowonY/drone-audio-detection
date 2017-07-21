# Real time UAV sound detection
## Credits
https://github.com/NeilYager/LittleSleeper

## Requirements
### Python 3.x or Anaconda3

### Python speech features
`pip install python_speech_features`

### Pyaudio
#### Windows
`pip install pyaudio`

#### OSX
Install Portaudio
`brew install portaudio`
Create/add this:
``` bash
cat >> ~/.pydistutils.cfg

[build_ext]
include_dirs=/usr/local/Cellar/portaudio/19.6/include/
library_dirs=/usr/local/Cellar/portaudio/19.6/lib/
```
Run this:
`pip install --allow-external pyaudio --allow-unverified pyaudio pyaudio`

### tornado
`pip install tornado`

### mongodb (not necessary now)
`brew install mongodb`

## How to run
* `python audio_server.py`
* `python web_server.py`

## How to access
* Type in localhost:8090/ in browser

## How to use Git
   1. Before start
      ``` bash
      $ git clone https://github.com/<your-id>/<your-repo-name>.git
      $ git remote add origin https://github.com/<your-id>/<your-repo-name>.git
      $ git remote add upstream https://github.com/<orig-id>/<orig-repo-name>.git
      ```
   1. How to update forked repository (to master branch)
      ``` bash
      $ git fetch upstream
      $ git checkout master
      $ git rebase upstream/master
      ```
   1. How to make 'pull request'
      ``` bash
      $ git commit -m "your comment"
      $ git push
      ```
      after you pushed commit to your repository, go to your repository and make pull request using 'New pull request' button.
   * If you got a error like `Cannot pull with rebase: You have unstaged changes.`
      ``` bash
      $ git stash
      $ git pull or $ git fetch
      $ git stash pop
      ```