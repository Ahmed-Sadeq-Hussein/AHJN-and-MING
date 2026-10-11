# Dramatizer
This file contains intructions for getting started and instructions for running the Dramatizer.

## Getting started
These instructions assume the user is on windows and has Miniconda or Anaconda.

### 1. Clone CosyVoice2 Repository
To clone the CosyVoice2 repository, run the following command in this directory: \
`git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git CosyVoice` 

### 2. Enter environment and install requirements
Python 3.10 is recommended (but maybe not necessary). \
Activate an already-existing virtual conda environment or create a new one: \
`conda create -n cosyvoice python=3.10 -y` \
And activate it: \
`conda activate cosyvoice` \
 \
Install pynini: \
`conda install -c conda-forge pynini=2.1.5 -y` \
 \
Install the cosyvoice requirements: \
`python -m pip install -r CosyVoice/requirements.txt` \
(*there may arise problems from this) \
 \
Install additional libraries for Dramatizer: \
`python -m pip install -r requirements-dramatizer.txt`

### 3. Download CosyVoice2-0.5B model
The Dramatizer uses the cosyvoice2-0.5B model. To download it, first install huggingface_hub: \
`python -m pip install huggingface_hub` \
 \
And then, download the model to CosyVoice/pretrained_models/: \
`python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='FunAudioLLM/CosyVoice2-0.5B', local_dir='CosyVoice/pretrained_models/CosyVoice2-0.5B')"`


## Run the program
Running `python main.py` runs the Dramatizer on specified voice in the Voices/Voices/ directory using the variable "VOICE_ID". \
The paramaters of the Dramatizer are: \
* steps - The number of steps the search algorithm takes
* n_outputs - The (maximum) number of candidates the algorithm outputs in the end
* use_history - A boolean determining whether the history of voices is used to find a good starting genome

The output is stored in output/exploration/, where every step's candidate is stored along with run.json containing every step's genome.