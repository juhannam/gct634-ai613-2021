# Homework #4: Pop Music Transformer [Colab notebook](https://colab.research.google.com/drive/1QzZfKiBv5oxBWS9f0HkZJpUZJjUoB80r?usp=sharing)

Generate pop piano music using Transformer models. In this task, we use REMI, which stands for `REvamped MIDI-derived events.` REMI is a new event representation for converting MIDI scores into text-like discrete tokens. 

- Experience MIDI, REMI data processing
- Experience Transformer architecture

## **Getting Started**

### **Install Dependencies**

- python 3.6
- tensorflow-gpu 1.14.0 (`pip install tensorflow-gpu==1.14.0`)
- [miditoolkit](https://github.com/YatingMusic/miditoolkit) (`pip install miditoolkit`)

### **Download Pre-trained Checkpoints**

We can use two pre-trained checkpoints for generating samples.

- `REMI-tempo-checkpoint`
- `REMI-tempo-chord-checkpoint` [(Optional)](https://drive.google.com/open?id=1nAKjaeahlzpVAX0F9wjQEG_hL4UosSbo)

### **Obtain the MIDI Data**

We can use the MIDI files including local tempo changes and estimated chord. [(5 MB)](https://drive.google.com/open?id=1JUDHGrVYGyHtjkfI2vgR1xb2oU8unlI3)

- `data/train`: 775 files used for training models
- `data/evaluation`: 100 files (prompts) used for the continuation experiments

## Tasks and Deliverables

### Task 1  : MIDI to Event representation conversion

1. Find your favorite piano songs
2. Transcribe the piano songs using the pre-trained onset and frames model or your own model (HW3)
3. Import your MIDI files, convert it to REMI and convert it back to MIDI file

You can try adjusting the variables. e.g. Modify the quantize_items' parameter

You should submit 

- Transcribed MIDI file
- REMI events.pkl
- Restored MIDI file(quantized)

### Task 2  : Fine-tune the model with your own MIDI files

1. Find your favorite piano songs
2. Transcribe the piano songs using the pre-trained onset and frames model or your own model (HW3)
3. Using transcription results to fine-tune the Pop Music Transformer pre-trained model

You should submit  

- **MIDI file used for training**
- **Various generation results**
- **Report** about the learning process and subjective evaluatio

FYI, We are not gonna judge the musical quality of the results

## How to synthesize the audio files?

We strongly recommend using DAW and VSTI : [Reaper](https://www.reaper.fm/) (FREE DAW), [Spitfire audio LABS](https://labs.spitfireaudio.com/#type=&search=&new=true) (Free Piano VSTI) to open/play the generated MIDI files. Or, you can use [FluidSynth](https://github.com/FluidSynth/fluidsynth) with a [SoundFont](https://sites.google.com/site/soundfonts4u/). However, it may not be able to correctly handle the tempo changes (see fluidsynth/issues/141).

## Reference

reference : [pop music transformer](https://github.com/YatingMusic/remi)