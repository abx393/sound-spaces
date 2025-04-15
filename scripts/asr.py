import deepspeech

# Load the pre-trained model
model_path = "/content/sound-spaces/deepspeech-0.9.3-models.pbmm"
ds = deepspeech.Model(model_path)

# Load an audio file
audio_path = "/content/output/reverberent_speech.wav"
with open(audio_path, "rb") as audio_file:
    audio_data = audio_file.read()

# Perform speech-to-text
text = ds.stt(audio_data)
print("Transcription: ", text)
