from unsupervised_measurements import transcribe, word_error_rate

wav = "output/test_candidate.wav"
text = "I have never seen this much gold."

print("Testing transcription...")
transcription = transcribe(wav)

print("Whisper heard:")
print(repr(transcription))

print("\nWER:")
print(word_error_rate(wav, text))