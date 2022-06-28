
from google.cloud import speech


client=speech.SpeechClient.from_service_account_json('key.json')

with open('output.mp3', 'rb') as audio_file:
    mp3_data = audio_file.read() 
    audio = speech.RecognitionAudio(content=mp3_data)
    config = speech.RecognitionConfig(
        
        sample_rate_hertz=44100,
        language_code='tr-TR',
        max_alternatives=1,
        profanity_filter=False,
        enable_automatic_punctuation=True,
    )

response=client.recognize(config=config, audio=audio)
print(response.results[0].alternatives[0].transcript)
    
        