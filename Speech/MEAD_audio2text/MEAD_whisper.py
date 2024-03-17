import whisper
import os
import json

root_path = './data/MEAD-Audio'
save_path = './data/MEAD_audio2text'
neutral_audio = 'neutral/level_1'
model = whisper.load_model("large")

speech_text = {}
for root, dirs, files in os.walk(root_path):
    for dir_name in dirs:
        if dir_name == 'W009':
            neutral_path = os.path.join(root, dir_name, neutral_audio)
            for audio_name in os.listdir(neutral_path):
                if not audio_name.startswith('.'):
                # load audio and pad/trim it to fit 30 seconds
                    audio_path = os.path.join(neutral_path, audio_name)
                    result = model.transcribe(audio_path)

                    # print the recognized text
                    print(audio_path)
                    print(result["text"])
                    speech_text[audio_name] = result["text"]

json_data = json.dumps(speech_text)
with open(os.path.join(save_path, 'whisper_large.json'), "w") as f:
    f.write(json_data)
