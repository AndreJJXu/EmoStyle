import whisper_at as whisper

audio_tagging_time_resolution = 10
checkpoint_file = "/mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/audiocpation/whisper_at/pretrained_models/large-v1.pt"
checkpoint_file_at = "/mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/audiocpation/whisper_at/pretrained_models/large-v1_ori.pth"
model = whisper.load_model(checkpoint_file, checkpoint_file_at)

# model = whisper.load_model("large-v1","large-v1")
# result = model.transcribe("/mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/audiocpation/whisper_at/outputs/ferrari.mp3", at_time_res=audio_tagging_time_resolution)
result = model.transcribe("/mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/sound-guided-semantic-image-manipulation/data/test.wav", at_time_res=audio_tagging_time_resolution)
# ASR Results
print(result["text"])
# Audio Tagging Results
audio_tag_result = whisper.parse_at_label(result, language='follow_asr', top_k=5, p_threshold=-1, include_class_list=list(range(527)))
print(audio_tag_result)