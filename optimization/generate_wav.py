from midi2audio import FluidSynth
fs=FluidSynth('/mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/Diffusion_gen/Generate-Lyrics-and-Melody-with-Emotions/MuseScore_General.sf2')
fs.midi_to_audio('negative0.mid', 'output1.wav')
# FluidSynth().midi_to_audio('input.mid', 'output.wav')


# input_path = '/mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/Diffusion_gen/Generate-Lyrics-and-Melody-with-Emotions/positive0.mid'
# output_path= '/mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/Diffusion_gen/Generate-Lyrics-and-Melody-with-Emotions/output.wav'
# 调用函数进行转换
# midi_to_wav(input_path, output_path)

# input:  /mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/Diffusion_gen/Generate-Lyrics-and-Melody-with-Emotions/positive0.mid
# output: /mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/Diffusion_gen/Generate-Lyrics-and-Melody-with-Emotions/output.wav