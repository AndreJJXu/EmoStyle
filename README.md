##  EmoStyle: Emotion-aware Semantic Image Manipulation with Audio Guidance

![Teaser image](./img/exp_1.png)


**EmoStyle: Emotion-aware Semantic Image Manipulation with Audio Guidance**<br>


Qiwei Shen, Junjie Xu, Jiahao Mei, Jialing Zou, Xingjiao Wu​, Daoguo Dong <br>

Abstract: *With the flourishing development of generative models, image manipulation is receiving increasing attention. Rather than text modality, several elegant designs have delved into leveraging audio to manipulate images. However, existing methodologies mainly focus on image generation conditional on semantic alignment, ignoring the vivid affective information depicted in the audio. We propose Emotion-aware StyleGAN Manipulator (EmoStyle), a framework where affective information from audio can be explicitly extracted and further utilized during the image manipulation. Specifically, we first leverage the Multi-modality model ImageBind for initial cross-modal retrieval between images and music and select the music-related image for further manipulation. Simultaneously, by extracting sentiment polarity from the lyrics of the audio, we generate an emotionally rich auxiliary music branch to accentuate the affective information. We then leverage pre-trained encoders to encode audio and the audio-related image into the same embedding space. With the aligned embeddings, we manipulate the image via a direct latent optimization method. We conduct objective and subjective evaluations on the generated images, and our results show that our framework is capable of generating images with specified human emotions conveyed in the audio.*

## Installation
For all the methods described in the paper, is it required to have:
- Anaconda
- [CLIP](https://github.com/openai/CLIP)



## Method
![Method image](/img/model.jpg)

### Extract Lyrics of the Audio
Visit https://github.com/YuanGongND/whisper-at, download the pretrained weight to dir"./whisper_at/pretrained_models/" and deploy the pretrained *whisper_at* model to extract lyrics from the given audio.



### ChatGLM Deploy
Visit https://huggingface.co/THUDM/chatglm3-6b/tree/main, download the weight of ChatGLM3-6B and deploy the model to classify sentiment polarity of the lyrics.

### Generate-Emotional-Music
Visit https://github.com/BaoChunhui/Generate-Emotional-Music and deploy the GRU-EBS branch to generate emotional music based on the sentiment polarity.

### Download Pretrained StyleGAN2 and text-aligned audio encoder

- [pretrained-StyleGAN2](https://kr.object.ncloudstorage.com/cvpr2022/landscape.pt)
- [pretrained-audio encoder](https://kr.object.ncloudstorage.com/cvpr2022/resnet18_57.pth)


### Manipulate Image Generation

```
cd optimization

bash run.sh
```


# Sound-guided Image Manipulation

This repository contains the code and data for the project *Sound-guided Image Manipulation*.

## Citation
If you find our work useful, please cite our paper:

Shen Q, Xu J, Mei J, et al. **EmoStyle: Emotion-Aware Semantic Image Manipulation with Audio Guidance**. *Applied Sciences*, 2024, 14(8): 3193. [Link to paper](https://www.mdpi.com/journal/applsci) (Add actual link if available)

```bibtex
@article{shen2024emostyle,
  title={EmoStyle: Emotion-Aware Semantic Image Manipulation with Audio Guidance},
  author={Shen, Qiwei and Xu, Junjie and Mei, Jiahao and Wu, Xingjiao and Dong, Daoguo},
  journal={Applied Sciences},
  volume={14},
  number={8},
  pages={3193},
  year={2024},
  publisher={MDPI}
}
