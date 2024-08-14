import argparse
import math
import os
from librosa.core import audio

import torch
import torchvision
from torch import optim
from tqdm import tqdm
import sys
import librosa
import numpy as np
import random
import torch.nn.functional as F

from PIL import Image
import cv2

# import clip
sys.path.append("./")
from criteria.soundclip_loss import SoundCLIPLoss
from criteria.id_loss import IDLoss
from models.stylegan2.model import Generator
from utils import ensure_checkpoint_exists

#import imagebind
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

#import whisper_at
import whisper_at as whisper

#import GRU_EBS
from GRU_EBS import emotional_beam_search
import pretty_midi
from LSTM_cls import LSTMClassifier
from GRU_generator import Encoder, Decoder, Seq2Seq
from random import randint
from midi2audio import FluidSynth
from gensim.models import Word2Vec

#import chatglm
from transformers import AutoTokenizer, AutoModel

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp

def tensor2im(var):
	# var shape: (3, H, W)
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))

def imagebind_compute(model, device, text = None, image = None, audio = None):
    inputs = {
        # ModalityType.TEXT: data.load_and_transform_text(text, device),
        ModalityType.VISION: data.load_and_transform_vision_data(image, device),
        ModalityType.AUDIO: data.load_and_transform_audio_data(audio, device),
    }

    with torch.no_grad():
        embeddings = model(inputs)
    # V2T = torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=0)
    V2A = torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=0)
    # print(
    #     "Vision x Text: ",
    #     # embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T,
    #     V2T,
    # )
    print(
        "Vision x Audio: ",
        # embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T,
        V2A,
    )
    return  V2A
    # print("VISION's shape:", embeddings[ModalityType.VISION].shape)
    # print("Audio's shape:", embeddings[ModalityType.AUDIO].shape)
    # print("TEXT's shape:", embeddings[ModalityType.TEXT].shape)

    # print(
    #     "Audio x Text: ",
    #     # embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T,
    #     torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1),
    # )

def audio_caption(audio_path):
    audio_tagging_time_resolution = 10
    checkpoint_file = "/mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/audiocpation/whisper_at/pretrained_models/large-v1.pt"
    checkpoint_file_at = "/mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/audiocpation/whisper_at/pretrained_models/large-v1_ori.pth"
    model = whisper.load_model(checkpoint_file, checkpoint_file_at)
    print(audio_path)
    result = model.transcribe(audio_path, at_time_res=audio_tagging_time_resolution)
    # ASR Results
    print("audio_caption:", result["text"])
    # del model
    return result["text"]
    # Audio Tagging Results
    # audio_tag_result = whisper.parse_at_label(result, language='follow_asr', top_k=5, p_threshold=-1, include_class_list=list(range(527)))
    # print(audio_tag_result)

def imagebind_select(args, model, model_imagebind, device, mean_latent):
    image_list = []
    latent_code_inits = []
    text_inputs = []
    img_paths = []
    audio_paths = []
    for i in range(args.num_pic):
        latent_code_init_not_trunc = torch.randn(1, 512).cuda()
        with torch.no_grad():
            img_orig, latent_code_init = model([latent_code_init_not_trunc], return_latents=True,
                                        truncation=args.truncation, truncation_latent=mean_latent)
            
        torchvision.utils.save_image(img_orig, f"img_ph1/{str(i)}.jpg", normalize=True, range=(-1, 1))
        img_path_tmp = f"img_ph1/{str(i)}.jpg"
        img_paths.append(img_path_tmp)
        image_list.append(img_orig)
        latent_code_inits.append(latent_code_init)
    text_inputs.append(args.description)
    audio_paths.append(args.audio_path)
    v2a = imagebind_compute(model_imagebind, device, text_inputs, img_paths, audio_paths)

    v2a = torch.squeeze(v2a, dim = -1)
    v2v = v2a
    # v2v = v2t * args.imb_weight + v2a * (1 - args.imb_weight)
    max_sim, max_index = torch.max(v2v, dim = 0)
    img_orig = image_list[max_index]
    latent_code_init = latent_code_inits[max_index]

    # print("v2t:", v2t)
    # print("v2a:", v2a)
    # print("v2v:", v2v)
    print("Max Simi:", max_sim) 
    
    return img_orig, latent_code_init

def gru_ebs(args, wav_path):
    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    syll_model_path = '/mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/Diffusion_gen/Generate-Lyrics-and-Melody-with-Emotions/Skip-gram_lyric_encoders/syllEncoding_skipgram_dim_'+str(args.lyc2vec)+'.bin'
    word_model_path = '/mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/Diffusion_gen/Generate-Lyrics-and-Melody-with-Emotions/Skip-gram_lyric_encoders/wordLevelEncoder_skipgram_dim_'+str(args.lyc2vec)+'.bin'
    syllModel = Word2Vec.load(syll_model_path)
    wordModel = Word2Vec.load(word_model_path)

    seqlen = args.seqlen
    lyc2vec = args.lyc2vec

    generator_file = '/mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/Diffusion_gen/Generate-Lyrics-and-Melody-with-Emotions/saved_model/GRU_generator_seqlen_50_embed_128_epoch_pkl/GRU_generator_seqlen_50_embed_128_epoch_59.pkl'
    #generator_file = 'saved_model/'+'GRU_generator_'+'seqlen_'+str(seqlen)+'_embed_'+str(lyc2vec)+'_epoch_59.pkl'
    binary_clf_file = '/mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/Diffusion_gen/Generate-Lyrics-and-Melody-with-Emotions/saved_model/'+'LSTM_datasetlen_'+str(seqlen)+'_fold_7_clf.pkl'

    word_vocabulary_file = '/mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/Diffusion_gen/Generate-Lyrics-and-Melody-with-Emotions/saved_model/word_vocabulary.npy'
    word_vocabulary = np.load(word_vocabulary_file)
    word_vocabulary = word_vocabulary.item()
        
    syllable_vocabulary_file = '/mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/Diffusion_gen/Generate-Lyrics-and-Melody-with-Emotions/saved_model/syllable_vocabulary.npy'
    syllable_vocabulary = np.load(syllable_vocabulary_file)
    syllable_vocabulary = syllable_vocabulary.item()
        
    music_vocabulary_file = '/mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/Diffusion_gen/Generate-Lyrics-and-Melody-with-Emotions/saved_model/music_vocabulary_'+str(seqlen)+'.npy'
    music_vocabulary = np.load(music_vocabulary_file)
    music_vocabulary = music_vocabulary.item()

    music_index2note = [x for x in music_vocabulary.keys()]

    Ge_lyric_vocabulary = syllModel.wv.key_to_index

    len_syllable_vocabulary = len(syllable_vocabulary)
    len_word_vocabulary = len(word_vocabulary)
    len_music_vocabulary = len(music_vocabulary)
    len_Ge_lyric_vocabulary = len(Ge_lyric_vocabulary)

    binary_clf = LSTMClassifier(input_txt_size=128, input_mus_size=10, hidden_size=256, num_layers=6, num_classes=2, syllable_vocabulary=syllable_vocabulary, music_vocabulary=music_vocabulary)
    binary_clf.load_state_dict(torch.load(binary_clf_file))

    encoder = Encoder(input_size=lyc2vec*2, hidden_size=256, num_layers=4, vocabulary=Ge_lyric_vocabulary)
    decoder = Decoder(embedding_dim=100, hidden_size=256, num_layers=4, vocabulary=music_vocabulary)
    generator = Seq2Seq(encoder, decoder)
    generator.load_state_dict(torch.load(generator_file))

    generator = generator.to(device)
    binary_clf = binary_clf.to(device)

    softmax = torch.nn.Softmax(dim=0)  

    if args.emotion in ['positive', 'negative']:
        classifier = binary_clf
    else:
        classifier = None

    # seed_lyric0 = [['I'], ['give'], ['you'], ['dream']]
    # seed_lyric1 = [['but'], ['when'], ['you'], ['told'], ['me']]
    # seed_lyric2 = [['if'], ['I'], ['was'], ['your'], ['man']]
    # seed_lyric3 = [['I'], ['have'], ['a'], ['dream']]
    # seed_lyric4 = [['when'], ['I'], ['got'], ['the']]

    # seed_lyric = [['I'], ['give'], ['you'], ['my']]
    split_words = args.description.split()
    seed_lyric = [[''.join(c for c in word if c.isalpha())] for word in split_words[0:5]]
    print("seed_lyric:", seed_lyric)

    lyric_i = emotional_beam_search(softmax, syllable_vocabulary, device, syllModel, wordModel, encoder, Ge_lyric_vocabulary, music_index2note, lyc2vec, args, seed_lyric, generator, classifier, outlen = args.outlen, b1=args.b1, b2=args.b2, b3=args.b3)
    # emotional_beam_search(seed_lyric0, generator, classifier, outlen = args.outlen, b1=args.b1, b2=args.b2, b3=args.b3)

    lyric_i = ' '.join([sublist[0] for sublist in lyric_i])

    print(lyric_i)
    # 打开文件并将 seed_lyric 写入文件
    with open('/mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/sound-guided-semantic-image-manipulation/lyric.txt', "w") as file:
        file.write("seed_lyric: " + args.description + "\n" + "final_lyric: " + lyric_i)

    # emotional_beam_search(seed_lyric4, generator, classifier, outlen = args.outlen, b1=args.b1, b2=args.b2, b3=args.b3)
    fs=FluidSynth('/mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/Diffusion_gen/Generate-Lyrics-and-Melody-with-Emotions/MuseScore_General.sf2')
    fs.midi_to_audio('negative0.mid', wav_path)

def chatglm_deploy(args):
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("/mnt/ssd/BeautifulXJJ/AIGC/chatglm2-6b", device='cuda')
    model = model.eval()
    print("Ready!")
    input_str = input()
    response, history = model.chat(tokenizer, input_str, history=[])
    print(response)
    # input_str = input("输入：")
    
    # response, history = model.chat(tokenizer, input_str, history=history)
    # print("ChatGLM : ", response)
    # # print("History : ", history)
    # input_str = input("输入：")

def main(args):

    #whisper_at
    acaption = audio_caption(args.audio_path)
    args.description = acaption

    #preprocess audio input
    gru_ebs(args, args.audio_extra) #利用lyric和audio生成music
    y, sr = librosa.load(args.audio_extra, sr=44100)
    n_mels = 128
    time_length = 864
    audio_inputs = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    audio_inputs = librosa.power_to_db(audio_inputs, ref=np.max) / 80.0 + 1
    audio_inputs = audio_inputs
    zero = np.zeros((n_mels, time_length))
    resize_resolution = 512
    h, w = audio_inputs.shape
    if w >= time_length:
        j = 0
        j = random.randint(0, w-time_length)
        audio_inputs = audio_inputs[:,j:j+time_length]
    else:
        zero[:,:w] = audio_inputs[:,:w]
        audio_inputs = zero
    audio_inputs_extra = cv2.resize(audio_inputs, (n_mels, resize_resolution))
    audio_inputs_extra = np.array([audio_inputs_extra])
    audio_inputs_extra = torch.from_numpy(audio_inputs_extra.reshape((1, 1, n_mels, resize_resolution))).float().cuda()

    y, sr = librosa.load(args.audio_path, sr=44100)
    n_mels = 128
    time_length = 864
    audio_inputs = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    audio_inputs = librosa.power_to_db(audio_inputs, ref=np.max) / 80.0 + 1
    audio_inputs = audio_inputs
    zero = np.zeros((n_mels, time_length))
    resize_resolution = 512
    h, w = audio_inputs.shape
    if w >= time_length:
        j = 0
        j = random.randint(0, w-time_length)
        audio_inputs = audio_inputs[:,j:j+time_length]
    else:
        zero[:,:w] = audio_inputs[:,:w]
        audio_inputs = zero
    audio_inputs = cv2.resize(audio_inputs, (n_mels, resize_resolution))
    audio_inputs = np.array([audio_inputs])
    audio_inputs = torch.from_numpy(audio_inputs.reshape((1, 1, n_mels, resize_resolution))).float().cuda()

    os.makedirs(args.results_dir, exist_ok=True)

    #load stylegan checkpoint
    ensure_checkpoint_exists(args.ckpt)
    g_ema = Generator(args.stylegan_size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()
    mean_latent = g_ema.mean_latent(4096)
    layer_masking_weight = torch.ones(14)

    if args.latent_path:
        latent_code_init = torch.load(args.latent_path).cuda()
        print(latent_code_init)
        with torch.no_grad():
            img_orig, _ = g_ema([latent_code_init], input_is_latent=True, randomize_noise=False)

    elif args.mode == "edit":
        #imagebind module
        device = "cuda:1" if torch.cuda.is_available() else "cuda:0"
        model_imagebind = imagebind_model.imagebind_huge(pretrained=True)
        model_imagebind.eval()
        model_imagebind.to(device)
        img_orig, latent_code_init = imagebind_select(args, g_ema, model_imagebind, device, mean_latent)
        del model_imagebind
        

    else:
        latent_code_init = mean_latent.detach().clone().repeat(1, 18, 1)

    latent = latent_code_init.detach().clone()
    torch.save(latent, 'latent_code.pt')
    print("latent:", latent)
    img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False)
    torchvision.utils.save_image(img_gen, f"original.jpg", normalize=True, range=(-1, 1))

    latent.requires_grad = True
    soundclip_loss = SoundCLIPLoss(args)
    id_loss = IDLoss(args)
    optimizer = optim.Adam([latent], lr=args.lr)
    pbar = tqdm(range(args.step))

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr
        img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False)
        cosine_distance_loss = soundclip_loss(img_gen, audio_inputs, audio_inputs_extra, args.description)
        if args.mode == "edit":
            if not args.adaptive_layer_masking:
                similarity_loss = ((latent_code_init - latent) ** 2).sum()
            else:
                similarity_loss = 0
                for idx in range(14):
                    layer_per_loss = F.sigmoid(layer_masking_weight[idx]) * ((latent_code_init[:,idx,:] - latent[:,idx,:]) ** 2).sum()
                    similarity_loss += layer_per_loss
                    layer_masking_weight[idx] = layer_masking_weight[idx] - 0.1 * layer_per_loss.item() * (1 - layer_per_loss.item())
                
            loss = args.lambda_similarity * similarity_loss + cosine_distance_loss  + args.lambda_identity * id_loss(img_orig, img_gen)[0]

        else:
            loss = cosine_distance_loss

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        pbar.set_description(
            (
                f"loss: {loss.item():.4f};"
            )
        )
        if args.save_intermediate_image_every > 0 and i % args.save_intermediate_image_every == 0:
            with torch.no_grad():
                img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False)
            torchvision.utils.save_image(img_gen, f"results/{str(i).zfill(5)}.jpg", normalize=True, range=(-1, 1))
        
        


    
    if args.mode == "edit":
        with torch.no_grad():
            img_orig, _ = g_ema([latent_code_init], input_is_latent=True, randomize_noise=False)

        final_result = torch.cat([img_orig, img_gen])
    else:
        final_result = img_gen

    return final_result



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", type=str, default=None, help="the text that guides the editing/generation")
    parser.add_argument("--audio_path", type=str, default="/mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/sound-guided-semantic-image-manipulation/data/test.wav")
    parser.add_argument("--ckpt", type=str, default="./pretrained_models/stylegan2-ffhq-config-f.pt", help="pretrained StyleGAN2 weights")
    parser.add_argument("--stylegan_size", type=int, default=1024, help="StyleGAN resolution")
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--step", type=int, default=300, help="number of optimization steps")
    parser.add_argument("--mode", type=str, default="edit", choices=["edit", "free_generation"], help="choose between edit an image an generate a free one")
    parser.add_argument("--lambda_similarity", type=float, default=0.008, help="weight of the latent distance (used for editing only)")
    parser.add_argument("--lambda_identity", type=float, default=0.005, help="weight of the identity loss")
    parser.add_argument("--latent_path", type=str, default=None, help="starts the optimization from the given latent code if provided. Otherwose, starts from"
                                                                      "the mean latent in a free generation, and from a random one in editing. "
                                                                       "Expects a .pt format")
    parser.add_argument("--truncation", type=float, default=0.7, help="used only for the initial latent vector, and only when a latent code path is"
                                                                      "not provided")
    parser.add_argument("--save_intermediate_image_every", type=int, default=20, help="if > 0 then saves intermidate results during the optimization")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--adaptive_layer_masking", type=bool, default=False)
    parser.add_argument("--save_latent_path", type=str, default=None)
    parser.add_argument("--save_source_image_path", type=str, default=None)
    parser.add_argument("--save_manipulated_image_path", type=str, default=None)
    parser.add_argument("--save_manipulated_latent_code_path", type=str, default=None)
    parser.add_argument("--num_pic", type=int, default=1, help="number of image generated in the first pharse")
    parser.add_argument("--imb_weight", type=float, default=0.5, help="weight of the v2a and v2t")
    #GRU_EBS
    parser.add_argument('--seqlen', type=int, default=50)
    parser.add_argument('--lyc2vec', type=int, default=128)
    parser.add_argument('--outlen', type=int, default=30)
    parser.add_argument('--b1', type=int, default=3)
    parser.add_argument('--b2', type=int, default=3)
    parser.add_argument('--b3', type=int, default=5)
    parser.add_argument('--emotion', default='negative')
    parser.add_argument('--output_num', type=int, default=1)
    parser.add_argument("--audio_extra", type=str, default="/mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/sound-guided-semantic-image-manipulation/tmp.wav")
    args = parser.parse_args()
    # chatglm_deploy(args)
    result_image = main(args)

    torchvision.utils.save_image(result_image.detach().cpu(), os.path.join(args.results_dir, "final_result.jpg"), normalize=True, scale_each=True, range=(-1, 1))


