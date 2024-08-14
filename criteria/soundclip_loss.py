
import torch
import clip
import torch 
from collections import OrderedDict
import math
import timm

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

class AudioEncoder(torch.nn.Module):
    def __init__(self, backbone_name="resnet18"):
        super(AudioEncoder, self).__init__()
        self.backbone_name = backbone_name
        self.conv = torch.nn.Conv2d(1, 3, (3, 3))
        # print(timm.create_model(self.backbone_name).default_cfg)
        pretrained_cfg = timm.models.create_model(self.backbone_name, num_classes=512).default_cfg
        pretrained_cfg['file'] = r'/mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/sound-guided-semantic-image-manipulation/timm/resnet18_a1_0-d63eafa0.pth'
        # print(pretrained_cfg)
        self.feature_extractor = timm.create_model(self.backbone_name, num_classes=512, pretrained=True, pretrained_cfg = pretrained_cfg)

    def forward(self, x):
        x = self.conv(x)
        x = self.feature_extractor(x)
        return x

class SoundCLIPLoss(torch.nn.Module):

    def __init__(self, opts):
        super(SoundCLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("/mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/sound-guided-semantic-image-manipulation/pretrained_models/ViT-B-32.pt", device="cuda")
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=opts.stylegan_size // 32)

        self.audio_encoder = AudioEncoder()
        
        self.audio_encoder.load_state_dict(copyStateDict(torch.load("./pretrained_models/resnet18.pth")))
        
        self.audio_encoder = self.audio_encoder.cuda()
        self.audio_encoder.eval()

    def forward(self, image, audio, audio_extra = None, text = None):
        image = self.avg_pool(self.upsample(image))
        image_features = self.model.encode_image(image).float()
        audio_features = self.audio_encoder(audio).float()
        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        #add
        if audio_extra == None:
            if text == None:
                sim_a2i = (image_features @ audio_features.T)[0] * math.exp(0.07)
                # sim_ae2i = (image_features @ audio_extra_features.T)[0] * math.exp(0.07)
                loss = 1 - sim_a2i
                # print("No Prompt!!!!!!!!!")
                return loss
            else:
                text_input = torch.cat([clip.tokenize(text)]).cuda()
                text_embedding = self.model.encode_text(text_input).float()
                text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
                sim_a2i = (image_features @ audio_features.T)[0] * math.exp(0.07)
                sim_t2i = (image_features @ audio_features.T)[0] * math.exp(0.07)
                loss = 1 - sim_a2i * 0.8 - sim_t2i * 0.2
                return loss
        audio_extra_features = self.audio_encoder(audio_extra).float()
        audio_extra_features = audio_extra_features / audio_extra_features.norm(dim=-1, keepdim=True)
        if text == None:
            sim_a2i = (image_features @ audio_features.T)[0] * math.exp(0.07)
            sim_ae2i = (image_features @ audio_extra_features.T)[0] * math.exp(0.07)
            loss = 1 - (sim_a2i*0.9 + sim_ae2i*0.1)
            return loss
        
        text_input = torch.cat([clip.tokenize(text)]).cuda()
        text_embedding = self.model.encode_text(text_input).float()
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        sim_a2i = (image_features @ audio_features.T)[0] * math.exp(0.07)
        sim_ae2i = (image_features @ audio_extra_features.T)[0] * math.exp(0.07)
        sim_t2i = (image_features @ text_embedding.T)[0] * math.exp(0.07)

        loss = 1 - sim_a2i * 0.6 - sim_ae2i * 0.2 - sim_t2i * 0.2

        return loss