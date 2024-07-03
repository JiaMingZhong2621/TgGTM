import torch
import torch.nn as nn
from collections import OrderedDict
# from utils import split_first_dim_linear
import math
from itertools import combinations 

from torch.autograd import Variable

from tqdm import tqdm
NUM_SAMPLES=1
from torch import einsum
import torch.nn.functional as F
import numpy as np
from itertools import combinations
from einops import rearrange
import os
from typing import Tuple, Union
import gzip
import html
import os
from functools import lru_cache
from einops.layers.torch import Rearrange
import ftfy
import regex as re
import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List
from pkg_resources import packaging
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
# from lavis.models import load_model_and_preprocess
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# text_model, _, _ = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)

# model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
SSV2_TRAIN =  ['Pouring [something] into [something]', 'Poking a stack of [something] without the stack collapsing', 'Pretending to poke [something]', 'Lifting up one end of [something] without letting it drop down', 'Moving [part] of [something]', 'Moving [something] and [something] away from each other', 'Removing [something], revealing [something] behind', 'Plugging [something] into [something]', 'Tipping [something] with [something in it] over, so [something in it] falls out', 'Stacking [number of] [something]', "Putting [something] onto a slanted surface but it doesn't glide down", 'Moving [something] across a surface until it falls down', 'Throwing [something] in the air and catching it', 'Putting [something that cannot actually stand upright] upright on the table, so it falls on its side', 'Holding [something] next to [something]', 'Pretending to put [something] underneath [something]', "Poking [something] so lightly that it doesn't or almost doesn't move", 'Approaching [something] with your camera', 'Poking [something] so that it spins around', 'Pushing [something] so that it falls off the table', 'Spilling [something] next to [something]', 'Pretending or trying and failing to twist [something]', 'Pulling two ends of [something] so that it separates into two pieces', 'Lifting up one end of [something], then letting it drop down', "Tilting [something] with [something] on it slightly so it doesn't fall down", 'Spreading [something] onto [something]', 'Touching (without moving) [part] of [something]', 'Turning the camera left while filming [something]', 'Pushing [something] so that it slightly moves', 'Uncovering [something]', 'Moving [something] across a surface without it falling down', 'Putting [something] behind [something]', 'Attaching [something] to [something]', 'Pulling [something] onto [something]', 'Burying [something] in [something]', 'Putting [number of] [something] onto [something]', 'Letting [something] roll along a flat surface', 'Bending [something] until it breaks', 'Showing [something] behind [something]', 'Pretending to open [something] without actually opening it', 'Pretending to put [something] onto [something]', 'Moving away from [something] with your camera', 'Wiping [something] off of [something]', 'Pretending to spread air onto [something]', 'Holding [something] over [something]', 'Pretending or failing to wipe [something] off of [something]', 'Pretending to put [something] on a surface', 'Moving [something] and [something] so they collide with each other', 'Pretending to turn [something] upside down', 'Showing [something] to the camera', 'Dropping [something] onto [something]', "Pushing [something] so that it almost falls off but doesn't", 'Piling [something] up', 'Taking [one of many similar things on the table]', 'Putting [something] in front of [something]', 'Laying [something] on the table on its side, not upright', 'Lifting a surface with [something] on it until it starts sliding down', 'Poking [something] so it slightly moves', 'Putting [something] into [something]', 'Pulling [something] from right to left', 'Showing that [something] is empty', 'Spilling [something] behind [something]', 'Letting [something] roll down a slanted surface', 'Holding [something] behind [something]']
SSV2_TEST = ['Twisting (wringing) [something] wet until water comes out', 'Poking a hole into [something soft]', 'Pretending to take [something] from [somewhere]', 'Putting [something] upright on the table', 'Poking a hole into [some substance]', 'Rolling [something] on a flat surface', 'Poking a stack of [something] so the stack collapses', 'Twisting [something]', '[Something] falling like a feather or paper', 'Putting [something] on the edge of [something] so it is not supported and falls down', 'Pushing [something] off of [something]', 'Dropping [something] into [something]', 'Letting [something] roll up a slanted surface, so it rolls back down', 'Pushing [something] with [something]', 'Opening [something]', 'Putting [something] on a surface', 'Taking [something] out of [something]', 'Spinning [something] that quickly stops spinning', 'Unfolding [something]', 'Moving [something] towards the camera', 'Putting [something] next to [something]', 'Scooping [something] up with [something]', 'Squeezing [something]', 'Failing to put [something] into [something] because [something] does not fit']
HMDB_TRAIN = ['brush hair', 'catch', 'chew', 'clap', 'climb', 'climb stairs', 'dive', 'draw_sword', 'dribble', 'drink', 'fall floor', 'flic flac', 'handstand', 'hug', 'jump', 'kiss', 'pullup', 'punch', 'push', 'ride_bike', 'ride horse', 'shake_hands', 'shoot_bow', 'situp', 'stand', 'sword', 'sword exercise', 'throw', 'turn', 'walk', 'wave']
HMDB_TEST = ['fencing', 'kick', 'kick ball', 'pick', 'pour', 'pushup', 'run', 'sit', 'smoke', 'talk']
K100_TRAIN = ['air drumming', 'arm wrestling', 'beatboxing', 'biking through snow', 'blowing glass', 'blowing out candles', 'bowling', 'breakdancing', 'bungee jumping', 'catching or throwing baseball', 'cheerleading', 'cleaning floor', 'contact juggling', 'cooking chicken', 'country line dancing', 'curling hair', 'deadlifting', 'doing nails', 'dribbling basketball', 'driving tractor', 'drop kicking', 'dying hair', 'eating burger', 'feeding birds', 'giving or receiving award', 'hopscotch', 'jetskiing', 'jumping into pool', 'laughing', 'making snowman', 'massaging back', 'mowing lawn', 'opening bottle', 'playing accordion', 'playing badminton', 'playing basketball', 'playing didgeridoo', 'playing ice hockey', 'playing keyboard', 'playing ukulele', 'playing xylophone', 'presenting weather forecast', 'punching bag', 'pushing cart', 'reading book', 'riding unicycle', 'shaking head', 'sharpening pencil', 'shaving head', 'shot put', 'shuffling cards', 'slacklining', 'sled dog racing', 'snowboarding', 'somersaulting', 'squat', 'surfing crowd', 'trapezing', 'using computer', 'washing dishes', 'washing hands', 'water skiing', 'waxing legs', 'weaving basket']
K100_TEST = ['blasting sand',  'busking',  'cutting watermelon',  'dancing ballet', 'dancing charleston',  'dancing macarena',  'diving cliff', 'filling eyebrows', 'folding paper',  'hula hooping', 'hurling (sport)',  'ice skating',  'paragliding', 'playing drums',  'playing monopoly', 'playing trumpet', 'pushing car', 'riding elephant',  'shearing sheep', 'side kick', 'stretching arm', 'tap dancing', 'throwing axe',  'unboxing']
UCF_TRAIN = ['Apply Eye Makeup', 'Archery', 'Baby Crawling', 'Balance Beam', 'Band Marching', 'Baseball Pitch', 'Basketball', 'Basketball Dunk', 'Bench Press', 'Biking', 'Billiards', 'Blow DryHair', 'Body Weight Squats', 'Bowling', 'Boxing Punching Bag', 'Boxing Speed Bag', 'Breast Stroke', 'Brushing Teeth', 'Cricket Bowling', 'Drumming', 'Fencing', 'Field Hockey Penalty', 'Frisbee Catch', 'Front Crawl', 'Haircut', 'Hammering', 'Head Massage', 'Hula Hoop', 'Javelin Throw', 'Juggling Balls', 'Jumping Jack', 'Kayaking', 'Knitting', 'Long Jump', 'Lunges', 'Military Parade', 'Mixing', 'Mopping Floor', 'Nunchucks', 'Parallel Bars', 'Pizza Tossing', 'Playing Cello', 'Playing Dhol', 'Playing Flute', 'Playing Piano', 'Playing Sitar', 'Playing Tabla', 'Playing Violin', 'Pole Vault', 'Pull Ups', 'Push Ups', 'Rafting', 'Rope Climbing', 'Rowing', 'Shaving Beard', 'Skijet', 'Soccer Juggling', 'Soccer Penalty', 'Sumo Wrestling', 'Swing', 'Table Tennis Shot', 'Tai Chi', 'Throw Discus', 'Trampoline Jumping', 'Typing', 'Uneven Bars', 'Walking WithDog', 'Wall Pushups', 'Writing On Board', 'Yo Yo']
UCF_TEST = ['Blowing Candles', 'Clean And Jerk', 'Cliff Diving', 'Cutting in Kitchen', 'Diving', 'Floor Gymnastics', 'Golf Swing', 'Handstand Walking', 'Horse Race', 'Ice Dancing', 'Jump Rope', 'Pommel Horse', 'Punch', 'Rock Climbing Indoor', 'Salsa Spin', 'Skiing', 'Sky Diving', 'Still Rings', 'Surfing', 'Tennis Swing', 'Volleyball Spiking']
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)#（5000,512）
        position = torch.arange(0, max_len).unsqueeze(1)#形状（max_len,1）
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor#偶数列是sin
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor#奇数列是cos
        pe = pe.unsqueeze(0)#维度为（1，max_len，d_model）
        self.register_buffer('pe', pe)#保存位置编码张量
                          
    def forward(self, x):#pe[:, :x.size(1)]的维度为（1,seq_len,d_model）
       x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)#不需要计算梯度，因为位置是固定的
       return self.dropout(x)
            
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")

@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out

__all__ = ["available_models", "load", "tokenize"]
_tokenizer = SimpleTokenizer()

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None, spatial=False):
    """Load a CLIP model "cuda" if torch.cuda.is_available() else "cpu"
    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model
    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).
    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"
    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    with open(model_path, 'rb') as opened_file:
        try:
            # loading JIT archive
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    if not jit:
        model = build_model(state_dict or model.state_dict()).to(device)
        if str(device) == "cpu":
            model.float()
        return model, _transform(model.visual.input_resolution)

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, _transform(model.input_resolution.item())


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 77 as the context length
    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result





class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None, spatial=False):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.spatial = spatial

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC

        if self.spatial:
            if self.spatial=="v2":
                cls_token, _ = F.multi_head_attention_forward(
                            query=x[:1], key=x, value=x,
                            embed_dim_to_check=x.shape[-1],
                            num_heads=self.num_heads,
                            q_proj_weight=self.q_proj.weight,
                            k_proj_weight=self.k_proj.weight,
                            v_proj_weight=self.v_proj.weight,
                            in_proj_weight=None,
                            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
                            bias_k=None,
                            bias_v=None,
                            add_zero_attn=False,
                            dropout_p=0,
                            out_proj_weight=self.c_proj.weight,
                            out_proj_bias=self.c_proj.bias,
                            use_separate_proj_weight=True,
                            training=self.training,
                            need_weights=False
                            )
                x_mid = self.v_proj(x[1:])
                x_mid = self.c_proj(x_mid)
                x = torch.cat([cls_token, x_mid], dim=0)
                return x.squeeze(0)
            else:

                x, _ = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=self.num_heads,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
                in_proj_weight=None,
                in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
                bias_k=None,
                bias_v=None,
                add_zero_attn=False,
                dropout_p=0,
                out_proj_weight=self.c_proj.weight,
                out_proj_bias=self.c_proj.bias,
                use_separate_proj_weight=True,
                training=self.training,
                need_weights=False
                )
                # return torch.cat([x, self.c_proj(x_copy)], dim=0)
                return x.squeeze(0)
        else:
            x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
            )
            return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, spatial=False):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim, spatial)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 spatial=False,
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,   # (3, 4, 6, 3)
                output_dim=embed_dim,   # 1024
                heads=vision_heads,     # 32
                input_resolution=image_resolution,    # 224
                width=vision_width,      # 64
                spatial=spatial
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))
    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, spatial=False):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, spatial=spatial,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()


class Up2(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, kernel_size=2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=kernel_size, mode='bilinear', align_corners=True)
            self.conv = DoubleConv2(in_channels, out_channels, in_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size, stride=kernel_size, groups=1)
            self.conv = DoubleConv2(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, groups=8),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv2(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def cos_sim(x, y, epsilon=0.01):
    """
    Calculates the cosine similarity between the last dimension of two tensors.计算余弦相似度
    """
    numerator = torch.matmul(x, y.transpose(-1,-2))
    xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
    ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(xnorm, ynorm.transpose(-1,-2)) + epsilon
    dists = torch.div(numerator, denominator)
    return dists


def extract_class_indices(labels, which_class):
    """
    Helper method to extract the indices of elements which have the specified label.
    :param labels: (torch.tensor) Labels of the context set.
    :param which_class: Label for which indices are extracted.
    :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
    """
    class_mask = torch.eq(labels, which_class)  
    class_mask_indices = torch.nonzero(class_mask, as_tuple=False)  
    class_mask = torch.reshape(class_mask_indices, (-1,))
    return class_mask  
def OTAM_cum_dist_v2(dists, lbda=0.5):
    """
    Calculates the OTAM distances for sequences in one direction (e.g. query to support).
    :input: Tensor with frame similarity scores of shape [n_queries, n_support, query_seq_len, support_seq_len] 
    TODO: clearn up if possible - currently messy to work with pt1.8. Possibly due to stack operation?
    """
    dists = F.pad(dists, (1,1), 'constant', 0)  

    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        # cum_dists[:,:,0,m] = dists[:,:,0,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,0,m-1]))
        # paper does continuous relaxation of the cum_dists entry, but it trains faster without, so using the simpler version for now:
        cum_dists[:,:,0,m] = dists[:,:,0,m] + cum_dists[:,:,0,m-1] 


    # remaining rows
    for l in range(1,dists.shape[2]):
        #first non-zero column
        cum_dists[:,:,l,1] = dists[:,:,l,1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,0] / lbda) + torch.exp(- cum_dists[:,:,l-1,1] / lbda) + torch.exp(- cum_dists[:,:,l,0] / lbda) )
        
        #middle columns
        for m in range(2,dists.shape[3]-1):
            cum_dists[:,:,l,m] = dists[:,:,l,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,m-1] / lbda) + torch.exp(- cum_dists[:,:,l,m-1] / lbda ) )
            
        #last column
        #cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
        cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l-1,-1] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
    
    return cum_dists[:,:,-1,-1]
def mask_matrix():
    size = 18
    mask = torch.eye(size, dtype=torch.float32)
    mask[torch.arange(size - 1), torch.arange(1, size)] = 1  
    mask[torch.arange(1, size), torch.arange(size - 1)] = 1 

    matrix = torch.zeros(size, size)

    matrix[:9, 9:] = 1

    matrix[9:, :9] = 1
    x = [8,9]
    y = [9,8]
    matrix[x,y]=0
    
    return matrix + mask
class Node_Edge(nn.Module):
    def __init__(self, base_c, dropout=0.0):
        """
        :param base_c: number of base channel
        :param device: the gpu device stores tensors
        :param dropout: dropout rate
        """
        super(Node_Edge, self).__init__()
        self.base_c = base_c
        self.dropout = dropout

        self.mask = mask_matrix()
        layer_list = []

        layer_list += [nn.Conv2d(in_channels=self.base_c, out_channels=self.base_c*2, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c*2),
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        layer_list += [nn.Conv2d(in_channels=self.base_c*2, out_channels=self.base_c, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c),
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        layer_list += [nn.Conv2d(in_channels=self.base_c, out_channels=1, kernel_size=1)]
        self.point_sim_transform = nn.Sequential(*layer_list)

    def forward(self, node, last_edge,distance_metric):

        vp_i = node.unsqueeze(2)
        vp_j = torch.transpose(vp_i, 1, 2)
        if distance_metric == 'l2':
            vp_similarity = (vp_i - vp_j)**2
        elif distance_metric == 'l1':
            vp_similarity = torch.abs(vp_i - vp_j) 
        trans_similarity = torch.transpose(vp_similarity, 1, 3)
        ep_ij = self.point_sim_transform(trans_similarity).squeeze(1)

        mask = self.mask.to(ep_ij.device)
        ep_ij = ep_ij * mask

        I = torch.eye(node.size(1)).unsqueeze(0).to(ep_ij.get_device())

        ep_ij = F.softmax(ep_ij+I,dim=-1).unsqueeze(3)#【5,18,18,1】
        return torch.cat([last_edge.unsqueeze(3),ep_ij],3)#[5,18,18,2]

class Init_Edge(nn.Module):
    def __init__(self, base_c, dropout=0.0):
        """
        :param base_c: number of base channel
        :param device: the gpu device stores tensors
        :param dropout: dropout rate
        """
        super(Init_Edge, self).__init__()
        self.base_c = base_c
        self.dropout = dropout

        self.mask = mask_matrix()

        layer_list = []

        layer_list += [nn.Conv2d(in_channels=self.base_c, out_channels=self.base_c*2, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c*2),
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        layer_list += [nn.Conv2d(in_channels=self.base_c*2, out_channels=self.base_c, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c),
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        layer_list += [nn.Conv2d(in_channels=self.base_c, out_channels=1, kernel_size=1)]
        self.point_sim_transform = nn.Sequential(*layer_list)

    def forward(self, node,distance_metric):

        vp_i = node.unsqueeze(2)#[5,18,1,512]
        vp_j = torch.transpose(vp_i, 1, 2)#[5,1,18,512]
        if distance_metric == 'l2':
            vp_similarity = (vp_i - vp_j)**2
        elif distance_metric == 'l1':
            vp_similarity = torch.abs(vp_i - vp_j) #[5,18,18,512]
        trans_similarity = torch.transpose(vp_similarity, 1, 3)#[5,512,18,18]
        
        ep_ij = self.point_sim_transform(trans_similarity).squeeze(1)#Ψ(|V|


        mask = self.mask.to(ep_ij.device)
        ep_ij = ep_ij * mask#[5,18,18]

        I = torch.eye(node.size(1)).unsqueeze(0).to(ep_ij.get_device())#[1,18,18]

        ep_ij = F.softmax(ep_ij+I,dim=-1)#E˜= Softmax (Ψ(|V| + I))
        return ep_ij
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features

        self.proj = nn.Linear(in_features*2, in_features)  # input_features, out_features

    def forward(self, x, W):
        x_size = x.size()#[5,18,512]
        W_size = W.size()#[5,18,18,2]
        N = W_size[-2]
        W = W.split(1, 3) 
        W = torch.cat(W, 1).squeeze(3) #[5,36,18]
        output = torch.bmm(W, x)# [5,36,512]
        output = output.split(N, 1)#
        output = torch.cat(output, 2)#[5,18,1024]
        output = self.proj(output)
        return output
   
class CrossGNN(nn.Module):
    def __init__(self, num_generations,emb_size,dropout, seq_len, point_metric):
        super(CrossGNN, self).__init__()
        self.generation = num_generations
        self.dropout = dropout
        self.emb_size = emb_size
        self.seq_len = seq_len
        self.point_metric = point_metric
        max_len = int(self.seq_len  * 1.5)
        self.pe = PositionalEncoding(emb_size, 0.1, max_len=max_len)   
        initial_edge = Init_Edge(self.emb_size, dropout=self.dropout)
        self.add_module('initial_edge', initial_edge)
        for l in range(self.generation):
            self.add_module('gcn_{}'.format(l), GraphConvolution(self.emb_size))
            self.add_module('node_edge_{}'.format(l), Node_Edge(self.emb_size, dropout=self.dropout if l < self.generation-1 else 0.0))
           
    def forward(self,context):


        q = self.pe(context[:,:9,:])
        s = self.pe(context[:,9:,:])

        protype_node = torch.cat([q,s],dim=1)

        last_edge = self._modules['initial_edge'](protype_node, self.point_metric)
        for l in range(self.generation):
            last_edge = self._modules['node_edge_{}'.format(l)](protype_node,last_edge, self.point_metric)
            protype_node = self._modules['gcn_{}'.format(l)](protype_node, last_edge)
            last_edge = last_edge.mean(-1)
        return protype_node

class CNN_GCN_FSAR(nn.Module):
    def __init__(self,args):
        super(CNN_GCN_FSAR, self).__init__()
        self.args = args
        self.clip_model,_ = load("ViT-B/16", device="cpu", jit=False)   # ViT-B/16
        self.backbone =  self.clip_model.visual   # model.load_state_dict(state_dict)
        choices={"ssv2":[SSV2_TRAIN,SSV2_TEST], "kinetics":[K100_TRAIN,K100_TEST], "hmdb":[HMDB_TRAIN,HMDB_TEST], "ucf":[UCF_TRAIN,UCF_TEST]}
        self.class_real_train = choices[self.args.dataset][0]
        self.class_real_test =  choices[self.args.dataset][1]
        self.mid_dim = 512
        
        self.class_token = nn.Parameter(torch.randn(1, 1, self.mid_dim))
        
        self.cross_gnn = CrossGNN(3,self.mid_dim,0.1,16, 'l1')
        with torch.no_grad():
            text_templete = ["a photo of a {}".format(self.class_real_test[int(ii)]) for ii in range(len(self.class_real_test))]
            text_templete = tokenize(text_templete)
            self.text_features_test =self.clip_model.encode_text(text_templete).cuda()
       
            text_templete = ["a photo of a {}".format(self.class_real_train[int(ii)]) for ii in range(len(self.class_real_train))]
            text_templete = tokenize(text_templete)
            
            self.text_features_train = self.clip_model.encode_text(text_templete).cuda()
        self.classification_layer = nn.Sequential() 
        self.scale = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.scale.data.fill_(1.0)


    def distribute_model(self):
        self.backbone.cuda(0)
        self.backbone = torch.nn.DataParallel(self.backbone, device_ids=[i for i in range(0, 3)])
        self.scale.cuda(0)
        self.cross_gnn.cuda(0)
                                         
    def get_feats(self, support_images, target_images):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        if self.training:
            support_features = self.backbone(support_images).squeeze()
            # os.system("nvidia-smi")
            target_features = self.backbone(target_images).squeeze()
            # os.system("nvidia-smi")
            # Decr.mean(target_features, dim = 1) # 160 x 2048

            dim = int(target_features.shape[1])#dim=512,[200,512]
            support_features = support_features.reshape(-1, 8, dim)
            target_features = target_features.reshape(-1, 8, dim)
            
        else:
            support_features = self.backbone(support_images).squeeze()
            # os.system("nvidia-smi")
            target_features = self.backbone(target_images).squeeze()

            dim = int(target_features.shape[1])
            support_features = support_features.reshape(-1, 8, dim)
            target_features = target_features.reshape(-1, 8, dim)

        return support_features, target_features

    def forward(self, inputs):
        support_images, support_labels, target_images, support_real_class = inputs['support_set'][0], inputs['support_labels'][0], inputs['target_set'][0], inputs['support_real_class'][0]
        # support_images, support_labels, target_images, support_real_class = inputs['support_set'], inputs['support_labels'], inputs['target_set'], inputs['support_real_class']

        support_features, target_features= self.get_feats(support_images, target_images)
        support_labels = support_labels.cuda()#25
        support_real_class = support_real_class.cuda()#25

        if self.training:
            context_support = self.text_features_train[support_real_class.long()].unsqueeze(1)#.repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1)
            
        else:
            context_support = self.text_features_test[support_real_class.long()].unsqueeze(1)#.repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1) # .repeat(support_bs+target_bs, 1, 1)
        
        unique_labels = torch.unique(support_labels)
        class_indices = [extract_class_indices(support_labels, c) for c in unique_labels]

        # Calculate support features for each class，support_features[25,8,512]
        support_features = torch.stack([torch.mean(torch.index_select(support_features, 0, indices), dim=0) for indices in class_indices])#【5,8,512】
        context_support = torch.stack([torch.mean(torch.index_select(context_support, 0, indices), dim=0) for indices in class_indices])#【5,1,512】
        
        # Split target features based on training/testing
        num_chunks = 25 if self.training else 5
        target_feature_chunks = torch.chunk(target_features, num_chunks, dim=0)

        q_nodes = []
        s_nodes = []
        
        # Process target features in chunks
        support_features = torch.cat((support_features, context_support), dim=1)#[5,9,512]
        for q in target_feature_chunks:
            # Perform cross GNN on concatenated data
            target_features = torch.cat((q.repeat(support_features.shape[0],1,1),self.class_token.expand(support_features.shape[0], -1, -1)), dim=1)#【5,9,512】
            all_video_frames_node= self.cross_gnn(torch.cat((target_features,support_features),dim=1))#【5,18,512】

            support_protype_node = all_video_frames_node[:, 9:, :]
            query_protype_node = torch.mean(all_video_frames_node[:, :9, :], dim=0)

            q_nodes.append(query_protype_node)
            s_nodes.append(support_protype_node)

        target_features = torch.stack(q_nodes)#[25,9,512]
        support_features = torch.mean(torch.stack(s_nodes), dim=0)#[5,9,512]
        n_queries = target_features.shape[0]#25
        n_support = support_features.shape[0]#5

        feature_classification_in= target_features[:,-1,:]
        feature_classification = self.classification_layer(feature_classification_in)
        class_text_logits = cos_sim(feature_classification, self.text_features_train)*self.scale

        support_features = rearrange(support_features, 'b s d -> (b s) d')  
        target_features = rearrange(target_features, 'b s d -> (b s) d')    

        frame_sim = cos_sim(target_features, support_features)
            
        frame_dists = 1 - frame_sim
                
        dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb = n_queries, sb = n_support)  # [25, 25, 8, 8]

        # calculate query -> support and support -> query

        cum_dists = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))
        class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)
        class_dists = rearrange(class_dists, 'c q -> q c')


        return_dict = {'logits': - class_dists,"class_text_logits":class_text_logits}
    
        return return_dict
