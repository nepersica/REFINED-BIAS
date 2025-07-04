import timm
import os
import logging
import torch
from .diffusionmodel import DiffusionNoiseModel

###########################################
#   Setting Global Parameters
###########################################
_PYTORCH_IMAGE_MODELS = "rwightman/pytorch-image-models"
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]       
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGENET_VIT_MEAN = [0.5]
IMAGENET_VIT_STD = [0.5]
NO_MEAN = [0, 0, 0]
NO_STD = [1, 1, 1]

URL_LOOKUP = {    
    # https://openreview.net/pdf?id=Bygh9j09KX: "Bias Augmentation"
    "resnet50_trained_on_SIN": "https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar",
    "resnet50_trained_on_SIN_and_IN": "https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar",
    "resnet50_trained_on_SIN_and_finetuned_on_IN": "https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar",
    # PRIME: A few primitives can boost robustness to common corruptions
    "resnet50_prime": "https://zenodo.org/record/5801872/files/ResNet50_ImageNet_PRIME_noJSD.ckpt?download=1",
    "resnet50_moco_v3_100ep": "https://dl.fbaipublicfiles.com/moco-v3/r-50-100ep/linear-100ep.pth.tar",
    "resnet50_moco_v3_300ep": "https://dl.fbaipublicfiles.com/moco-v3/r-50-300ep/linear-300ep.pth.tar",
    "resnet50_moco_v3_1000ep": "https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/linear-1000ep.pth.tar",
    # from https://github.com/kakaobrain/fast-autoaugment
    "resnet50.fastautoaugment_official": "https://arena.kakaocdn.net/brainrepo/fast-autoaugment/imagenet_resnet50_top1_22.2.pth",
    # from https://github.com/zhanghang1989/Fast-AutoAug-Torch
    "resnet50.autoaugment_270ep": "https://hangzh.s3-us-west-1.amazonaws.com/others/resnet50_aa-0cb27f8e.pth",
    "resnet50.fastautoaugment_270ep": "https://hangzh.s3-us-west-1.amazonaws.com/others/resnet50_fast_aa-3342410e.pth",
    "resnet50.randaugment_270ep": "https://hangzh.s3-us-west-1.amazonaws.com/others/resnet50_rand_aug-e38097c7.pth",
}

GID_LOOKUP = {
    "resnet50_pixmix_90ep": "1_i45yvC88hos50QjkoD97OgbDGreKdp9",
    "resnet50_pixmix_180ep": "1cgKYXDym3wgquf-4hwr_Ra3qje6GNHPH",
    "resnet50_augmix_180ep": "1z-1V3rdFiwqSECz7Wkmn4VJVefJGJGiF",
    "resnet50_deepaugment": "1DPRElQnBG66nd7GUphVm1t-5NroL7t7k",
    "resnet50_deepaugment_augmix": "1QKmc_p6-qDkh51WvsaS9HKFv8bX5jLnP",
    "resnet50_noisymix": "1Na79fzPZ0Azg01h6kGn1Xu5NoWOElSuG",
    # https://arxiv.org/abs/2010.05981: "Shape-Texture Debiased Neural Network Training"
    "resnet50_tsbias_tbias": "1tFy2Q28rAqqreaS-27jifpyPzlohKDPH",
    "resnet50_tsbias_sbias": "1p0fZ9rU-J1v943tA7PlNMLuDXk_1Iy3K",
    "resnet50_tsbias_debiased": "1r-OSfTtrlFhB9GPiQXUE1h6cqosaBePw",
    "resnet50_frozen_random": "1IT65JJauql-Jdw-0AjAhEGGApGMVx-VE",
    # Patrick Müller
    "resnet50_opticsaugment": "1y0MSlVfzZBKZQEiZF2FCOmsk-91miIHR",
    # Model with DiffusionNoise similar to Jaini et al. "Intriguing properties of generative classifiers", https://arxiv.org/abs/2309.16779
    "resnet50_diffusionnoise": "1VZz-2Du2kngTZWrQnL2CM7PNspCuJi_x",
    # SimCLRv2
    "resnet50_simclrv2": "1fDaMhujPnxo4SYOe4asPqCKQ3XB_sZvj",
}
###########################################


###########################################
#   Input-normalizing  Model Wrapper
###########################################
class NormalizedModel(torch.nn.Module):
    def __init__(self, model, mean, std):
        """
        Args:
            model: The model to wrap.
            mean: The mean to normalize the input with.
            std: The standard deviation to normalize the input with.
        """
        super(NormalizedModel, self).__init__()
        self.model = model
        self.mean = torch.nn.Parameter(
            torch.Tensor(mean).view(-1, 1, 1), requires_grad=False
        )
        self.std = torch.nn.Parameter(
            torch.Tensor(std).view(-1, 1, 1), requires_grad=False
        )

    def forward(self, x):
        out = (x - self.mean) / self.std
        out = self.model(out)
        return out
###########################################


###########################################
#   Tensorflow 
###########################################
def r50_tf_to_torch(state):

    torch_state = {}
    for k, v in state.items():

        if "blocks" not in k:
            new_key = (
                k.replace("net.0.0.", "conv1.")
                .replace("net.0.1.0.", "bn1.")
                .replace("net0.0.", "conv1.")
                .replace("net.0.1.0.", "bn1.")
            )
        else:
            s = k.split(".")
            new_key = (
                "layer"
                + s[1]
                + "."
                + s[3]
                + "."
                + (k.replace(f"net.{s[1]}.blocks.{s[3]}.", ""))
            )
            new_key = (
                new_key.replace(".net.0", ".conv1")
                .replace(".net.1.0", ".bn1")
                .replace(".net.2", ".conv2")
                .replace(".net.3.0", ".bn2")
                .replace(".net.4", ".conv3")
                .replace(".net.5.0", ".bn3")
                .replace("projection.shortcut", "downsample.0")
                .replace("projection.bn.0", "downsample.1")
            )

        torch_state[new_key] = v

    return torch_state


def load_state_dict_from_gdrive(id, model_name, force_download=False):
    state_path = os.path.join(torch.hub.get_dir(), "checkpoints", f"{model_name}.pth")

    if not os.path.exists(state_path) or force_download:
        import gdown

        logging.info(f"Downloading {id} to {state_path}")
        os.makedirs(torch.hub.get_dir(), exist_ok=True)
        gdown.download(id=id, output=state_path, quiet=False)
        # download_file_from_google_drive(id, state_path)
    state = torch.load(state_path, map_location="cpu")
    return state


def load_pretrained_model_timm(model_name, pretrained=True):
    """
    Load timm pretrained model for evaluating across learning strategy
    
    Args:
        model_name: model name
    Returns:
        model: prtrained model
    """
    model = None
    # AT ResNets from Microsoft
    if model_name.startswith("robust_resnet50"):
        model = timm.create_model("resnet50", pretrained=False)
        # get torch state from url
        if pretrained:
            tag = model_name.replace("robust_", "")
            state = torch.hub.load_state_dict_from_url(
                f"https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/{tag}.ckpt",
                map_location="cpu",
            )["model"]
            state = {k.replace("module.model.", ""): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
        model = NormalizedModel(
            model, model.default_cfg.get("mean"), model.default_cfg.get("std")
        )
    # DINOv1 -> requires handling of seperated backbone/head
    elif model_name == "resnet50_dino":
        model = timm.create_model("resnet50", pretrained=False)
        if pretrained:
            # load backbone
            state = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth",
                map_location="cpu",
            )
            model.load_state_dict(state, strict=False)
            # load classifier head
            state = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_linearweights.pth",
                map_location="cpu",
            )["state_dict"]
            state = {k.replace("module.linear.", "fc."): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
        model = NormalizedModel(
            model, model.default_cfg.get("mean"), model.default_cfg.get("std")
        )
    elif model_name == "resnet50_swav":
        model = timm.create_model("resnet50", pretrained=False)
        if pretrained:
            # load backbone
            state = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar",
                map_location="cpu",
            )
            state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
            # load classifier head
            state = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_eval_linear.pth.tar",
                map_location="cpu",
            )["state_dict"]
            state = {k.replace("module.linear.", "fc."): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
        model = NormalizedModel(
            model, model.default_cfg.get("mean"), model.default_cfg.get("std")
        )
    # Models accessible via http
    elif model_name in URL_LOOKUP:
        url = URL_LOOKUP.get(model_name)
        model = timm.create_model("resnet50", pretrained=False)
        if pretrained:
            state = torch.hub.load_state_dict_from_url(url, map_location="cpu")
            if "state_dict" in state:
                state = state["state_dict"]
            state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(state)
        model = NormalizedModel(
            model, model.default_cfg.get("mean"), model.default_cfg.get("std")
        )
    # Improved torchvision R50
    elif model_name == "tv2_resnet50":
        model = torch.hub.load(
            "pytorch/vision",
            "resnet50",
            weights="ResNet50_Weights.IMAGENET1K_V2" if pretrained else None,
        )
        model = NormalizedModel(model, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    # Shape/texture bias regularized models -> require AuxBN handling
    elif model_name.startswith("resnet50_tsbias"):
        model = timm.create_model("resnet50", pretrained=False)
        if pretrained:
            url = GID_LOOKUP.get(model_name)
            state = load_state_dict_from_gdrive(url, model_name)["state_dict"]
            state = {
                k.replace("module.", ""): v
                for k, v in state.items()
                if "aux_bn" not in k
            }
            model.load_state_dict(state)
        model = NormalizedModel(
            model, model.default_cfg.get("mean"), model.default_cfg.get("std")
        )
    # Fixed DiffusionNoise w/o noise
    elif model_name == "resnet50_diffusionnoise_fixed_nonoise":
        model = timm.create_model("resnet50", pretrained=False)
        model = DiffusionNoiseModel(model)

        if pretrained:
            state = torch.hub.load_state_dict_from_url(
                url="https://huggingface.co/paulgavrikov/resnet50.in1k_diffusionnoise_90ep/resolve/main/resnet50_diffusionnoise_nonorm_last.pth",
                map_location="cpu",
            )["model"]
            model.load_state_dict(state)

        # Note: we pass model.module here to bypass the frontend!
        model = NormalizedModel(model.module, NO_MEAN, NO_STD)  # no normalization
    # SupCon
    elif model_name == "resnet50_supcon":
        model = timm.create_model("resnet50", pretrained=False)
        if pretrained:
            url = GID_LOOKUP.get(model_name)
            state = load_state_dict_from_gdrive(url, model_name)
            state = state["model"]
            state = {
                k.replace("module.", "")
                .replace("encoder.", "")
                .replace("head.2.", "fc."): v
                for k, v in state.items()
            }
            model.load_state_dict(state, strict=False)
        model = NormalizedModel(
            model, model.default_cfg.get("mean"), model.default_cfg.get("std")
        )
    # SimCLRv2
    elif model_name == "resnet50_simclrv2":
        model = timm.create_model("resnet50", pretrained=False)
        if pretrained:
            url = GID_LOOKUP.get(model_name)
            state = load_state_dict_from_gdrive(url, model_name)
            model.load_state_dict(r50_tf_to_torch(state["resnet"]))
        model = NormalizedModel(model, NO_MEAN, NO_STD)  # no normalization

    elif model_name.startswith("file://"):
        path = model_name.replace("file://", "")
        model = timm.create_model("resnet50", pretrained=False)
        if pretrained:
            state = torch.load(path, map_location="cpu")
            if "state_dict" in state.keys():
                state = state["state_dict"]
            if "model_state_dict" in state.keys():
                state = state["model_state_dict"]
            if "online_backbone" in state.keys():
                state = state["online_backbone"]
            state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(state)
        model = NormalizedModel(
            model, model.default_cfg.get("mean"), model.default_cfg.get("std")
        )

    # R50s hosted on Google Drive
    elif model_name in GID_LOOKUP and model_name.startswith("resnet50"):
        model = timm.create_model("resnet50", pretrained=False)
        if pretrained:
            url = GID_LOOKUP.get(model_name)
            state = load_state_dict_from_gdrive(url, model_name)
            if "state_dict" in state.keys():
                state = state["state_dict"]
            elif "model_state_dict" in state.keys():
                state = state["model_state_dict"]
            elif "online_backbone" in state.keys():
                state = state["online_backbone"]
            elif "model" in state.keys():
                state = state["model"]
            state = {k.replace("module.", ""): v for k, v in state.items()}

            model.load_state_dict(state)
        model = NormalizedModel(
            model, model.default_cfg.get("mean"), model.default_cfg.get("std")
        )

    # Default to timm models
    else:
        if model_name.endswith("_untrained"):
            pretrained = False
            model_name = model_name.replace("_untrained", "")
        model = timm.create_model(model_name, pretrained=pretrained)
        model = NormalizedModel(
            model, model.default_cfg.get("mean"), model.default_cfg.get("std")
        )

    model.eval()
    return model

def load_pretrained_model_torchvision(model_name, eval=True, pretrained=True):
    """
    Load torchvision pretrained model for evaluating across model
    
    Args:
        model_name: model name
    Returns:
        model: prtrained model
    """
    model = None
    if model_name.startswith("convnext"):
        if 'tiny' in model_name:
            from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
            model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        elif 'small' in model_name:
            from torchvision.models import convnext_small, ConvNeXt_Small_Weights
            model = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
        elif 'base' in model_name:
            from torchvision.models import convnext_base, ConvNeXt_Base_Weights
            model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        elif 'large' in model_name:
            from torchvision.models import convnext_large, ConvNeXt_Large_Weights
            model = convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1)
        model = NormalizedModel(model, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    elif model_name.startswith("bagnet"):
        if '9' in model_name: 
            from .bagnets.pytorchnet import bagnet9
            model = bagnet9(pretrained=True)
        elif '17' in model_name:
            from .bagnets.pytorchnet import bagnet17
            model = bagnet17(pretrained=True)
        elif '33' in model_name:
            from .bagnets.pytorchnet import bagnet33
            model = bagnet33(pretrained=True)
        model = NormalizedModel(model, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    elif model_name.startswith("vit"):
        model = torch.hub.load(
                                _PYTORCH_IMAGE_MODELS, model_name, pretrained=True
                                )
        print(model)
        model = NormalizedModel(model, IMAGENET_VIT_MEAN, IMAGENET_VIT_STD)

    else:
        import torchvision.models as zoomodels
        model = zoomodels.__dict__[model_name](pretrained=True)
        model = NormalizedModel(model, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    model.eval()
    return model