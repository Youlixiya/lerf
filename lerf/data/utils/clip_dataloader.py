import typing

import torch
from torchvision import transforms
from lerf.open_clip import create_model
from lerf.data.utils.dino_extractor import ViTExtractor
from lerf.data.utils.feature_dataloader import FeatureDataloader
from tqdm import tqdm



class CLIPDataloader(FeatureDataloader):
    clip_model_type = "ViT-B/16"
    clip_pretrained = 'laion2b_s34b_b88k'

    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        image_list: torch.Tensor,
        cache_path: str = None,
    ):
        assert "image_shape" in cfg
        super().__init__(cfg, device, image_list, cache_path)

    def create(self, image_list):
        clip_model = create_model(self.clip_model_type, pretrained=self.clip_pretrained, precision='fp16').cuda()
        patch_size = clip_model.visual.patch_size
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
        ])
        # preproc_image_lst = extractor.preprocess(image_list, self.dino_load_size)[0].to(self.device)

        clip_features = []
        for image in tqdm(image_list, desc="clip", total=len(image_list), leave=False):
            with torch.no_grad():
                image_tensor = preprocess(image)[None].cuda().half()
                clip_feature = clip_model.encode_image(image_tensor, 'ClearCLIP', True)
                feature_dim = clip_feature.shape[-1]
                h, w = image_tensor[0].shape[-2] // patch_size[0], image_tensor[0].shape[-1] // patch_size[1]
                clip_feature = clip_feature.reshape(1, h, w, feature_dim)
            clip_features.append(clip_feature.cpu().detach())

        self.data = torch.cat(clip_features, dim=0)

    def __call__(self, img_points):
        # img_points: (B, 3) # (img_ind, x, y)
        img_scale = (
            self.data.shape[1] / self.cfg["image_shape"][0],
            self.data.shape[2] / self.cfg["image_shape"][1],
        )
        x_ind, y_ind = (img_points[:, 1] * img_scale[0]).long(), (img_points[:, 2] * img_scale[1]).long()
        return (self.data[img_points[:, 0].long(), x_ind, y_ind]).to(self.device, dtype=torch.float32)
