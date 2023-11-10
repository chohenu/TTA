import logging
import torch
import torch.nn as nn
import torchvision.models as models
import timm

class Classifier(nn.Module):
    def __init__(self, args, train_target=False, checkpoint_path=None):
        super().__init__()
        self.args = args
        model = None
        self.KNOWN_MODELS = {
                'ViT-B16': 'vit_base_patch16_224_in21k', 
                'ViT-B32': 'vit_base_patch32_224_in21k',
                'ViT-L16': 'vit_large_patch16_224_in21k',
                'ViT-L32': 'vit_large_patch32_224_in21k',
                'ViT-H14': 'vit_huge_patch14_224_in21k',
                'Deit-B16':'deit_base_patch16_224.fb_in1k'
            }
        self.FEAT_DIM = {
                'ViT-B16': 768, 
                'ViT-B32': 768,
                'ViT-L16': 1024,
                'ViT-L32': 1024,
                'ViT-H14': 1280,
                'Deit-B16': 768,

            }    
        # 1) ResNet backbone (up to penultimate layer)
        if not self.use_bottleneck: 
            model = models.__dict__[args.arch](pretrained=True)
            modules = list(model.children())[:-1]
            self.encoder = nn.Sequential(*modules)
            self._output_dim = model.fc.in_features

        # 2) ResNet backbone + bottlenck (last fc as bottleneck)
        else:
            if 'ViT' in args.arch or 'Deit' in args.arch: 
                self.encoder = timm.create_model(self.KNOWN_MODELS[args.arch],pretrained=True,num_classes=0)
                self._output_dim = self.FEAT_DIM[args.arch]
            else:
                model = models.__dict__[args.arch](pretrained=True)
                model.fc = nn.Linear(model.fc.in_features, args.bottleneck_dim)
                bn = nn.BatchNorm1d(args.bottleneck_dim)
                self.encoder = nn.Sequential(model, bn)
                self._output_dim = args.bottleneck_dim

        self.fc = nn.Linear(self.output_dim, args.num_classes)
        
        if self.use_weight_norm:
            self.fc = nn.utils.weight_norm(self.fc, dim=args.weight_norm_dim)

        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)
        
        # if train_target and args.learn.do_noise_detect:
            # self.projector_q = nn.Sequential(nn.Linear(self._output_dim, self._output_dim),
            #                                 nn.BatchNorm1d(self._output_dim),
            #                                 nn.ReLU(inplace=True),
            #                                 nn.Linear(self._output_dim, int(self._output_dim/2)),
            #                                 nn.BatchNorm1d(int(self._output_dim/2)),
            #                                 )
        self.classifier_q = self.fc
    

    def forward(self, x, return_feats=False):
        # 1) encoder feature
        feat = self.encoder(x)
        feat = torch.flatten(feat, 1)
        
        # if self.args.learn.refine_method == 'detect_noisy':
        #     project_feat = self.projector_q(feat)
        #     cluster_label = self.classifier_q(feat)
        # else: 
        #     project_feat = None
        #     cluster_label = None
            
        logits = self.fc(feat)
        
        if return_feats:
            return feat, logits
        return logits

    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = dict()
        for name, param in checkpoint["state_dict"].items():
            # get rid of 'module.' prefix brought by DDP
            name = name.replace("module.", "")
            state_dict[name] = param
        msg = self.load_state_dict(state_dict, strict=False)
        logging.info(
            f"Loaded from {checkpoint_path}; missing params: {msg.missing_keys}"
        )

    def get_params(self):
        """
        Backbone parameters use 1x lr; extra parameters use 10x lr.
        """
        backbone_params = []
        extra_params = []
        # case 1)
        if not self.use_bottleneck:
            backbone_params.extend(self.encoder.parameters())
        # case 2)
        else:
            resnet = self.encoder[0]
            for module in list(resnet.children())[:-1]:
                backbone_params.extend(module.parameters())
            # bottleneck fc + (bn) + classifier fc
            extra_params.extend(resnet.fc.parameters())
            extra_params.extend(self.encoder[1].parameters())
            extra_params.extend(self.fc.parameters())

        # exclude frozen params
        backbone_params = [param for param in backbone_params if param.requires_grad]
        extra_params = [param for param in extra_params if param.requires_grad]

        return backbone_params, extra_params

    @property
    def num_classes(self):
        return self.fc.weight.shape[0]

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def use_bottleneck(self):
        return self.args.bottleneck_dim > 0

    @property
    def use_weight_norm(self):
        return self.args.weight_norm_dim >= 0


