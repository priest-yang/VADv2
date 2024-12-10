import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.models.backbones.base_module import BaseModule
from ..builder import BACKBONES, build_backbone, build_neck


@BACKBONES.register_module()
class VectorQuantizer(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, commitment_cost: float = 0.25, **kwargs
    ):
        super(VectorQuantizer, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, x: torch.Tensor):
        # Flatten input
        x_shape = x.shape
        flat_x = x.reshape(-1, self.embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_x**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_x, self.embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.num_embeddings, device=x.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and calculate loss
        quantized = torch.matmul(encodings, self.embedding.weight).view(x.shape)
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized.reshape(x_shape), loss, perplexity


@BACKBONES.register_module()
class VQVAE2D(BaseModule):
    """Implements the VQVAE module.
    Args:
    """

    def __init__(
        self,
        encoder: dict = None,
        quantizer: dict = None,
        decoder: dict = None,
        init_cfg: dict = None,
        img_norm_cfg: dict = None,
        **kwargs
    ):
        super(VQVAE2D, self).__init__(init_cfg, **kwargs)
        self.encoder = build_backbone(encoder)
        self.quantizer = build_backbone(quantizer)
        self.decoder = build_neck(decoder)

        self.out_channels = decoder.get("out_channels", 256)
        self.embedding_dim = quantizer.get("embedding_dim", 64)

        assert img_norm_cfg is not None, "img_norm_cfg must be provided"
        self.mean = torch.tensor(img_norm_cfg["mean"]).view(1, 3, 1, 1)
        self.std = torch.tensor(img_norm_cfg["std"]).view(1, 3, 1, 1)

    def forward(self, x: torch.Tensor):
        assert x is not None
        if isinstance(x, list):
            assert len(x) == 1, "Only support single image input"
            x = x[0]
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        feas = list(self.encoder(x))
        feas[-1], vq_loss, _ = self.quantizer(feas[-1])
        out_logits = self.decoder(feas)
        return feas[-1], vq_loss, out_logits
