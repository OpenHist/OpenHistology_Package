"""Exact compatibility overlays for the pinned transformer working trees."""

from .loader import repository


def replacements(slug):
    if slug == "transmil":
        return {
            "models/TransMIL.py": [
                (
                    "self.cls_token.expand(B, -1, -1).cuda()",
                    "self.cls_token.expand(B, -1, -1).to(h.device)",
                )
            ]
        }
    if slug == "rrt_mil":
        return {
            "modules/rrt.py": [
                ("nn.Linear(input_dim, 512)", "nn.Linear(input_dim, mlp_dim)"),
                (
                    "from torch import nn",
                    "from torch import nn\nfrom openhisto.mil.models.code_bases.compat.transformer_helpers import BagAveragePool",
                ),
                ("else nn.AdaptiveAvgPool1d(1)", "else BagAveragePool()"),
            ]
        }
    if slug == "nnmil":
        return {
            "network_architecture/models/simple_mil.py": [
                ("torch.var(eta_stack, dim=0)", "torch.var(eta_stack, dim=0, correction=0)"),
                ("torch.var(probs_stack, dim=0)", "torch.var(probs_stack, dim=0, correction=0)"),
                (
                    "torch.stack(attn_sum).std(dim=0)",
                    "torch.stack(attn_sum).std(dim=0, correction=0)",
                ),
            ]
        }
    if slug == "mamba_mil":
        init = "mamba/mamba_ssm/__init__.py"
        result = {
            init: [
                (
                    repository(slug).joinpath(init).read_text(),
                    "from .modules.mamba_simple import Mamba\nfrom .modules.srmamba import SRMamba\nfrom .modules.bimamba import BiMamba\n",
                )
            ],
            "models/MambaMIL.py": [("h = h.squeeze(0)", "h = h.squeeze(1)")],
        }
        for module in ("mamba_simple", "srmamba", "bimamba"):
            names = "selective_scan_fn, mamba_inner_fn"
            prefix = "mamba_ssm" if module == "mamba_simple" else "mamba.mamba_ssm"
            if module != "mamba_simple":
                names += ", mamba_inner_fn_no_out_proj"
            result[f"mamba/mamba_ssm/modules/{module}.py"] = [
                (
                    f"from {prefix}.ops.selective_scan_interface import {names}",
                    f"from openhisto.mil.models.code_bases.compat.mamba_reference import {names}",
                ),
                (
                    "causal_conv1d_fn, causal_conv1d_update = None",
                    "causal_conv1d_fn, causal_conv1d_update = None, None",
                ),
            ]
        return result
    if slug == "mhim_mil_v1":
        return {
            "modules/mhim.py": [
                (
                    "x,a = self.online_encoder(x,return_attn=True,no_norm=no_norm)",
                    "if self.baseline == 'dsmil':\n                x,_,a = self.online_encoder(x,return_attn=True,no_norm=no_norm)\n            else:\n                x,a = self.online_encoder(x,return_attn=True) if self.baseline == 'selfattn' else self.online_encoder(x,return_attn=True,no_norm=no_norm)",
                ),
                (
                    "DSMIL(mlp_dim=mlp_dim,mask_ratio=mask_ratio)",
                    "DSMIL(n_classes=n_classes,mlp_dim=mlp_dim,mask_ratio=mask_ratio)",
                ),
                (
                    "mask_ids =  torch.tensor(list(b.difference(a)),device=attn.device)",
                    "mask_ids = torch.tensor(sorted(b.difference(a)),device=attn.device,dtype=torch.long)",
                ),
                (
                    "mask_ids_other[:,len_keep_other:].squeeze()",
                    "mask_ids_other[:,len_keep_other:].reshape(-1)",
                ),
                ("mask_ids[:,:len_keep].squeeze()", "mask_ids[:,:len_keep].reshape(-1)"),
                ("mask_ids[:,len_keep:].squeeze()", "mask_ids[:,len_keep:].reshape(-1)"),
                (
                    "return len_keep,mask_ids\n\n    def get_mask",
                    "return max(1, len_keep),mask_ids\n\n    def get_mask",
                ),
            ]
        }
    if slug == "mhim_mil":
        return {
            "modules/__init__.py": [
                (
                    repository(slug).joinpath("modules/__init__.py").read_text(),
                    '"""Native models are imported individually by the adapter."""\n',
                )
            ],
            "modules/mhim.py": [
                (
                    "from modules.mhim_modules.merge import Merge",
                    "from modules.mhim_modules.merge import Merge\nfrom openhisto.mil.models.code_bases.compat.transformer_helpers import ArgumentIdentity",
                ),
                (
                    "mask_type='random'\n        ) if merge_enable else nn.Identity()",
                    "mask_type='random'\n        ) if merge_enable else ArgumentIdentity()",
                ),
                (
                    "        return len_keep, mask_ids\n\n    @torch.no_grad()",
                    "        if mask_ids is None:\n            mask_ids = torch.arange(ps, device=attn.device).unsqueeze(0)\n        return len_keep, mask_ids\n\n    @torch.no_grad()",
                ),
                ("mask_ids[:, :len_keep].squeeze()", "mask_ids[:, :len_keep].reshape(-1)"),
                ("mask_ids[:, len_keep:].squeeze()", "mask_ids[:, len_keep:].reshape(-1)"),
            ],
            "modules/mhim_modules/masking.py": [
                (
                    "torch.tensor(list(b.difference(a)), device=attn.device)",
                    "torch.tensor(sorted(b.difference(a)), device=attn.device, dtype=torch.long)",
                ),
                (
                    "mask_ids_other[:, len_keep_other:].squeeze()",
                    "mask_ids_other[:, len_keep_other:].reshape(-1)",
                ),
                ("return len_keep, mask_ids", "return max(1, len_keep), mask_ids"),
            ],
            "modules/mhim_modules/baseline.py": [
                ("output.append(act.squeeze(1))", "output.append(act)")
            ],
            "modules/mhim_modules/scoring.py": [
                (
                    "torch.einsum('ns,n->ns', feat.squeeze(0), attention.squeeze(0))",
                    "torch.einsum('ns,n->ns', feat.reshape(-1, feat.shape[-1]), attention.reshape(-1))",
                )
            ],
            "modules/mhim_modules/merge.py": [
                (
                    "self.global_q_mm = torch.zeros(1, k, dim)",
                    "self.global_q_mm = nn.Parameter(torch.zeros(1, k, dim), requires_grad=False)",
                ),
                ("elif not g_q_grad and g_q_mm != 1.:", "elif not g_q_grad:"),
                ("len_keep = int(L * merge_ratio)", "len_keep = min(L - 1, int(L * merge_ratio))"),
            ],
        }
    return {}
