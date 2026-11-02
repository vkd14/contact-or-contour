"""Scientific contracts shared by training, inference, and regression tests."""
import hashlib
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import ndimage as ndi
import segmentation_models_pytorch as smp


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def binary_mask(mask):
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.isfinite(mask).all() or (mask < 0).any():
        raise ValueError("Expected a finite nonnegative 2D foreground mask")
    return (mask > 0).astype(np.uint8)


def grayscale_uint8(image):
    image = np.asarray(image)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Expected a finite 2D image or RGB image")
    image = image.astype(np.float32)
    lo, hi = np.percentile(image, [1, 99.5])
    return (255 * np.clip((image-lo)/(hi-lo+1e-6), 0, 1)).astype(np.uint8)


class Gabor:
    def __init__(self, orientations=8, scales=3, sigma=5., gamma=.5, size=31,
                 frequencies=None):
        if orientations < 1 or scales < 1 or sigma <= 0 or size % 2 != 1:
            raise ValueError("Invalid Gabor parameters")
        if frequencies is None:
            frequencies = np.linspace(.05, .25, scales)
        else:
            frequencies = np.asarray(frequencies, dtype=np.float32)
            if len(frequencies) != scales or not np.isfinite(frequencies).all() or (frequencies <= 0).any():
                raise ValueError("Frequencies must contain one finite positive value per scale")
        self.kernels = []
        for theta in np.linspace(0, np.pi, orientations, endpoint=False):
            for freq in frequencies:
                k = cv2.getGaborKernel((size,size), sigma, theta, 1/freq, gamma, 0, ktype=cv2.CV_32F)
                self.kernels.append(k/(np.abs(k).sum()+1e-8))

    def __call__(self, gray):
        f = gray.astype(np.float32)/255
        e = np.maximum.reduce([np.abs(cv2.filter2D(f, cv2.CV_32F, k)) for k in self.kernels])
        return ((e-e.min())/(e.max()-e.min()+1e-8)).astype(np.float32)


def features(image, edge=True, size=512, edge_kind="gabor", gabor_params=None, native=False):
    g = grayscale_uint8(image)
    rgb = np.repeat(g[...,None], 3, axis=2)
    if edge:
        if edge_kind == "gabor": e=Gabor(**(gabor_params or {}))(g)
        elif edge_kind == "sobel":
            f=g.astype(np.float32)/255
            e=cv2.magnitude(cv2.Sobel(f,cv2.CV_32F,1,0),cv2.Sobel(f,cv2.CV_32F,0,1))
            e=(e-e.min())/(e.max()-e.min()+1e-8)
        elif edge_kind == "neutral":
            # Mean 0.5 becomes exactly zero after the fourth-channel standardization.
            e=np.full(g.shape,.5,dtype=np.float32)
        else: raise ValueError(edge_kind)
        rgb = np.dstack([rgb, (e*255).astype(np.uint8)])
    x = (rgb if native else cv2.resize(rgb, (size,size), interpolation=cv2.INTER_LINEAR)).astype(np.float32)/255
    mean = np.array([.485,.456,.406]+([.5] if edge else []), dtype=np.float32)
    std = np.array([.229,.224,.225]+([.25] if edge else []), dtype=np.float32)
    x = (x-mean)/std
    if edge and edge_kind == "neutral":
        # An 8-bit value cannot represent 0.5 exactly; impose the standardized
        # zero explicitly so this is a true no-edge architecture control.
        x[...,3] = 0.
    return torch.from_numpy(x.transpose(2,0,1).copy())


def color_features(image, edge=True, size=512, edge_kind="gabor", gabor_params=None):
    """ImageNet-normalized RGB features for H&E, optionally with an edge map.

    Unlike :func:`features`, this path retains stain/color information. It is
    intentionally separate so fluorescence experiments cannot change silently
    when a histology-specific preprocessing study is run.
    """
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] != 3 or not np.isfinite(image).all():
        raise ValueError("Expected a finite 2D image or three-channel RGB image")
    if image.dtype == np.uint16:
        image = (image / 257).round().astype(np.uint8)
    elif image.dtype != np.uint8:
        lo, hi = np.percentile(image.astype(np.float32), [1, 99.5])
        image = (255 * np.clip((image - lo) / (hi - lo + 1e-6), 0, 1)).astype(np.uint8)
    channels = image
    if edge:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if edge_kind == "gabor":
            response = Gabor(**(gabor_params or {}))(gray)
        elif edge_kind == "sobel":
            values = gray.astype(np.float32) / 255
            response = cv2.magnitude(
                cv2.Sobel(values, cv2.CV_32F, 1, 0),
                cv2.Sobel(values, cv2.CV_32F, 0, 1),
            )
            response = (response - response.min()) / (response.max() - response.min() + 1e-8)
        else:
            raise ValueError(edge_kind)
        channels = np.dstack([channels, (response * 255).astype(np.uint8)])
    x = cv2.resize(channels, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
    mean = np.array([.485, .456, .406] + ([.5] if edge else []), dtype=np.float32)
    std = np.array([.229, .224, .225] + ([.25] if edge else []), dtype=np.float32)
    x = (x - mean) / std
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def sampling_weights(masks, weighted):
    if not weighted:
        return None
    w = []
    for mask in masks:
        m = binary_mask(mask)
        r = m.mean()
        e = (cv2.Canny(m*255,50,150)>0).mean()
        w.append(.3 if r<=.001 else (2+3*r)*(1.5 if r<.05 else 1)*(1+e))
    w = np.array(w)
    return w/w.mean()


def validate_split(records):
    ids, groups, digests = {}, {}, {}
    for r in records:
        for key, mapping in (("id",ids),("group",groups),("image_sha256",digests)):
            value = r.get(key)
            if value is None:
                raise ValueError(f"Missing {key}")
            if value in mapping and mapping[value] != r["split"]:
                raise ValueError(f"Cross-partition overlap in {key}: {value}")
            mapping[value] = r["split"]
    if len(ids) != len(records):
        raise ValueError("Duplicate image identifier")


class Net(nn.Module):
    def __init__(self, variant="gabor_b7", pretrained=False, classes=1):
        super().__init__()
        if variant not in ("vanilla_b7","gabor_b7","gabor_boundary_b7","scse_b7",
            "residual_gabor_b7","residual_sobel_b7","residual_gabor_boundary_b7"):
            raise ValueError(variant)
        if classes < 1:
            raise ValueError("classes must be positive")
        self.variant = variant
        self.classes = classes
        self.use_edge = variant not in ("vanilla_b7","scse_b7")
        self.residual = variant.startswith("residual_")
        self.base_model = smp.UnetPlusPlus(encoder_name="efficientnet-b7", encoder_weights="imagenet" if pretrained else None,
            in_channels=3, classes=classes, activation=None, decoder_attention_type="scse" if variant != "vanilla_b7" else None,
            decoder_channels=(256,128,64,32,16))
        if self.use_edge:
            self.input_proj = nn.Sequential(nn.Conv2d(4,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(True),
                nn.Conv2d(16,8,3,padding=1),nn.BatchNorm2d(8),nn.ReLU(True),nn.Conv2d(8,3,1))
        if self.residual: self.edge_gain=nn.Parameter(torch.zeros(()))

    def forward(self,x):
        if self.residual: x=x[:,:3]+self.edge_gain.tanh()*self.input_proj(x)
        elif self.use_edge: x=self.input_proj(x)
        return self.base_model(x)


def soft_dice(p,t):
    p,t = p.flatten(1), t.flatten(1)
    return (1-(2*(p*t).sum(1)+1)/(p.sum(1)+t.sum(1)+1)).mean()


def boundary(x):
    return F.max_pool2d(x,3,1,1)+F.max_pool2d(-x,3,1,1)


class SegmentationLoss(nn.Module):
    def __init__(self, boundary_weight=0., adaptive=False):
        super().__init__()
        self.boundary_weight, self.adaptive = boundary_weight, adaptive

    def forward(self,z,t):
        if not torch.isfinite(z).all() or not torch.isfinite(t).all():
            raise FloatingPointError("Nonfinite logits/targets; stop instead of hiding failed batches")
        if (t<0).any() or (t>1).any():
            raise ValueError("Targets must be in [0,1]")
        z,t = z.float(), t.float()
        p = z.sigmoid()
        ratio = t.mean()
        pw = torch.where(ratio>0, ((1-ratio)/ratio.clamp_min(1e-7)).clamp(1,50), torch.ones_like(ratio)) if self.adaptive else None
        loss = .5*soft_dice(p,t)+.5*F.binary_cross_entropy_with_logits(z,t,pos_weight=pw)
        if self.boundary_weight:
            loss = loss+self.boundary_weight*soft_dice(boundary(p),boundary(t))
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss")
        return loss


def logits_mask(z):
    # Never guess whether a tensor is logits based on its values or grad flag.
    return z.sigmoid()>.5


def metrics(pred, target):
    p,g = np.asarray(pred,dtype=bool), np.asarray(target,dtype=bool)
    if p.shape != g.shape or p.ndim != 2:
        raise ValueError("Prediction and reference must share a native 2D grid")
    tp,fp,fn = int((p&g).sum()),int((p&~g).sum()),int((~p&g).sum())
    denom=2*tp+fp+fn
    sp,sg=p&~ndi.binary_erosion(p),g&~ndi.binary_erosion(g)
    if sp.any() and sg.any():
        dp,dg=ndi.distance_transform_edt(~sg)[sp],ndi.distance_transform_edt(~sp)[sg]
        bp,br=float((dp<=2).mean()),float((dg<=2).mean())
        bf=2*bp*br/(bp+br) if bp+br else 0.
        nsd=float(((dp<=2).sum()+(dg<=2).sum())/(len(dp)+len(dg)))
        hd=float(np.percentile(np.r_[dp,dg],95))
    elif not sp.any() and not sg.any():
        bf,nsd,hd=1.,1.,0.
    else:
        bf,nsd,hd=0.,0.,float(np.hypot(*g.shape))
    return dict(dice=2*tp/denom if denom else 1.,iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,
        boundary_f1_2px=bf,surface_dice_2px=nsd,hd95_px=hd,empty_prediction=int(not p.any()),empty_reference=int(not g.any()),tp=tp,fp=fp,fn=fn)
