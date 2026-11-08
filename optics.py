"""Shared phase encoding and reconstruction metrics for the lens solution."""
import math
import numpy as np
import torch
DISTANCES=(0.0241,0.0253,0.0265,0.0277)
WAVELENGTH=532e-9
PITCH=10e-6

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    def gaussian_filter(a, sigma, truncate=4.0, mode="reflect"):
        # NumPy fallback for the server environment; half-sample reflect padding.
        radius = int(truncate*sigma+0.5)
        x = np.arange(-radius,radius+1)
        kernel = np.exp(-x*x/(2*sigma*sigma)); kernel /= kernel.sum()
        result = np.asarray(a,dtype=float)
        for axis in range(result.ndim):
            pad = [(0,0)]*result.ndim; pad[axis]=(radius,radius)
            result = np.apply_along_axis(lambda v: np.convolve(v,kernel,'valid'),axis,
                                         np.pad(result,pad,mode='symmetric'))
        return result


def encode_phase(phase):
    phase = torch.where((phase >= -math.pi) & (phase <= math.pi), phase,
                        torch.remainder(phase + math.pi, 2 * math.pi) - math.pi)
    return torch.round((phase + math.pi) * 255 / (2 * math.pi)).to(torch.uint8)


def decode_phase(code):
    return code.to(torch.float32) * (2 * math.pi / 255) - math.pi


def metrics(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    a, b = x - x.mean(), y - y.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    pearson = float(np.sum(a * b) / den) if den > 1e-14 else None
    # Gaussian SSIM, sigma=1.5, 11x11, population covariance; exclude borders.
    filt = lambda a: gaussian_filter(a, sigma=1.5, truncate=3.5, mode="reflect")
    ux, uy = filt(x), filt(y)
    vx, vy = filt(x*x)-ux*ux, filt(y*y)-uy*uy
    cov = filt(x*y)-ux*uy
    s = ((2*ux*uy+0.01**2)*(2*cov+0.03**2) /
         ((ux*ux+uy*uy+0.01**2)*(vx+vy+0.03**2)))
    return {"pearson": pearson, "ssim": float(s[5:-5, 5:-5].mean()),
            "mae": float(np.abs(x-y).mean())}

