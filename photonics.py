"""Shared batched optics and one-pass inverse models. No simulation in model.forward."""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from optics import DISTANCES, WAVELENGTH, PITCH


@dataclass
class OpticalConfig:
    mode: str = 'no_lens'
    size: int = 128
    wavelength: float = WAVELENGTH
    pitch: float = PITCH
    distances: tuple = DISTANCES
    # Fixed hardware, shared by teacher, PINN training and inference.
    powers: tuple = (80., 55., 30., 10.)

    def __post_init__(self):
        if self.mode not in ('lens', 'no_lens'):
            raise ValueError('mode must be lens or no_lens')
        if self.size < 16 or self.size % 4 or len(self.distances) != 4 or len(self.powers) != 4:
            raise ValueError('size must be >=16 and divisible by 4; four planes required')
        if self.pitch <= 0 or self.wavelength <= 0 or min(self.distances) <= 0:
            raise ValueError('optical lengths must be positive')


class Simulator(nn.Module):
    """ASM on a fixed-pitch grid; lens variant reproduces lens_experiment.field.

    Lens output uses the single-FFT Fresnel grid dx=lambda*z/(N*pitch).
    Each of four switchable lenses sits immediately after the shared DOE.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        n = config.size
        f = torch.fft.fftfreq(n, d=config.pitch, dtype=torch.float64)
        fy, fx = torch.meshgrid(f, f, indexing='ij')
        kz = 2 * math.pi * torch.sqrt((1/config.wavelength**2-fx**2-fy**2).clamp_min(0))
        z = torch.tensor(config.distances, dtype=torch.float64)
        # Compute large propagation angles in double precision, store complex64.
        self.register_buffer('transfer', torch.exp(1j*z[:, None, None]*kz.double()).to(torch.complex64))
        x = (torch.arange(n)-n//2)*config.pitch
        y, x = torch.meshgrid(x, x, indexing='ij')
        q = 1/torch.tensor(config.distances)-torch.tensor(config.powers)
        self.register_buffer('chirp', q[:, None, None]*math.pi*(x*x+y*y)/config.wavelength)

    def forward(self, phase):
        if phase.ndim != 3 or phase.shape[-2:] != (self.config.size,)*2:
            raise ValueError('phase must be [B,H,W] matching optical config')
        u = torch.exp(1j*phase[:, None])
        if self.config.mode == 'no_lens':
            field = torch.fft.ifft2(torch.fft.fft2(u)*self.transfer)
        else:
            field = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(
                u*torch.exp(1j*self.chirp), dim=(-2,-1)), norm='ortho'), dim=(-2,-1))
        return field.abs().square()


def validate_targets(x, size):
    if x.ndim != 4 or x.shape[1:] != (4, size, size):
        raise ValueError('targets must be [B,4,H,W]')
    if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError('targets must be finite intensities in [0,1]')


def correlation(pred, target):
    a = pred-pred.mean((-2,-1), keepdim=True)
    b = target-target.mean((-2,-1), keepdim=True)
    # Clamp BEFORE sqrt: a uniform initial reconstruction otherwise produces
    # an infinite sqrt derivative and NaN gradients during PINN training.
    return (a*b).sum((-2,-1))/(a.square().sum((-2,-1))*b.square().sum((-2,-1))).clamp_min(1e-16).sqrt()


def objective(intensity, target):
    # Per-example loss. Equal-energy display gain; no clipping in objective.
    pred = intensity*target.mean((-2,-1), keepdim=True)
    return (1-correlation(intensity, target)).mean(1)+.1*(pred-target).square().mean((1,2,3))


def quantize(phase):
    from optics import encode_phase, decode_phase
    return decode_phase(encode_phase(phase))


def optimize(targets, simulator, steps=300, initial=None, lr=.05):
    """Offline teacher synthesis, or refinement AFTER a model prediction."""
    validate_targets(targets, simulator.config.size)
    if steps < 0:
        raise ValueError('steps cannot be negative')
    if initial is None:
        # A constant ASM field has zero intensity variance. Starting there makes
        # correlation optimization ill-conditioned and produced poor teachers.
        initial = ((torch.rand_like(targets[:,0])*2-1)*math.pi
                   if simulator.config.mode == 'no_lens' else torch.zeros_like(targets[:,0]))
    p = nn.Parameter(initial.detach().clone())
    best = p.detach().clone()
    best_loss = torch.full((len(p),), float('inf'), device=p.device)
    optimizer = torch.optim.Adam([p], lr=lr)
    for step in range(steps+1):
        loss = objective(simulator(p), targets)
        improved = loss.detach() < best_loss
        best[improved] = p.detach()[improved]
        best_loss = torch.minimum(best_loss, loss.detach())
        if step == steps:
            break
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
    return torch.remainder(best+math.pi, 2*math.pi)-math.pi


def block(a, b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1), nn.SiLU(),
                         nn.Conv2d(b,b,3,padding=1), nn.SiLU())


class UNet(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.enc1 = block(4,width)
        self.enc2 = block(width,width*2)
        self.middle = block(width*2,width*4)
        self.dec2 = block(width*6,width*2)
        self.dec1 = block(width*3,width)
        self.head = nn.Conv2d(width,2,1)

    def components(self, x):
        a = self.enc1(x)
        b = self.enc2(F.avg_pool2d(a,2))
        c = self.middle(F.avg_pool2d(b,2))
        d = self.dec2(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='nearest'),b],1))
        e = self.dec1(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='nearest'),a],1))
        return self.head(e)

    def forward(self, x):
        # Circular output avoids the -pi/+pi regression discontinuity.
        c = self.components(x)
        return torch.atan2(c[:,0], c[:,1])


class PINN(UNet):
    """Physics-informed inverse U-Net, trained through diffraction loss.

    The differentiable simulator belongs to the training objective only.
    Deployment inherits the single U-Net forward with no optical calculations.
    """
    pass


class GlobalUNet(UNet):
    """Four-level U-Net with full-image receptive field and spatial coordinates.

    Diffraction inversion is nonlocal. Coordinates distinguish mask-plane pixels
    from target-plane pixels; neither feature construction nor forward simulates optics.
    """
    def __init__(self, width=16):
        nn.Module.__init__(self)
        self.enc1=block(8,width)
        self.enc2=block(width,width*2)
        self.enc3=block(width*2,width*4)
        self.enc4=block(width*4,width*6)
        self.middle=block(width*6,width*8)
        self.dec4=block(width*14,width*6)
        self.dec3=block(width*10,width*4)
        self.dec2=block(width*6,width*2)
        self.dec1=block(width*3,width)
        self.head=nn.Conv2d(width,2,1)

    def components(self,x):
        h,w=x.shape[-2:]
        yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=x.device,dtype=x.dtype),
                             torch.linspace(-1,1,w,device=x.device,dtype=x.dtype),indexing='ij')
        coords=torch.stack([xx,yy,xx.square(),yy.square()])[None].expand(len(x),-1,-1,-1)
        a=self.enc1(torch.cat([x,coords],1))
        b=self.enc2(F.avg_pool2d(a,2))
        c=self.enc3(F.avg_pool2d(b,2))
        d=self.enc4(F.avg_pool2d(c,2))
        e=self.middle(F.avg_pool2d(d,2))
        up=lambda value,skip:F.interpolate(value,size=skip.shape[-2:],mode='nearest')
        f=self.dec4(torch.cat([up(e,d),d],1))
        g=self.dec3(torch.cat([up(f,c),c],1))
        j=self.dec2(torch.cat([up(g,b),b],1))
        k=self.dec1(torch.cat([up(j,a),a],1))
        return self.head(k)


class GlobalPINN(GlobalUNet):
    """Global inverse U-Net trained through the physical simulator."""
    pass


class FFN(nn.Module):
    """Four intensity images -> dense global MLP -> circular phase components.

    Every output pixel sees the entire quartet. No convolution, propagation,
    iterative solve or per-example trainable parameter is used in forward.
    """
    def __init__(self, size=128, hidden_width=512):
        super().__init__()
        if size < 1 or hidden_width < 1:
            raise ValueError('size and hidden_width must be positive')
        self.size = size
        self.hidden_width = hidden_width
        pixels = size*size
        self.net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(4*pixels, hidden_width),
            nn.LayerNorm(hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, 2*pixels),
        )

    def components(self, x):
        if x.ndim != 4 or x.shape[1:] != (4,self.size,self.size):
            raise ValueError(f'FFN expects [B,4,{self.size},{self.size}]')
        return self.net(x).reshape(len(x),2,self.size,self.size)

    def forward(self, x):
        c = self.components(x)
        return torch.atan2(c[:,0],c[:,1])


class FFNPINN(FFN):
    """The same one-pass FFN, trained with a differentiable optical loss."""
    pass


def make_model(kind, width=16, architecture='ffn', size=128, hidden_width=512):
    if kind not in ('supervised','pinn') or architecture not in ('local','global','ffn'):
        raise ValueError('unsupported model kind or architecture')
    if architecture == 'ffn':
        return (FFN if kind == 'supervised' else FFNPINN)(size,hidden_width)
    cls = {'local':{'supervised':UNet,'pinn':PINN},
           'global':{'supervised':GlobalUNet,'pinn':GlobalPINN}}[architecture][kind]
    return cls(width)


def load_model(path, device='cpu'):
    ck = torch.load(path, map_location=device, weights_only=True)
    if ck['kind'] not in ('supervised', 'pinn'):
        raise ValueError('expected supervised or pinn checkpoint')
    model = make_model(ck['kind'],ck.get('width',16),ck.get('architecture','local'),
                       size=ck['optics']['size'],hidden_width=ck.get('hidden_width',512)).to(device)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    return model, OpticalConfig(**ck['optics']), ck


@torch.no_grad()
def predict(model, targets):
    """Exactly one model call. Does not construct or call a simulator."""
    validate_targets(targets, targets.shape[-1])
    return model(targets)
