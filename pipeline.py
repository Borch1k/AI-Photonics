"""Four images -> one PINN pass -> optional refinement -> five grayscale PNGs."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from optics import encode_phase, decode_phase
from photonics import load_model, predict, Simulator, optimize

SIZE = 128
DEFAULT_REFINEMENT_STEPS = 100
MODEL_DIR = Path(__file__).resolve().parent / 'models'


def run(mode, images, output, device='cpu', model_dir=MODEL_DIR,
        refine_steps=DEFAULT_REFINEMENT_STEPS):
    if mode not in ('lens', 'no_lens'):
        raise ValueError('mode must be lens or no_lens')
    if not isinstance(refine_steps, int) or refine_steps < 0:
        raise ValueError('refine_steps must be a non-negative integer')
    if len(images) != 4:
        raise ValueError('exactly four input images are required')
    output = Path(output)
    inputs = [Path(path) for path in images]
    filenames = ['phase_mask.png'] + [f'reconstruction_{i}.png' for i in range(1, 5)]
    if any((output/name).resolve() in {p.resolve() for p in inputs} for name in filenames):
        raise ValueError('output files must not overwrite input images')
    arrays = []
    for path in inputs:
        with Image.open(path) as image:
            arrays.append(np.asarray(image.convert('L').resize(
                (SIZE, SIZE), Image.Resampling.BILINEAR), dtype=np.float32)/255)
    targets = torch.from_numpy(np.stack(arrays))[None].to(device)
    model, config, metadata = load_model(Path(model_dir)/f'pinn_{mode}.pt', device)
    if metadata['kind'] != 'pinn' or metadata.get('architecture') != 'ffn':
        raise ValueError('a trained FFN PINN checkpoint is required')
    if config.mode != mode or config.size != SIZE:
        raise ValueError('checkpoint mode/size does not match requested pipeline')

    # Strict order: the complete initial mask comes from exactly one network call.
    initial = predict(model, targets)
    if not torch.isfinite(initial).all():
        raise ValueError('model produced a non-finite phase mask')
    del model, metadata
    simulator = Simulator(config).to(device)
    phase = (optimize(targets, simulator, steps=refine_steps, initial=initial)
             if refine_steps else initial)
    output.mkdir(parents=True, exist_ok=True)
    mask_path = output/'phase_mask.png'
    Image.fromarray(encode_phase(phase[0]).cpu().numpy()).save(mask_path)

    # Reconstruct from the ACTUAL exported 8-bit mask, including quantization.
    with Image.open(mask_path) as image:
        code = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).to(device)
    with torch.no_grad():
        intensity = simulator(decode_phase(code)[None])[0]
        reconstruction = (intensity*targets[0].mean((-2, -1), keepdim=True)).clamp(0, 1)
        pixels = torch.round(reconstruction*255).to(torch.uint8).cpu().numpy()
    for index in range(4):
        Image.fromarray(pixels[index]).save(output/f'reconstruction_{index+1}.png')
    return [output/name for name in filenames]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--mode', choices=['lens', 'no_lens'], required=True)
    parser.add_argument('--images', nargs=4, required=True, metavar='IMAGE')
    parser.add_argument('--out', default='/output')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--refine-steps', type=int, default=DEFAULT_REFINEMENT_STEPS,
                        help='number of refinement updates (default: 100; 0: PINN only)')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    if args.refine_steps < 0:
        parser.error('--refine-steps must be non-negative')
    torch.set_num_threads(args.threads)
    try:
        paths = run(args.mode, args.images, args.out, args.device,
                    refine_steps=args.refine_steps)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f'Error: {error}\n')
    for path in paths:
        print(path)


if __name__ == '__main__':
    main()
