"""Extract actual DDEM Fd and reproducible feature visualizations on CPU."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import DualDegradationEstimationModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Refusing to overwrite existing output: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    ckpt = torch.load(args.weights, map_location='cpu', weights_only=False)
    state = {}
    excluded = []
    for key, value in ckpt['state_dict'].items():
        key = key[7:] if key.startswith('module.') else key
        if key.startswith('ddem.'):
            key = key[len('ddem.'):]
            if key.startswith('response_head.'):
                excluded.append(key)  # Downstream from Fd, not needed to extract Fd.
            else:
                state[key] = value
    net = DualDegradationEstimationModule(
        in_channels=3, prior_dim=32, num_blocks=2,
        backbone='local_global', num_heads=1, win_size=8).eval()
    net.load_state_dict(state, strict=True)
    del ckpt, state
    rgb = Image.open(args.input).convert('RGB')
    w, h = rgb.size
    x = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float()[None] / 255.
    pw, ph = (-w) % 32, (-h) % 32
    x = F.pad(x, (0, pw, 0, ph), mode='reflect')
    mask = torch.zeros(1, 1, h + ph, w + pw)
    mask[:, :, :h, :w] = 1
    captured = {}
    def capture(module, inputs):
        captured['fd'] = inputs[0].detach().cpu().clone()
    hook = net.z1_phi.register_forward_pre_hook(capture)
    with torch.inference_mode():
        net(x, valid_mask=mask)
    hook.remove()
    full = captured['fd']
    fd = full[0, :, :(h + 3)//4, :(w + 3)//4].numpy()
    np.save(out / 'Fd_raw_CHW.npy', fd)
    torch.save(full, out / 'Fd_padded_BCHW.pt')
    # RMS is a response-strength visualization, not a physical blur severity map.
    energy = np.sqrt(np.mean(fd.astype(np.float64)**2, axis=0))
    lo, hi = np.percentile(energy, [1, 99])
    normalized = np.clip((energy-lo)/max(hi-lo, 1e-12), 0, 1)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import colormaps
    heat = Image.fromarray((colormaps['viridis'](normalized)[..., :3]*255).round().astype('uint8'))
    heat.save(out / 'Fd_rms_heatmap.png')
    heat.resize((w, h), Image.Resampling.BILINEAR).save(out / 'Fd_rms_heatmap_fullsize.png')
    Image.fromarray((normalized*255).round().astype('uint8')).save(out / 'Fd_rms_gray.png')
    # Each channel uses independent percentile contrast; do not compare colors
    # across channel thumbnails as absolute activation magnitudes.
    thumb_w, thumb_h, label_h = 244, 204, 24
    sheet = Image.new('RGB', (8*thumb_w, 4*(thumb_h+label_h)), 'white')
    draw = ImageDraw.Draw(sheet)
    channel_dir = out / 'channels'
    channel_dir.mkdir()
    for index, ch in enumerate(fd):
        a, b = np.percentile(ch, [1, 99])
        z = np.clip((ch-a)/max(b-a, 1e-12), 0, 1)
        im = Image.fromarray((colormaps['viridis'](z)[..., :3]*255).round().astype('uint8'))
        im.save(channel_dir / f'Fd_channel_{index:02d}.png')
        ox, oy = (index%8)*thumb_w, (index//8)*(thumb_h+label_h)
        sheet.paste(im.resize((thumb_w,thumb_h), Image.Resampling.BILINEAR), (ox,oy+label_h))
        draw.text((ox+8,oy+5), f'Channel {index:02d}', fill='black')
    sheet.save(out / 'Fd_32_channels.png')
    rgb.save(out / 'input_original.png')
    meta = dict(input=str(Path(args.input).resolve()), weights=str(Path(args.weights).resolve()),
                weight_key='state_dict', input_size_wh=[w,h], padding_right_bottom=[pw,ph],
                raw_shape_chw=list(fd.shape), padded_shape_bchw=list(full.shape),
                extraction='Input of ddem.z1_phi: after stem and both DDEM blocks; before Z0/Z1/R heads.',
                visualization='sqrt(mean(Fd**2, channels)); percentile 1-99 normalization; viridis',
                rms_percentiles=[float(lo),float(hi)], device='cpu', input_scaling='RGB float32 / 255',
                excluded_downstream_response_head_keys=excluded,
                notes='No synthetic blur or resize applied. Full-image DDEM extraction, not tiled restoration. Heatmap is activation strength, not calibrated degradation severity.')
    (out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(meta,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
