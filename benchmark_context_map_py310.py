import argparse
from contextlib import contextmanager
import os
import shutil
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.special import gammaln

import CNNImageCodec as codec

@contextmanager
def chdir(path):
    previous_dir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous_dir)


ROOT = Path(__file__).resolve().parent
DIRECTIONS = ('left', 'above', 'mean')
PRIOR_STRENGTHS = {1: 16, 2: 64, 3: 256, 4: 1024}
PIXELS = codec.w * codec.h


def load_images(folder, limit):
    files = sorted(Path(folder).glob('*.png'), key=lambda path: int(path.stem))[:limit]
    images = []
    for path in files:
        with codec.Image.open(path) as image:
            images.append(np.asarray(image.convert('RGB')))
    return files, np.asarray(images, dtype=np.float64) / 255


def configure(bits):
    codec.ALPHABET_SIZE = 2 ** bits
    codec.CONTEX_SIZE = codec.ALPHABET_SIZE * codec.n3M1
    codec.freqs = np.ones((codec.CONTEX_SIZE, codec.ALPHABET_SIZE + 1), dtype=np.int64)
    codec.cumfreqs = np.zeros(codec.ALPHABET_SIZE + 1, dtype=np.int64)


def make_prior(counts, strength):
    smoothed = counts + 1.0
    alphabet = codec.ALPHABET_SIZE
    grouped = smoothed.reshape(alphabet, -1, alphabet)
    marginal = grouped.sum(axis=0)
    grouped += 24 * marginal / marginal.sum(axis=-1, keepdims=True)
    return 1 + np.rint(strength * smoothed / smoothed.sum(axis=1, keepdims=True)).astype(np.int64)


def ideal_bytes(counts, prior):
    totals = prior.sum(axis=-1)
    numerator = gammaln(prior + counts) - gammaln(prior)
    denominator = gammaln(totals + counts.sum(axis=-1)) - gammaln(totals)
    return (denominator.sum() - numerator.sum()) / (8 * np.log(2))


def measure(path, latent, mode, prior=None):
    codec.EntropyEncoder(path, latent, *latent.shape, mode=mode, prior=prior)
    restored = codec.EntropyDecoder(path, *latent.shape, mode=mode, prior=prior)
    if not np.array_equal(latent, restored):
        raise RuntimeError(f'Entropy round trip failed: {path}, {mode}')
    return path.stat().st_size


def symbol_features(latents):
    left = np.zeros_like(latents)
    above = np.zeros_like(latents)
    left[:, :, 1:] = latents[:, :, :-1]
    above[:, 1:] = latents[:, :-1]
    mean = (left + above + 1) // 2
    mean[:, 0] = left[:, 0]
    mean[:, :, 0] = above[:, :, 0]
    left[:, :, 0] = above[:, :, 0]
    above[:, 0] = left[:, 0]
    flat = latents.reshape(len(latents), -1)
    previous = np.zeros_like(flat)
    previous[:, 1:] = flat[:, :-1]
    return dict(zip(DIRECTIONS, (left, above, mean))), previous.reshape(latents.shape)


def select_context_map(latents, strength):
    spatial, previous = symbol_features(latents)
    alphabet = codec.ALPHABET_SIZE
    channels = latents.shape[-1]
    check = np.arange(len(latents)) % 4 == 0
    directions, guide_channels, priors = [], [], []
    for channel in range(channels):
        symbols = latents[..., channel]
        guides = {-1: previous[..., channel], -2: np.zeros_like(symbols)}
        guides.update({c: latents[..., c] for c in range(max(0, channel - 1))})
        best_score = np.inf
        for direction, neighbors in spatial.items():
            for guide, guide_symbols in guides.items():
                context = neighbors[..., channel] + alphabet * guide_symbols
                keys = (context * alphabet + symbols).reshape(len(latents), -1)
                counts = np.stack([np.bincount(row, minlength=alphabet ** 3) for row in keys])
                counts = counts.reshape(len(latents), alphabet ** 2, alphabet)
                prior = make_prior(counts[~check].sum(axis=0), strength)
                score = ideal_bytes(counts[check], prior)
                if score < best_score:
                    best_score = score
                    best_direction, best_guide = direction, guide
                    best_counts = counts.sum(axis=0)
        directions.append(best_direction)
        guide_channels.append(best_guide)
        priors.append(make_prior(best_counts, strength))
    mode = codec.EntropyContextMap(tuple(directions), tuple(guide_channels))
    return mode, np.stack(priors, axis=1).reshape(-1, alphabet)


def write_csv(path, header, rows):
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def bd_rate_in_window(baseline, improved, quality, min_bpp=0.2, max_bpp=0.9):
    """Integrate log rate over the shared SSIM interval within the task's bpp limits."""
    curves = [PchipInterpolator(quality, np.log(rate)) for rate in (baseline, improved)]
    lo, hi = quality[0], quality[-1]
    lower, upper = [lo], [hi]
    for curve in curves:
        for limit, bounds in ((np.log(min_bpp), lower), (np.log(max_bpp), upper)):
            if curve(lo) < limit < curve(hi):
                bounds.append(brentq(lambda ssim: curve(ssim) - limit, lo, hi))
        if curve(hi) <= np.log(min_bpp) or curve(lo) >= np.log(max_bpp):
            raise ValueError('No measured operating points in the requested bpp window')
    lo, hi = max(lower), min(upper)
    if lo >= hi:
        raise ValueError('No shared SSIM interval within the requested bpp window')
    delta = (curves[1].integrate(lo, hi) - curves[0].integrate(lo, hi)) / (hi - lo)
    return 100 * np.expm1(delta), lo, hi


def write_bd_rates(output, baseline, improved, quality):
    bd_rate = codec.bj_delta(baseline, quality, improved, quality, mode=1)
    scoped, lo, hi = bd_rate_in_window(baseline, improved, quality)
    message = (f'BD-Rate (all measured points): {bd_rate:+.3f}%\n'
               f'BD-Rate (0.2–0.9 bpp, PCHIP, common SSIM {lo:.4f}–{hi:.4f}): {scoped:+.3f}%\n')
    print(message, end='', flush=True)
    (output / 'bd-rate.txt').write_text(message)


def evaluate(output, winners, test_files, test_images):
    codec.plt.switch_backend('Agg')
    Path('jpg').mkdir(exist_ok=True)
    results, summary = [], []
    panel = None
    for bits in range(1, 5):
        configure(bits)
        mode, prior, test_symbols, scales, decoder = winners[bits]
        reconstructed = decoder.predict(codec.DequantizeLatents(test_symbols, scales, bits), verbose=0)
        archive = output / 'compressed' / f'b{bits}'
        for folder in ('baseline', 'improved', 'jpeg'):
            (archive / folder).mkdir(parents=True, exist_ok=True)
        np.save(archive / 'scales.npy', np.asarray(scales), allow_pickle=False)
        shutil.copyfile(ROOT / f'models/decoder_1_{bits}.h5', archive / 'decoder.h5')
        metadata = {
            'bits': bits,
            'latent_shape': list(test_symbols.shape[1:]),
            'latent_dtype': str(test_symbols.dtype),
            'image_order': [file.name for file in test_files],
            'baseline_mode': 'previous',
            'context_maps': f'../../context-maps.json',
            'prior': f'../../learned-both-prior-b{bits}.npy',
            'scales': 'scales.npy',
            'decoder': 'decoder.h5',
        }
        (archive / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        rows = []
        jpeg_panels = []
        for i, (file, source, latent, restored) in enumerate(
                zip(test_files, test_images, test_symbols, reconstructed)):
            original_size = measure(archive / 'baseline' / f'{file.stem}.bin', latent, 'previous')
            improved_size = measure(archive / 'improved' / f'{file.stem}.bin', latent, mode, prior)
            quality = codec.ssim(source, restored, data_range=1.0, channel_axis=-1)
            original_bpp = original_size * 8 / PIXELS
            improved_bpp = improved_size * 8 / PIXELS
            _, jpeg_bpp, jpeg_quality = codec.JPEGRDSingleImage(source, original_bpp, i)
            shutil.copyfile(Path('jpg') / f'image{i}.jpeg', archive / 'jpeg' / f'{file.stem}.jpeg')
            rows.append((original_bpp, improved_bpp, quality, jpeg_bpp, jpeg_quality))
            results.append((bits, file.name, original_size, improved_size, *rows[-1]))
            if bits == 2 and i < 5:
                with codec.Image.open(f'jpg/image{i}.jpeg') as image:
                    jpeg_panels.append(np.array(image))
        summary.append((bits, *np.mean(rows, axis=0)))
        if bits == 2:
            panel = (test_images[:5], reconstructed[:5], reconstructed[:5], jpeg_panels, rows[:5])

    print(f'Compressed test files saved to: {output / "compressed"}', flush=True)
    write_csv(output / 'test.csv',
              ('b', 'image', 'baseline_bytes', 'improved_bytes', 'baseline_bpp', 'improved_bpp',
               'ae_ssim', 'jpeg_bpp', 'jpeg_ssim'), results)
    write_csv(output / 'summary.csv',
              ('b', 'baseline_bpp', 'improved_bpp', 'ae_ssim', 'jpeg_bpp', 'jpeg_ssim'), summary)
    _, baseline, improved, quality, jpeg, jpeg_quality = np.array(summary).T
    write_bd_rates(output, baseline, improved, quality)
    fig, ax = codec.plt.subplots(figsize=(8, 5))
    for rate, ssim, name in ((baseline, quality, 'Baseline AE'), (improved, quality, 'Improved AE'),
                             (jpeg, jpeg_quality, 'JPEG')):
        ax.plot(rate, ssim, 'o-', label=name)
    ax.set(xlabel='bpp', ylabel='Mean SSIM')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(output / 'rd-performance.png', dpi=180, bbox_inches='tight')
    codec.plt.close(fig)

    fig, axes = codec.plt.subplots(4, 5, figsize=(13, 9), layout='constrained')
    for row, (images, name) in enumerate(zip(panel[:4], ('RAW', 'Baseline AE', 'Improved AE', 'JPEG'))):
        for col, image in enumerate(images):
            ax = axes[row, col]
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(name)
            base, new, ssim, jpg, jssim = panel[4][col]
            title = test_files[col].name if row == 0 else (
                f'{(base, new, jpg)[row - 1]:.3f} bpp / {(ssim, ssim, jssim)[row - 1]:.3f} SSIM')
            ax.set_title(title, fontsize=9)
    fig.savefig(output / 'comparison-b2.png', dpi=180)
    codec.plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/context-map'))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    _, train_images = load_images(ROOT / 'train', None)
    test_files, test_images = load_images(ROOT / 'test', 20)
    if len(test_files) != 20:
        raise ValueError('The report requires 20 test images')
    fit = np.arange(len(train_images)) % 5 != 0

    validation, configurations, winners = [], {}, {}
    with TemporaryDirectory(prefix='cnn-codec-') as workdir:
        scratch = Path(workdir) / 'scratch.bin'
        for bits, strength in PRIOR_STRENGTHS.items():
            configure(bits)
            encoder = codec.keras.models.load_model(ROOT / f'models/encoder_1_{bits}.h5', compile=False)
            decoder = codec.keras.models.load_model(ROOT / f'models/decoder_1_{bits}.h5', compile=False)
            train, _ = codec.QuantizeLatents(encoder.predict(train_images, batch_size=8, verbose=0), bits)
            mode, prior = select_context_map(train[fit], strength)
            baseline_bytes = sum(measure(scratch, latent, 'previous') for latent in train[~fit])
            improved_bytes = sum(measure(scratch, latent, mode, prior) for latent in train[~fit])
            validation.append((bits, baseline_bytes, improved_bytes))
            print(f'b={bits}: validation {baseline_bytes} -> {improved_bytes} bytes', flush=True)
            np.save(output / f'learned-both-prior-b{bits}.npy', prior)
            configurations[bits] = {
                'prior_strength': strength, 'spatial': mode.spatial, 'guides': mode.guides,
            }
            test, scales = codec.QuantizeLatents(encoder.predict(test_images, batch_size=8, verbose=0), bits)
            winners[bits] = (mode, prior, test, scales, decoder)

        write_csv(output / 'roundtrip-validation.csv', ('b', 'baseline_bytes', 'improved_bytes'), validation)
        (output / 'context-maps.json').write_text(json.dumps(configurations, indent=2) + '\n')
        with chdir(workdir):
            evaluate(output, winners, test_files, test_images)


if __name__ == '__main__':
    main()
