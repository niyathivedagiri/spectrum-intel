"""Send an image over the radio link and see what arrives.

    python scripts/demo_image_link.py                          # Grace Hopper photo, QPSK, 2 dB
    python scripts/demo_image_link.py --image my.jpg --snr 4 --mod qam16 --size 64

Three receivers see the SAME transmission (same seed = same noise):
  strict    : packets failing the CRC are discarded (grey blocks)
  tolerant  : damaged packets are kept if their header is self-consistent (speckle)
  PNG file  : the image sent as a compressed PNG instead of raw pixels
Output: results/experiments/image_link_demo/<run>/
"""
import argparse

import _common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from spectrum_intel import experiments as ex
from spectrum_intel import link
from spectrum_intel import metrics as m
from spectrum_intel import payload as pl
from spectrum_intel.plots import INK, save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="path to an image (default: Grace Hopper sample)")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--gray", action="store_true")
    ap.add_argument("--mod", default="qpsk", choices=["bpsk", "qpsk", "qam16"])
    ap.add_argument("--snr", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    run = ex.new_run("image_link_demo", vars(a), a.seed)

    img = pl.load_image(a.image, a.size, a.gray)
    data = pl.image_to_bytes(img)
    n_pk = int(np.ceil(len(data) / 64))
    print(f"\nIMAGE      {img.shape} pixels -> {len(data)} bytes (8-byte header + raw pixels) "
          f"-> {len(data) * 8} bits -> {n_pk} packets of <= 64 bytes")
    print(f"CHANNEL    {a.mod.upper()} over AWGN, SNR {a.snr:g} dB per sample, random phase/timing per packet")

    out = {}
    for mode, keep in [("strict", False), ("tolerant", True)]:
        r = link.send(data, a.mod, a.snr, np.random.default_rng(a.seed), keep_corrupted=keep, fill=128)
        rx_img, hdr_ok = pl.bytes_to_image(r.received, img.shape)
        out[mode] = {"img": rx_img, "psnr": m.psnr(img, rx_img), "ssim": m.ssim(img, rx_img),
                     "pixel_errors": m.pixel_error_rate(img, rx_img), "missing": len(r.missing),
                     "corrupted": len(r.corrupted), "raw_ber": r.raw_ber(),
                     "packets_crc_ok": int(sum(p.crc_ok for p in r.packets)), "image_header_ok": hdr_ok}
        Image.fromarray(rx_img).resize((256, 256), Image.NEAREST).save(run / f"received_{mode}.png")
        print(f"{mode.upper():9s}  {out[mode]['packets_crc_ok']}/{n_pk} packets pass CRC, "
              f"{out[mode]['corrupted']} damaged kept, {out[mode]['missing']} lost -> "
              f"PSNR {out[mode]['psnr']:.1f} dB, SSIM {out[mode]['ssim']:.3f}, raw BER {out[mode]['raw_ber']:.2e}")

    png = pl.png_bytes(img)
    rp = link.send(png, a.mod, a.snr, np.random.default_rng(a.seed))
    dec = pl.decode_png(rp.received)
    png_ok = dec is not None and np.array_equal(dec, img)
    print(f"PNG FILE   {len(png)} bytes ({100 * len(png) / len(data):.0f}% of raw) in {int(np.ceil(len(png) / 64))} "
          f"packets, {len(rp.missing)} lost -> {'decoded perfectly' if png_ok else 'UNREADABLE'}")
    Image.fromarray(img).resize((256, 256), Image.NEAREST).save(run / "original.png")

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9))
    panels = [(img, "Original", f"{img.shape[0]}x{img.shape[1]}, {len(data)} bytes"),
              (out["strict"]["img"], "Strict (drop bad packets)",
               f"PSNR {out['strict']['psnr']:.1f} dB, SSIM {out['strict']['ssim']:.2f}"),
              (out["tolerant"]["img"], "Error-tolerant (keep damaged)",
               f"PSNR {out['tolerant']['psnr']:.1f} dB, SSIM {out['tolerant']['ssim']:.2f}"),
              (dec if png_ok else np.full_like(img, 230), "Compressed PNG",
               "decoded perfectly" if png_ok else f"unreadable ({len(rp.missing)} packets lost)")]
    for ax, (im, title, sub) in zip(axes, panels):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(sub, fontsize=9, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle(f"Image over a {a.mod.upper()} link at SNR {a.snr:g} dB (same noise for all three receivers)")
    save(fig, run / "image_link_comparison.png")
    ex.save_metrics(run, {k: {kk: vv for kk, vv in v.items() if kk != "img"} for k, v in out.items()}
                    | {"png": {"bytes": len(png), "packets_lost": len(rp.missing), "decoded_exact": png_ok}})
    print(f"Saved to {run}")


if __name__ == "__main__":
    main()
