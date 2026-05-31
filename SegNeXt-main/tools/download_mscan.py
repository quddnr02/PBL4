#!/usr/bin/env python
"""Download MSCAN ImageNet pretraining checkpoints for PBL SegNeXt runs."""

import argparse
import os
import urllib.request

URLS = {
    's': 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_s_20230227-f33ccdf2.pth',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Download MSCAN pretrained weights.')
    parser.add_argument('--variant', default='s', choices=sorted(URLS), help='MSCAN variant to download.')
    parser.add_argument('--out', default=None, help='Output checkpoint path.')
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(__file__)
    segnext_root = os.path.abspath(os.path.join(script_dir, '..'))
    out_path = args.out or os.path.join(segnext_root, 'pretrained', f'mscan_{args.variant}.pth')
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    url = URLS[args.variant]

    print('[PBL] Downloading official MSCAN checkpoint')
    print(f'  variant = {args.variant}')
    print(f'  url     = {url}')
    print(f'  out     = {out_path}')
    try:
        urllib.request.urlretrieve(url, out_path)
    except Exception as exc:
        print('\n[PBL] Automatic download failed.')
        print(f'  reason: {exc}')
        print('  Please download the official MSCAN-S checkpoint manually and save it as:')
        print(f'  {out_path}')
        raise SystemExit(1) from exc
    print('[PBL] Done. The config will now use this checkpoint instead of random init.')


if __name__ == '__main__':
    main()
