import argparse
from pathlib import Path
import yaml
from .charuco import generate_board_image
from .live import live_capture
from .solver import solve_wrist, solve_external, inspect_all
from .intrinsics import capture_intrinsics, solve_intrinsics


def load_config(filename):
    path = Path(filename).expanduser().resolve()
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    # Relative paths resolve relative to the working directory. This matches README: run at project root.
    for name in ('dataset', 'output'):
        cfg['paths'][name] = str(Path(cfg['paths'][name]).expanduser().resolve())
    for camera in cfg['cameras'].values():
        camera['intrinsics'] = str(Path(camera['intrinsics']).expanduser().resolve())
    if cfg['robot'].get('pose_file'):
        cfg['robot']['pose_file'] = str(Path(cfg['robot']['pose_file']).expanduser().resolve())
    return cfg


def main():
    ap = argparse.ArgumentParser(description='Online Wrist + Third Person -> Robot Base calibration')
    ap.add_argument('--config', default='configs/example.yaml', help='Run at project root or supply absolute file paths')
    sub = ap.add_subparsers(dest='command', required=True)
    board = sub.add_parser('board', help='Generate printable ChArUco image')
    board.add_argument('--out', default='board.png')
    board.add_argument('--square-pixels', default=240, type=int)
    for name in ('capture', 'solve'):
        p = sub.add_parser(name)
        p.add_argument('--stage', choices=['wrist', 'cam1', 'cam2'], required=True)
    intr = sub.add_parser('intrinsics', help='ChArUco RGB intrinsic calibration before external hand-eye stages')
    intr_sub = intr.add_subparsers(dest='intrinsics_command', required=True)
    intr_capture = intr_sub.add_parser('capture', help='Live capture; no robot FK required')
    intr_capture.add_argument('--camera', choices=['wrist', 'cam1', 'cam2'], required=True)
    intr_solve = intr_sub.add_parser('solve', help='Estimate K/D from saved ChArUco images')
    intr_solve.add_argument('--camera', choices=['wrist', 'cam1', 'cam2', 'all'], required=True)
    intr_solve.add_argument('--image-dir', help='Optional image folder for a single camera')
    intr_solve.add_argument('--overwrite', action='store_true', help='Back up and replace existing non-placeholder intrinsics')
    validation = sub.add_parser('validate', help='Project the real-time robot EEF XYZ pose on wrist/cam1/cam2 images')
    validation.add_argument('--images-dir', help='Offline mode: folder with wrist.png, cam1.png, cam2.png')
    validation.add_argument('--pose-file', help='Offline mode: JSON containing T_base_gripper')
    validation.add_argument('--save-dir', help='Output folder, default: output/projection_validation')
    validation.add_argument('--origin-offset-m', type=float, nargs=3, default=(0., 0., 0.),
                            metavar=('X','Y','Z'), help='Point in robot G frame [m], e.g. physical TCP')
    validation.add_argument('--axis-length-m', type=float, default=0.05)
    validation.add_argument('--points-json', help='Optional physical reference points {name:[x,y,z]} in G frame [m]')
    sub.add_parser('inspect')
    sub.add_parser('simulate', help='Generate a synthetic dataset and solve all stages (not real hardware)')
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.command == 'board':
        generate_board_image(cfg['board'], args.out, args.square_pixels)
        print(f"Generated {args.out}. PRINT WITH PHYSICAL DIMENSIONS EXACTLY: "
              f"{cfg['board']['squares_x']*cfg['board']['square_length_m']*1000:.1f} x "
              f"{cfg['board']['squares_y']*cfg['board']['square_length_m']*1000:.1f} mm; no fit-to-page!")
    elif args.command == 'intrinsics':
        if args.intrinsics_command == 'capture':
            capture_intrinsics(cfg, args.camera)
        else:
            if args.camera == 'all' and args.image_dir:
                ap.error('--image-dir cannot be used with --camera all')
            names = ['wrist', 'cam1', 'cam2'] if args.camera == 'all' else [args.camera]
            for name in names:
                solve_intrinsics(cfg, name, image_dir=args.image_dir, overwrite=args.overwrite)
    elif args.command == 'capture':
        live_capture(cfg, args.stage)
    elif args.command == 'solve':
        if args.stage == 'wrist':
            solve_wrist(cfg['paths']['dataset'], cfg['paths']['output'])
        else:
            solve_external(cfg['paths']['dataset'], cfg['paths']['output'], args.stage)
    elif args.command == 'validate':
        from .projection import validate_projection
        validate_projection(cfg, args.images_dir, args.pose_file, args.save_dir,
                            args.origin_offset_m, args.axis_length_m, args.points_json)
    elif args.command == 'inspect':
        inspect_all(cfg['paths']['output'])
    elif args.command == 'simulate':
        from .simulation import run_synthetic
        run_synthetic(cfg)


if __name__ == '__main__':
    main()
