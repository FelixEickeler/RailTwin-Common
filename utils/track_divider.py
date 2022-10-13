import argparse
from pathlib import Path

from python.common.shared.algorithms.track_divider import track_divider
from python.common.shared.common.io import registered_reader, create_csv_reader
from python.common.shared.common.io.io_options import IOOptions
from python.common.shared.common.io.pcs_formats import ProjectPTS
from python.common.shared.common.io.read_router import scandir_supported
from python.common.shared.common.logger import RailTwinLogger

logger = RailTwinLogger.create()
project_shift = [3420000, 5370000, 0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tools for pre, post, and visualization')
    parser.add_argument('--src', type=Path, required=True)
    parser.add_argument('--out', dest="dst", type=Path, default=None, required=True)
    parser.add_argument('--filetype', dest="filetype", type=str, default=None, required=False)
    parser.add_argument('--length', dest="length", type=int, default=100, required=False)
    parser.add_argument('--local', dest="local", action='store_true', required=False)
    parser.add_argument('--ascii', dest="ascii", action='store_true', required=False)
    parser.add_argument('--xyz_accuracy', dest="xyz_accuracy", type=str, default=None, required=False)
    parser.add_argument('--intensity', dest="intensity", type=str, default=None, required=False)
    parser.add_argument('--recursive', dest="recursive", type=str, default=None, required=False)
    parser.add_argument("--flatten", dest="flatten", type=str, default=None, required=False)
    parser.add_argument("--tool", dest="tool", type=str, default="divider", required=False)
    parser.add_argument("--dst", dest="dst", type=Path, default="", required=False)
    args = parser.parse_args()

    if args.filetype == "railtwin":
        og = ProjectPTS()
        registered_reader[".pts"] = create_csv_reader(format=og, row_skip=1)
        io_options = IOOptions.do_nothing()
        io_options.filetype = ".ply"
        io_options.binary = True
        io_options.xyz_accuracy = "<f4"
        io_options.intensity = "normalized"
        io_options.shift_system = project_shift
        args.filetype = ".ply"

    else:
        io_options = IOOptions(binary=~args.ascii,
                               local=args.local,
                               xyz_accuracy=args.xyz_accuracy,
                               intensity=args.intensity,
                               reduce_to=None,
                               filetype=args.filetype,
                               custom_index=False)

    if not args.recursive:
        stack = [args.src]
    else:
        stack = scandir_supported(args.src)
        logger.info(f"Recursive option was chosen and {len(stack)}, files were identified. ")

    # process starts
    for src in stack:
        if not args.dst:
            args.dst = (args.src / args.src.stem).with_suffix(args.filetype)

        if args.recursive:
            if args.flatten:
                dst = args.dst
            else:
                rel = src.relative_to(args.src)
                rel = rel.stem / rel.with_suffix(args.filetype)
                dst = args.dst / rel
        else:
            dst = args.dst

        if dst.is_dir():
            dst.mkdir(parents=True, exist_ok=True)

        if args.tool == "divider":
            track_divider(src=src, dst=dst, length=args.length, in_options=io_options, out_options=io_options)
