# 06.05.22----------------------------------------------------------------------------------------------------------------------
#  created by: Felix Eickeler 
#              felix.eickeler@tum.de       
# ----------------------------------------------------------------------------------------------------------------
#
#
import argparse
from pathlib import Path

from python.common.shared.common.io import registered_reader, create_csv_reader
from python.common.shared.common.io.io_options import IOOptions
from python.common.shared.common.io.pcs_formats import ProjectPTS
from python.common.shared.common.io.read_router import scandir_supported
from python.common.shared.common.logger import RailTwinLogger
from python.common.shared.common.io import read
from python.common.shared.common.io import write


logger = RailTwinLogger.create()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tools for pre, post, and visualization')
    parser.add_argument('--src', type=Path, required=True)
    parser.add_argument('--out', dest="dst", type=Path, default=None, required=True)
    parser.add_argument('--filetype', dest="filetype", type=str, default=None, required=False)
    args = parser.parse_args()

    io_options = IOOptions.do_nothing()
    if args.filetype == "concrete_dataset":
        og = ProjectPTS()
        registered_reader[".pts"] = create_csv_reader(format=og, row_skip=1)
        io_options.filetype = ".ply"
        io_options.binary = True
        io_options.intensity = "normalized"
        io_options.shift_system = [3420000, 5370000, 0]
        args.filetype = ".ply"

    if not args.src.is_dir():
        stack = [args.src]
        recursive = False
    else:
        stack = scandir_supported(args.src)
        logger.info(f"Recursive option was chosen and {len(stack)}, files were identified.")
        recursive = True

    # process starts
    for src in stack:
        if not args.dst:
            args.dst = (args.src / args.src.stem).with_suffix(args.filetype)

        if recursive:
            rel = src.relative_to(args.src)
            rel = rel.stem / rel.with_suffix(args.filetype)
            dst = args.dst / rel
            dst.mkdir(parents=True, exist_ok=True)
        else:
            dst = args.dst

        logger.info(f"Reading: {src}")
        data = read(src)
        logger.info(f"Writing to: {dst}")
        write(dst, data, options=io_options)
