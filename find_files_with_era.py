from pathlib import Path
import sys
import h5py as h5
import vis_util


if __name__ == "__main__":

    era_min = float(sys.argv[1])
    era_max = float(sys.argv[2])
    files = [Path(s) for s in sys.argv[3:]]

    good_files = []

    for file in files:
        _, _, T = vis_util.read_dims_from_file(file)
        t = vis_util.TimeInfo(T)
        with h5.File(file, 'r') as f:
            t.fill_from_file(f)
        exist = t.seq_start > 0

        if ((t.bin_era_deg[exist] >= era_min)
                & (t.bin_era_deg[exist] <=era_max)).any():
            good_files.append(file)

    print(*good_files)
