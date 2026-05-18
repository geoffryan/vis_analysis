from pathlib import Path
import sys
from astropy.time import Time
import astropy.units as units
import hdf5plugin
import h5py as h5
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import vis_util


def make_waterfall(i, j, files):

    Tf = 0

    for file in files:
        F, P, T = vis_util.read_dims_from_file(file)
        Tf += T

    fds_fac = 1
    frame0_ns = -1

    with h5.File(files[0], "r") as f:
        fmap = np.array([freq[0] for freq in f['index_map/freq'][::fds_fac]]
                        ).astype(float)
        frame0_ns = f.attrs['frame0_t_inst_ns']
        if 'label' in f['index_map']:
            name_i = f['index_map/label'][i]
            name_j = f['index_map/label'][j]
        dseq_ns = f.attrs['fpga_seq_length_nsec']

        name_i = vis_util.load_feed_name_from_file_handle(i, f)
        name_j = vis_util.load_feed_name_from_file_handle(j, f)

    Fds = len(fmap)

    vis_r = np.empty((Fds, Tf), dtype=np.float32)
    vis_i = np.empty((Fds, Tf), dtype=np.float32)
    vis_m = np.empty((Fds, Tf), dtype=np.float32)
    vis_p = np.empty((Fds, Tf), dtype=np.float32)
    seq_e = np.empty((Tf+1,), dtype=int)
    seq_len = np.empty((Tf,), dtype=int)

    df = np.diff(fmap).mean()

    fe = np.linspace(fmap[0]-0.5*df, fmap[-1]+0.5*df, len(fmap)+1)

    idx = 0

    for file in files:
        p = vis_util.find_prod(i, j, file)
        with h5.File(file, "r") as f:
            T = f['vis'].shape[2]
            vis_chunk = f['vis'][::fds_fac, p, :][...]
            vis_r[:, idx:idx+T] = vis_chunk.real
            vis_i[:, idx:idx+T] = vis_chunk.imag
            vis_m[:, idx:idx+T] = np.abs(vis_chunk)
            vis_p[:, idx:idx+T] = np.angle(vis_chunk)
            seq_e[idx:idx+T] = f['fpga_start_tick'][...]
            seq_len[idx:idx+T] = f['frame_length_fpga_ticks'][...]
            idx += T

    seq_e[-1] = seq_e[-2] + seq_len[-1]
    ns_e = seq_e * dseq_ns
    dt_e = (ns_e - ns_e[seq_e > 0].min()) / (1.0e9 * 3600)

    t0_str = (Time(frame0_ns * units.ns, format='unix', precision=0).isot
              if frame0_ns >= 0 else "start")

    dt_min = dt_e[seq_e > 0].min()
    dt_max = dt_e[seq_e > 0].max()

    rNorm = mpl.colors.SymLogNorm(linthresh=0.1, vmin=-100, vmax=100)
    iNorm = mpl.colors.SymLogNorm(linthresh=0.1, vmin=-100, vmax=100)

    size = (9, 6)
    pix_height = 6000

    fig, ax = plt.subplots(2, 2, figsize=size)

    c = ax[0, 0].pcolormesh(dt_e, fe, vis_r, cmap='RdBu', norm=rNorm)
    cb = fig.colorbar(c)
    cb.set_label('Vis Real')
    ax[0, 0].set(xlim=(dt_min, dt_max), ylim=(fe.min(), fe.max()))

    c = ax[0, 1].pcolormesh(dt_e, fe, vis_i, cmap='BrBG', norm=iNorm)
    cb = fig.colorbar(c)
    cb.set_label('Vis Imag')
    ax[0, 1].set(xlim=(dt_min, dt_max), ylim=(fe.min(), fe.max()))

    c = ax[1, 0].pcolormesh(dt_e, fe, vis_m, norm='log')
    cb = fig.colorbar(c)
    cb.set_label('Vis Mag')
    ax[1, 0].set(xlim=(dt_min, dt_max), ylim=(fe.min(), fe.max()))

    c = ax[1, 1].pcolormesh(dt_e, fe, vis_p, norm='linear', cmap='twilight',
                            vmin=-np.pi, vmax=np.pi)
    cb = fig.colorbar(c)
    cb.set_label('Vis Phase')
    ax[1, 1].set(xlim=(dt_min, dt_max), ylim=(fe.min(), fe.max()))

    ax[0, 0].set(ylabel="Frequency (MHz)")
    ax[1, 0].set(xlabel="Hours since {:s}".format(t0_str),
                 ylabel="Frequency (MHz)")
    ax[1, 1].set(xlabel="Hours since {:s}".format(t0_str))

    print('vis_r', vis_r.min(), vis_r.max())
    print('vis_i', vis_i.min(), vis_i.max())
    print('vis_m', vis_m[vis_m > 0].min(), vis_m.max())

    fig.suptitle(r"Uncalibrated Visibility {:s} - {:s}$^*$"
                 .format(name_i, name_j))
    fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)

    figname = "waterfall_{0:d}-{1:d}.png".format(i, j)
    print("Saving", figname)
    fig.savefig(figname, dpi=pix_height/size[1])
    plt.close()


if __name__ == "__main__":

    filenames = [Path(s) for s in sys.argv[1:]]

    for i in range(6):
        for j in range(i, 6):
            make_waterfall(i, j, filenames)
