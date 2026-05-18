from astropy.time import Time
import astropy.units as units
import hdf5plugin
import h5py as h5
import numpy as np


def read_dims_from_file(filename):

    with h5.File(filename, 'r') as f:
        F, P, T = f['vis'].shape

    return F, P, T


def load_timeseries_from_file(f, p, filename):

    with h5.File(filename, 'r') as hdl:
        vis = hdl['vis'][f, p, :][...]
        w = hdl['vis_weight'][f, p, :][...]
        seq_start = hdl['fpga_start_tick'][...]
        seq_len = hdl['frame_length_fpga_ticks'][...]
        seq_good = hdl['valid_fpga_count'][f, :][...]
        seq_rfi = hdl['rfi_only_fpga_count'][f, :][...]
        seq_pl = hdl['pl_fpga_count'][f, :][...]
        bin_start_ERA_deg = hdl['bin_start_ERA_deg'][...]
        bin_t_inst_ns = hdl['bin_t_inst_ns'][...]
        bin_ut1_ns = hdl['bin_ut1_ns'][...]
        bin_delta_ut1_inst = hdl['bin_delta_ut1_inst'][...]
        bin_ERA_deg = hdl['bin_ERA_deg'][...]

    timeseries = dict(vis=vis, w=w, seq_start=seq_start, seq_len=seq_len,
                      seq_good=seq_good, seq_rfi=seq_rfi, seq_pl=seq_pl,
                      bin_start_ERA_deg=bin_start_ERA_deg,
                      bin_t_inst_ns=bin_t_inst_ns, bin_ut1_ns=bin_ut1_ns,
                      bin_delta_ut1_inst=bin_delta_ut1_inst,
                      bin_ERA_deg=bin_ERA_deg)

    return timeseries


def find_freq_MHz(f, filename):

    with h5.File(filename, 'r') as hdl:
        fmap = hdl['index_map']['freq'][...]

    if f >= len(fmap):
        return -1.0

    return fmap[f]['centre']


def find_prod(i, j, filename):

    with h5.File(filename, 'r') as f:
        pmap = f['index_map']['prod'][...]

    for p in range(len(pmap)):
        if pmap[p][0] == i and pmap[p][1] == j:
            return p

    return -1


def load_frame0_from_file(file):

    frame0_keys = ['frame0_unix_ns', 'frame0_t_inst_ns']

    with h5.File(file, "r") as f:
        for key in frame0_keys:
            if key in f.attrs:
                return f.attrs[key]


def load_dseq_from_file(file):

    dseq_keys = ['fpga_seq_length_ns', 'fpga_seq_length_nsec']

    with h5.File(file, "r") as f:
        for key in dseq_keys:
            if key in f.attrs:
                return f.attrs[key]


def load_timeseries_from_files(f, i, j, files):

    Tf = 0

    for file in files:
        F, P, T = read_dims_from_file(file)
        Tf += T

    vis = np.empty((Tf,), dtype=np.complex64)
    w = np.empty((Tf,), dtype=np.float32)
    seq_start = np.empty((Tf,), dtype=int)
    seq_len = np.empty((Tf,), dtype=int)
    seq_good = np.empty((Tf,), dtype=int)
    seq_rfi = np.empty((Tf,), dtype=int)
    seq_pl = np.empty((Tf,), dtype=int)
    era_bin_start = np.empty((Tf,), dtype=float)
    t_inst_bin = np.empty((Tf,), dtype=int)
    ut1_bin = np.empty((Tf,), dtype=int)
    dut1_bin = np.empty((Tf,), dtype=float)
    era_bin = np.empty((Tf,), dtype=float)

    idx = 0

    for file in files:
        _, _, T = read_dims_from_file(file)
        p = find_prod(i, j, file)

        ts = load_timeseries_from_file(f, p, file)
        vis[idx:idx+T] = ts['vis']
        w[idx:idx+T] = ts['w']
        seq_start[idx:idx+T] = ts['seq_start']
        seq_len[idx:idx+T] = ts['seq_len']
        seq_good[idx:idx+T] = ts['seq_good']
        seq_rfi[idx:idx+T] = ts['seq_rfi']
        seq_pl[idx:idx+T] = ts['seq_pl']
        era_bin_start[idx:idx+T] = ts['bin_start_ERA_deg']
        t_inst_bin[idx:idx+T] = ts['bin_t_inst_ns']
        ut1_bin[idx:idx+T] = ts['bin_ut1_ns']
        dut1_bin[idx:idx+T] = ts['bin_delta_ut1_inst']
        era_bin[idx:idx+T] = ts['bin_ERA_deg']
        idx += T

    return dict(vis=vis, w=w, seq_start=seq_start, seq_len=seq_len,
                seq_good=seq_good, seq_rfi=seq_rfi, seq_pl=seq_pl,
                era_bin_start=era_bin_start, t_inst_bin=t_inst_bin,
                ut1_bin=ut1_bin, dut1_bin=dut1_bin, era_bin=era_bin)
