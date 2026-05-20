from astropy.time import Time
import astropy.units as units
import hdf5plugin
import h5py as h5
import numpy as np
import time_utils


class TimeInfo:

    def __init__(self, N):
        # Quantities we'll read from the file
        self.frame0_unix_ns = np.empty((N,), dtype=int)
        self.dseq_ns = np.empty((N,), dtype=int)
        self.seq_start = np.empty((N,), dtype=int)
        self.seq_len = np.empty((N,), dtype=int)
        self.bin_idx = np.empty((N,), dtype=int)
        self.bin_start_era_deg = np.empty((N,), dtype=float)
        self.bin_end_era_deg = np.empty((N,), dtype=float)
        self.bin_start_eral_deg = np.empty((N,), dtype=float)
        self.bin_end_eral_deg = np.empty((N,), dtype=float)
        self.bin_t_inst_ns = np.empty((N,), dtype=int)
        self.bin_ut1_ns = np.empty((N,), dtype=int)
        self.bin_delta_ut1_inst = np.empty((N,), dtype=float)
        self.bin_era_deg = np.empty((N,), dtype=float)
        self.bin_xp_as = np.empty((N,), dtype=float)
        self.bin_yp_as = np.empty((N,), dtype=float)

        # Quantities we'll calculate later
        self.seq_end = np.empty((N,), dtype=int)
        self.t_inst_ns_start = np.empty((N,), dtype=int)
        self.bin_nrot = np.empty((N,), dtype=int)

    def fill_from_file(self, f_handle, idx=0):

        T = f_handle['fpga_start_tick'].shape[0]
        self.frame0_unix_ns[idx:idx+T] = load_frame0_from_attrs(f_handle.attrs) 
        self.dseq_ns[idx:idx+T] = load_dseq_from_attrs(f_handle.attrs) 
        self.seq_start[idx:idx+T] = f_handle['fpga_start_tick'][...]
        self.seq_len[idx:idx+T] = f_handle['frame_length_fpga_ticks'][...]
        self.bin_idx[idx:idx+T] = (f_handle['abs_time_idx'][...] if 'abs_time_idx' in f_handle
                                   else -1)
        self.bin_start_era_deg[idx:idx+T] = f_handle['bin_start_ERA_deg'][...]
        self.bin_end_era_deg[idx:idx+T] = f_handle['bin_end_ERA_deg'][...]
        self.bin_start_eral_deg[idx:idx+T] = f_handle['bin_start_ERAL'][...]
        self.bin_end_eral_deg[idx:idx+T] = f_handle['bin_end_ERAL'][...]
        self.bin_t_inst_ns[idx:idx+T] = f_handle['bin_t_inst_ns'][...]
        self.bin_ut1_ns[idx:idx+T] = f_handle['bin_ut1_ns'][...]
        self.bin_delta_ut1_inst[idx:idx+T] = f_handle['bin_delta_ut1_inst'][...]
        self.bin_era_deg[idx:idx+T] = f_handle['bin_ERA_deg'][...]
        self.bin_xp_as[idx:idx+T] = f_handle['bin_xp_as'][...]
        self.bin_yp_as[idx:idx+T] = f_handle['bin_yp_as'][...]

    def finalize(self):

        self.seq_end = self.seq_start + self.seq_len
        self.t_inst_ns_start = time_utils.get_t_inst_ns(self.seq_start,
                                                        self.frame0_unix_ns, self.dseq_ns)
        self.t_inst_ns_end = time_utils.get_t_inst_ns(self.seq_end,
                                                      self.frame0_unix_ns, self.dseq_ns)
        self.t_start = time_utils.calc_astropy_time_from_inst_ns(self.t_inst_ns_start,
                                                                 self.frame0_unix_ns)
        self.t_end = time_utils.calc_astropy_time_from_inst_ns(self.t_inst_ns_end,
                                                               self.frame0_unix_ns)
        self.bin_t = time_utils.calc_astropy_time_from_inst_ns(self.bin_t_inst_ns,
                                                               self.frame0_unix_ns)
        self.bin_nrot = time_utils.get_nrot_at_t(self.bin_t)



def read_dims_from_file(filename):

    with h5.File(filename, 'r') as f:
        F, P, T = f['vis'].shape

    return F, P, T


def load_timeseries_from_file(f, p, filename, ti=None, idx=None):

    ts = {}

    with h5.File(filename, 'r') as hdl:
        ts['vis'] = hdl['vis'][f, p, :][...]
        ts['w'] = hdl['vis_weight'][f, p, :][...]
        ts['seq_good'] = hdl['valid_fpga_count'][f, :][...]
        ts['seq_rfi'] = hdl['rfi_only_fpga_count'][f, :][...]
        ts['seq_pl'] = hdl['pl_fpga_count'][f, :][...]
        if ti is None:
            T = ts['vis'].shape[0]
            ts['time'] = TimeInfo(T)
            ts['time'].fill_from_file(hdl)
        else:
            if idx is None:
                raise ValueError("Cannot load into a TimeInfo without an idx")
            ti.fill_from_file(hdl, idx)

    return ts


def find_freq_MHz(f, filename):

    with h5.File(filename, 'r') as hdl:
        fmap = hdl['index_map']['freq'][...]

    if f >= len(fmap):
        raise RuntimeError("freq array index {:d} bigger than freq array len {:d}".format(f, len(fmap)))

    return fmap[f]['centre']


def find_prod(i, j, filename):

    with h5.File(filename, 'r') as f:
        pmap = f['index_map']['prod'][...]

    for p in range(len(pmap)):
        if pmap[p][0] == i and pmap[p][1] == j:
            return p

    raise RuntimeError("Product ({:d}, {:d}) not in file: {:s}".format(i, j, filename))


def load_value_from_attrs(keys, attrs):

    if isinstance(keys, str):
        keys = [keys]
    
    for key in keys:
        if key in attrs:
            return attrs[key]

    raise RuntimeError("attrs did not contain any of: {:s}".format(" ".join(*keys)))


def load_frame0_from_attrs(attrs):

    frame0_keys = ['frame0_unix_ns', 'frame0_t_inst_ns']

    return load_value_from_attrs(frame0_keys, attrs)


def load_dseq_from_attrs(attrs):

    dseq_keys = ['fpga_seq_length_ns', 'fpga_seq_length_nsec']
    
    return load_value_from_attrs(dseq_keys, attrs)


def load_frame0_from_file(file):

    with h5.File(file, "r") as f:
        return load_frame0_from_attrs(f.attrs)


def load_dseq_from_file(file):

    dseq_keys = ['fpga_seq_length_ns', 'fpga_seq_length_nsec']

    with h5.File(file, "r") as f:
        return load_dseq_from_attrs(f.attrs)


def load_feed_name_from_file_handle(i, hndl):

    if 'label' in hndl['index_map']:
        return hndl['index_map/label'][i]
    else:
        return str(i)


def load_feed_name_from_file(i, file):

    with h5.File(file, "r") as f:
        return load_feed_name_from_file_handle(i, f)


def load_feed_pos_from_file_handle(i, hndl):

    if 'dish_positions_in_grid_coords' in hndl['index_map']:
        return hndl['index_map/dish_positions_in_grid_coords'][i]

    raise RuntimeError("No dish positions in file", hndl)


def load_feed_pos_from_file(i, file):

    with h5.File(file, "r") as f:
        return load_feed_pos_from_file_handle(i, f)


def load_timeseries_from_files(f, i, j, files):

    Tf = 0

    for file in files:
        F, P, T = read_dims_from_file(file)
        Tf += T

    vis = np.empty((Tf,), dtype=np.complex64)
    w = np.empty((Tf,), dtype=np.float32)
    seq_good = np.empty((Tf,), dtype=int)
    seq_rfi = np.empty((Tf,), dtype=int)
    seq_pl = np.empty((Tf,), dtype=int)

    ti = TimeInfo(Tf)

    idx = 0

    for file in files:
        _, _, T = read_dims_from_file(file)
        p = find_prod(i, j, file)

        ts = load_timeseries_from_file(f, p, file, ti, idx)
        vis[idx:idx+T] = ts['vis']
        w[idx:idx+T] = ts['w']
        seq_good[idx:idx+T] = ts['seq_good']
        seq_rfi[idx:idx+T] = ts['seq_rfi']
        seq_pl[idx:idx+T] = ts['seq_pl'] 
        idx += T

    ti.finalize()

    return dict(vis=vis, w=w, time=ti,
                seq_good=seq_good, seq_rfi=seq_rfi, seq_pl=seq_pl)


def load_times_from_files(files):

    Tf = 0

    print("Determining sizes")

    for file in files:
        _, _, T = read_dims_from_file(file)
        Tf += T
    
    ti = TimeInfo(Tf)

    idx = 0

    for file in files:
        print("Loading", file)
        _, _, T = read_dims_from_file(file)

        with h5.File(file, 'r') as f:
            ti.fill_from_file(f, idx)

        idx += T
    
    ti.finalize()

    return ti 
