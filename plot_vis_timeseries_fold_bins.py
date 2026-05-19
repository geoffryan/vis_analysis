import math
from pathlib import Path
import sys
import astropy.units as units
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import vis_util


files = [Path(f) for f in sys.argv[1:]]

t = vis_util.load_times_from_files(files)
exist = t.seq_start > 0

nrots = np.unique(t.bin_nrot[exist])

row_masks = []
nrot_edges = []
era_edges = []

i0 = t.t_inst_ns_start[exist].argmin()
t0 = t.t_start[i0]
t0.precision = 3
t0_str = t0.isot

for nrot in nrots:
    mask = (t.bin_nrot == nrot)
    mask &= exist

    era_start = t.t_start[mask].earth_rotation_angle('tio').to_value('deg')
    era_end = t.t_end[mask].earth_rotation_angle('tio').to_value('deg')
    era_bin = t.bin_era_deg[mask]
    era_start[era_start > era_bin] -= 360.0
    era_end[era_end < era_bin] += 360.0
    
    dera = (era_end-era_start).mean()

    era_a = era_start.min()
    era_b = era_end.max()

    nera = int(round((era_b - era_a) / dera))

    era_e = np.linspace(era_a, era_b, nera+1)

    for i in range(len(era_bin)):
        idx = int(math.floor(era_bin[i] - era_a) / nera)
        era_e[idx] = era_start[i]
        era_e[idx+1] = era_end[i]

    row_masks.append(mask)
    nrot_edges.append([nrot - nrots.min(), nrot+1 - nrots.min()])
    era_edges.append(era_e)

nrows = len(row_masks)

for i, j in [(0, 0), (0, 2), (1, 3), (0, 4), (1, 5), (2, 4), (3, 5)]:
    name_i = vis_util.load_feed_name_from_file(i, files[0])
    name_j = vis_util.load_feed_name_from_file(j, files[0])

    for f in range(4200, 5000, 10):

        f_MHz = vis_util.find_freq_MHz(f, files[0])

        print(f, f_MHz, name_i, name_j)
        
        title = ("Uncalibrated Visibility f[{0:04d}] = {1:f} MHz {2:s}-{3:s}$^*$"
                 .format(f, f_MHz, name_i, name_j))

        data = vis_util.load_timeseries_from_files(f, i, j, files)

        rNorm = mpl.colors.SymLogNorm(linthresh=0.1, vmin=-100, vmax=100)
        iNorm = mpl.colors.SymLogNorm(linthresh=0.1, vmin=-100, vmax=100)
        mNorm = mpl.colors.LogNorm(vmin=1.0e-6, vmax=100)
        pNorm = mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi)

        vis_r = data['vis'].real
        vis_i = data['vis'].imag
        vis_m = np.abs(data['vis'])

        vis_p = np.zeros(data['vis'].shape, dtype=float)
        good = data['seq_good'] > 0
        vis_p[good] = np.angle(data['vis'][good])

        fig, ax = plt.subplots(2, 2, figsize=(12, 6))

        for mask, nrot_e, era_e in zip(row_masks, nrot_edges, era_edges):
            cr = ax[0, 0].pcolormesh(era_e, nrot_e, vis_r[mask][None, :], norm=rNorm, cmap='RdBu')
            ci = ax[0, 1].pcolormesh(era_e, nrot_e, vis_i[mask][None, :], norm=iNorm, cmap='BrBG')
            cm = ax[1, 0].pcolormesh(era_e, nrot_e, vis_m[mask][None, :], norm=mNorm, cmap='viridis')
            cp = ax[1, 1].pcolormesh(era_e, nrot_e, vis_p[mask][None, :], norm=pNorm, cmap='twilight')

        cbr = fig.colorbar(cr)
        cbi = fig.colorbar(ci)
        cbm = fig.colorbar(cm)
        cbp = fig.colorbar(cp)

        cbr.set_label('Vis Real')
        cbi.set_label('Vis Imag')
        cbm.set_label('Vis Mag')
        cbp.set_label('Vis Phase')

        ax[0, 0].set(ylabel='Rotations since start at\n{:s}'.format(t0_str))
        ax[1, 0].set(xlabel='ERA (deg)', ylabel='Rotations since start at\n{:s}'.format(t0_str))
        ax[1, 1].set(xlabel='ERA (deg)')

        fig.suptitle(title)

        figname = "timeseries_era_stack_f_{0:04d}_{1:d}-{2:d}.png".format(
                f, i, j)
        print("Saving", figname)
        fig.savefig(figname, dpi=600)
        plt.close(fig)

        era_min = 200
        era_max = 208

        ax[0, 0].set(xlim=(era_min, era_max))
        ax[0, 1].set(xlim=(era_min, era_max))
        ax[1, 0].set(xlim=(era_min, era_max))
        ax[1, 1].set(xlim=(era_min, era_max))

        figname = "timeseries_era_stack_zoom_f_{0:04d}_{1:d}-{2:d}.png".format(
                f, i, j)
        print("Saving", figname)
        fig.savefig(figname, dpi=600)
        plt.close(fig)

        era_min = 202
        era_max = 203

        ax[0, 0].set(xlim=(era_min, era_max))
        ax[0, 1].set(xlim=(era_min, era_max))
        ax[1, 0].set(xlim=(era_min, era_max))
        ax[1, 1].set(xlim=(era_min, era_max))

        figname = "timeseries_era_stack_zoom2_f_{0:04d}_{1:d}-{2:d}.png".format(
                f, i, j)
        print("Saving", figname)
        fig.savefig(figname, dpi=600)
        plt.close(fig)



