from pathlib import Path
import sys
from astropy.time import Time, TimeDelta
import astropy.units as units
import numpy as np
import matplotlib.pyplot as plt
import vis_util


files = [Path(f) for f in sys.argv[1:]]

nbin_per_rot = 8640

t = vis_util.load_times_from_files(files)

exist = t.seq_start > 0

era_start = t.t_start[exist].earth_rotation_angle('tio').to_value('deg')
era_end = t.t_end[exist].earth_rotation_angle('tio').to_value('deg')

i0 = np.argmin(t.t_inst_ns_start[exist])

t0_inst = t.t_inst_ns_start[exist][i0]
dt_start = t.t_inst_ns_start[exist] - t0_inst

dt_start_h = dt_start * 1.0e-9 / 3600

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

ax.plot(era_start, dt_start_h, '.')
ax.plot(era_end, dt_start_h, '.')

ax.set(xlabel='ERA (deg)', ylabel='Hours since start at {:s}'.format(t.t_start[exist][i0].isot))
fig.tight_layout()

figname = 'era_bins.png'
print("Saving", figname)
fig.savefig(figname, dpi=200)

bin_idx = np.floor(t.bin_era_deg[exist] * nbin_per_rot / 360.0).astype(int)

era_min = 200
era_max = 201

for b in range(nbin_per_rot):
    mask = (bin_idx == b)
    if ((era_start[mask] > era_min) & (era_start[mask] < era_max)).any():
        ax.plot(era_start[mask], dt_start_h[mask], color='k', ls='-', marker=None, lw=1)
    if ((era_end[mask] > era_min) & (era_end[mask] < era_max)).any():
        ax.plot(era_end[mask], dt_start_h[mask], color='k', ls='-', marker=None, lw=1)

ax.set(xlim=(era_min, era_max))
fig.tight_layout()

figname = 'era_bins_zoom.png'
print("Saving", figname)
fig.savefig(figname, dpi=200)


