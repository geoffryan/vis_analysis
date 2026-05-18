from pathlib import Path
import sys
from astropy.time import Time, TimeDelta
import astropy.units as units
import numpy as np
import matplotlib.pyplot as plt
import vis_util

if __name__ == "__main__":

    filenames = [Path(s) for s in sys.argv[1:]]

    output_times = True

    for i in range(6):
        for j in range(i, 6):

            name_i = vis_util.load_feed_name_from_file(i, filenames[0])
            name_j = vis_util.load_feed_name_from_file(j, filenames[0])

            # for f in np.arange(0, 8192):
            for f in np.arange(4200, 4300):

                f_MHz = vis_util.find_freq_MHz(f, filenames[0])

                if f_MHz < 0:
                    break

                data = vis_util.load_timeseries_from_files(f, i, j, filenames)

                good = data['seq_good'] > 0

                if not good.any():
                    continue

                exist = data['time'].seq_start > 0

                if output_times:
                    t = data['time'].t_start[exist]

                    era = t.earth_rotation_angle('tio').to_value('deg')
                    dera = (era[1:] - era[:-1]).mean()
                    n_bin_per_rot = int(round(360.0 / dera))
                    dera_exp = 360 / n_bin_per_rot

                    era_idx_start = int(round(era[0] / dera_exp))
                    era_idx = era_idx_start + np.arange(len(era))

                    era_res = era - dera_exp * era_idx

                    fig, ax = plt.subplots(1, 1)
                    ax.plot(era_idx, era_res, '.')
                    ax.set(xlabel='ERA bin',
                           ylabel=r'ERA start - ERA bin start (deg)')
                    fig.tight_layout()
                    figname = 'era_res.png'
                    print("Saving", figname)
                    fig.savefig(figname)
                    plt.close(fig)

                    output_times = False

                i0 = data['time'].t_inst_ns_start[exist].argmin()
                t0 = data['time'].t_start[exist][i0]
                t0.precision = 3
                t0_ns = data['time'].t_inst_ns_start[exist][i0]
                dt_h = (data['time'].bin_t_inst_ns - t0_ns) * 1.0e-9 / 3600

                good_w = data['w'] > 0
                err = np.zeros_like(data['w'])
                err[good_w] = 1.0 / np.sqrt(data['w'][good_w])

                fig, ax = plt.subplots(4, 1, figsize=(8, 8))

                ax[0].plot(dt_h[good], data['vis'][good].real)
                ax[0].plot(dt_h[good], data['vis'][good].imag)
                ax[0].errorbar(dt_h[good],
                               np.abs(data['vis'][good]), err[good],
                               color='k', marker='', ls='-')

                ax[1].plot(dt_h[good],
                           np.angle(data['vis'][good]))

                ax[2].plot(dt_h[exist], data['seq_good'][exist],
                           '.', label='good')
                ax[2].plot(dt_h[exist], data['seq_rfi'][exist],
                           '.', label='rfi')
                ax[2].plot(dt_h[exist], data['seq_pl'][exist],
                           '.', label='pl')
                ax[2].plot(dt_h[exist], data['time'].seq_len[exist],
                           'k.', label='len')
                ax[2].legend()
                ax[2].set_yscale('symlog', linthresh=100)

                ax[3].plot(dt_h[exist],
                           data['seq_good'][exist] / data['time'].seq_len[exist],
                           '.', label='good')
                ax[3].plot(dt_h[exist],
                           data['seq_rfi'][exist] / data['time'].seq_len[exist],
                           '.', label='rfi')
                ax[3].plot(dt_h[exist],
                           data['seq_pl'][exist] / data['time'].seq_len[exist], '.',
                           label='pl')
                ax[3].set_yscale('symlog', linthresh=1.0e-4)

                ax[0].set(xlim=(dt_h[exist].min(), dt_h[exist].max()))
                ax[1].set(xlim=(dt_h[exist].min(), dt_h[exist].max()))
                ax[2].set(xlim=(dt_h[exist].min(), dt_h[exist].max()))
                ax[3].set(xlim=(dt_h[exist].min(), dt_h[exist].max()),
                          xlabel=r'Hours since start at {:s}'.format(t0.utc.isot))

                fig.suptitle("Uncalibrated Visibility f[{0:04d}] = {1:f} MHz {2:s}-{3:s}$^*$"
                             .format(f, f_MHz, name_i, name_j))

                figname = "timeseries_f_{0:04d}_{1:d}-{2:d}.png".format(
                        f, i, j)
                print("Saving", figname)
                fig.savefig(figname)
                plt.close(fig)
