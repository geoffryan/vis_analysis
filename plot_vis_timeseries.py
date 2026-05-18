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

            # for f in np.arange(0, 8192):
            for f in np.arange(4200, 4300):

                f_MHz = vis_util.find_freq_MHz(f, filenames[0])

                if f_MHz < 0:
                    break

                data = vis_util.load_timeseries_from_files(f, i, j, filenames)

                good = data['seq_good'] > 0

                if not good.any():
                    continue

                exist = data['seq_start'] > 0

                if output_times:
                    frame0_ns = vis_util.load_frame0_from_file(filenames[0])
                    dseq_ns = vis_util.load_dseq_from_file(filenames[0])

                    seq = data['seq_start'][exist]

                    dt_ns = seq * dseq_ns

                    t0 = Time(frame0_ns * units.ns, format='unix')
                    dt = TimeDelta(dt_ns * units.ns, scale='tai')
                    t = t0 + dt

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

                good_w = data['w'] > 0
                err = np.zeros_like(data['w'])
                err[good_w] = 1.0 / np.sqrt(data['w'][good_w])

                fig, ax = plt.subplots(4, 1, figsize=(8, 8))

                ax[0].plot(data['seq_start'][good], data['vis'][good].real)
                ax[0].plot(data['seq_start'][good], data['vis'][good].imag)
                ax[0].errorbar(data['seq_start'][good],
                               np.abs(data['vis'][good]), err[good],
                               color='k', marker='', ls='-')

                ax[1].plot(data['seq_start'][good],
                           np.angle(data['vis'][good]))

                ax[2].plot(data['seq_start'][exist], data['seq_good'][exist],
                           '.', label='good')
                ax[2].plot(data['seq_start'][exist], data['seq_rfi'][exist],
                           '.', label='rfi')
                ax[2].plot(data['seq_start'][exist], data['seq_pl'][exist],
                           '.', label='pl')
                ax[2].plot(data['seq_start'][exist], data['seq_len'][exist],
                           'k.', label='len')
                ax[2].legend()
                ax[2].set_yscale('symlog', linthresh=100)

                ax[3].plot(data['seq_start'][exist],
                           data['seq_good'][exist] / data['seq_len'][exist],
                           '.', label='good')
                ax[3].plot(data['seq_start'][exist],
                           data['seq_rfi'][exist] / data['seq_len'][exist],
                           '.', label='rfi')
                ax[3].plot(data['seq_start'][exist],
                           data['seq_pl'][exist] / data['seq_len'][exist], '.',
                           label='pl')
                ax[3].set_yscale('symlog', linthresh=1.0e-4)

                min_seq = (data['seq_start'][exist][0]
                           - data['seq_len'][exist][0])
                max_seq = (data['seq_start'][exist][-1]
                           - data['seq_len'][exist][-1])

                ax[0].set(xlim=(min_seq, max_seq))
                ax[1].set(xlim=(min_seq, max_seq))
                ax[2].set(xlim=(min_seq, max_seq))
                ax[3].set(xlim=(min_seq, max_seq))

                fig.suptitle("f[{0:04d}] = {1:f} MHz".format(f, f_MHz))

                figname = "timeseries_f_{0:04d}_{1:d}-{2:d}.png".format(
                        f, i, j)
                print("Saving", figname)
                fig.savefig(figname)
                plt.close(fig)
