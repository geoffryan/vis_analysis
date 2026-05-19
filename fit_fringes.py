from pathlib import Path
import sys
import vis_util


def f_vis(era, amp, era_cen, sigma, rate_era, era0):

    envelope = np.exp(-0.5*((era - era_cen) / sigma)**2)
    fringe = np.exp(1.0j * rate_era * (era - era0))

    return amp * envelope * fringe


def f_chi2(params, x_dat, y_dat, var_dat, func):

    y_model = func(x_dat, *params)

    chi2 = (np.abs(y_dat - y_model)**2 / var_dat).sum()

    return chi2


if __name__ == "__main__":

    files = [Path(s) for s in sys.argv[1:]]

    t = vis_util.load_times_from_files(files)
    exist = t.seq_start > 0

    era_min = 200
    era_max = 210

    nrots = np.unique(t.bin_nrot[exist])

    rot_masks = []
    for nrot in nrots:
        mask = exist.copy()
        mask &= (t.bin_era_deg > era_min) & (t.bin_era_deg < era_max)
        rot_masks.append(mask)

    for i, j in [(0, 0), (0, 2), (1, 3), (0, 4), (1, 5), (2, 4), (3, 5)]:
        name_i = vis_util.load_feed_name_from_file(i, files[0])
        name_j = vis_util.load_feed_name_from_file(j, files[0])

        for f in range(4200, 4201):

            f_MHz = vis_util.find_freq_MHz(f, files[0])

            print(f, f_MHz, name_i, name_j)
            
            data = load_timeseries_from_files(f, i, j, files)
            good = (data['seq_good'] > 0)

            for nrot, rot_mask in zip(nrots, rot_masks):

                mask = rot_mask & good

                if not mask.any():
                    continue

                vis = data['vis'][mask]
                w = data['vis'][mask]
                vis_var = 1.0 / w


