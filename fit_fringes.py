from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sp_opt
import vis_util


era_min = 201
era_max = 205


def f_vis(era, amp, env_era_cen, sigma, fringe_era_cen, rate_era, phi0):

    env_arg = (era - env_era_cen) / sigma
    envelope = np.exp(-0.5*(env_arg)**2)
    # envelope = np.cos(np.pi * env_arg)**2
    fringe_arg = np.pi * (era - fringe_era_cen) / 180
    fringe = np.exp(1.0j * (rate_era * np.sin(fringe_arg) + phi0))

    return amp * envelope * fringe


def df_vis(era, amp, env_era_cen, sigma, fringe_era_cen, rate_era, phi0):

    env_arg = (era - env_era_cen) / sigma
    envelope = np.exp(-0.5*(env_arg)**2)
    # envelope = np.cos(np.pi * env_arg)**2
    fringe_arg = np.pi * (era - fringe_era_cen) / 180
    fringe = np.exp(1.0j * (rate_era * np.sin(fringe_arg) + phi0))

    denvarg_deeracen = -1.0 / sigma
    denvarg_dsigma = -env_arg / sigma
    denv_deeracen = denvarg_deeracen * (-env_arg) * envelope
    denv_dsigma = denvarg_dsigma * (-env_arg) * envelope
    # denv_dderacen = np.pi * denvarg_dderacen * (-2 * np.cos(np.pi*env_arg) * np.sin(np.pi*env_arg))
    # denv_dsigma = np.pi * denvarg_dsigma * (-2 * np.cos(np.pi*env_arg) * np.sin(np.pi*env_arg))
    dfriarg_dferacen = -np.pi/180
    dfri_dferacen = 1.0j * rate_era * np.cos(fringe_arg) * dfriarg_dferacen * fringe
    dfri_drate = 1.0j * np.sin(fringe_arg) * fringe
    dfri_dphi0 = 1.0j * fringe

    return np.array([envelope * fringe, amp * denv_deeracen * fringe, amp * denv_dsigma * fringe,
                     amp * envelope * dfri_dferacen, amp * envelope * dfri_drate, amp * envelope * dfri_dphi0])


def f_chi2(params, x_dat, y_dat, var_dat, func, dfunc):

    y_model = func(x_dat, *params)

    chi2_arr = np.abs(y_dat - y_model)**2 / var_dat
    chi2 = chi2_arr.sum()

    return chi2


def df_chi2(params, x_dat, y_dat, var_dat, func, dfunc):

    # |yd - y|**2 = (yd* - y*) x (yd - y)
    # d |yd-y|**2 / dx = -dy/dx* x (yd - y) - (yd* - y*) x dy/dx
    #                  = dy/dx* x y + y* x dy/dx - dy/dx* yd - yd* dy/dx
    #                  = -2 Re((yd* - y*) dy/dx)

    y_model = func(x_dat, *params)
    dy_model = dfunc(x_dat, *params)

    dchi2_arr = -2 * ((y_dat - y_model)[None, :].conj() * dy_model).real / var_dat
    dchi2 = dchi2_arr.sum(axis=1)

    return dchi2


def check_grad(x0, args, func, dfunc):

    f = func(x0, *args)
    df_ex = dfunc(x0, *args)

    h = 1.0e-6 * f / df_ex

    for i in range(len(x0)):
        xb = x0.copy()
        xa = x0.copy()

        hi = h[i] if df_ex[i] != 0.0 else 1.0e-6
        xa[i] -= hi
        xb[i] += hi

        fa = func(xa, *args)
        fb = func(xb, *args)

        df_num = (fb - fa) / (2*hi)

        print(i, df_ex[i], df_num, (df_ex[i] - df_num) / df_num)



if __name__ == "__main__":

    files = [Path(s) for s in sys.argv[1:]]

    t = vis_util.load_times_from_files(files)
    exist = t.seq_start > 0

    nrots = np.unique(t.bin_nrot[exist])

    rot_masks = []
    for nrot in nrots:
        mask = exist.copy()
        mask &= (t.bin_era_deg > era_min) & (t.bin_era_deg < era_max) & (t.bin_nrot == nrot)
        rot_masks.append(mask)

    for i, j in [(0, 0), (0, 2), (1, 3), (0, 4), (1, 5), (2, 4), (3, 5)]:
        name_i = vis_util.load_feed_name_from_file(i, files[0])
        name_j = vis_util.load_feed_name_from_file(j, files[0])

        pos_i = vis_util.load_feed_pos_from_file(i, files[0])
        pos_j = vis_util.load_feed_pos_from_file(j, files[0])

        ew_sep = pos_i[0] - pos_j[0]

        for f in range(4200, 4201):

            f_MHz = vis_util.find_freq_MHz(f, files[0])
            lam_m = 2.99792458e8 / (1.0e-6 * f_MHz)
            u_ij = ew_sep / lam_m

            rate0 = 2*np.pi * u_ij * (180/np.pi)**8  # Huh?

            print(f, f_MHz, name_i, name_j)
            
            data = vis_util.load_timeseries_from_files(f, i, j, files)
            good = (data['seq_good'] > 0)

            N = len(nrots)

            fig, ax = plt.subplots(N, 1, figsize=(4, 12))

            for rot, (nrot, rot_mask) in enumerate(zip(nrots, rot_masks)):

                mask = rot_mask & good

                if not mask.any():
                    print("skipping rot", rot)
                    continue

                era = t.bin_era_deg[mask]
                vis = data['vis'][mask]
                w = data['w'][mask]
                vis_var = 1.0 / w

                amp = (vis.real).max()
                env_era_cen = 203.0
                sigma = 1.0
                fringe_era_cen = 203.0
                rate_era = rate0
                phi0 = 1.0

                par0 = np.array([amp, env_era_cen, sigma, fringe_era_cen, rate_era, phi0])
                args = (era, vis, vis_var, f_vis, df_vis)

                check_grad(par0, args, f_chi2, df_chi2)

                res = sp_opt.minimize(f_chi2, par0, args=args,
                                      jac=df_chi2,
                                      bounds=((0.01, 1.0), (200.0, 206.0), (0.1, 10.0),
                                              (200.0, 206.0), (0.1, 200000), (0.0, 2*np.pi)),
                                      method='L-BFGS-B',
                                      options={'maxiter': 1000})
                print(rot)
                if res.success:
                    print(res.success)
                else:
                    print(res.success, res.message)
                # print(par0)
                # vis_fit = f_vis(era, *par0)
                print(res.x)
                vis_fit = f_vis(era, *res.x)

                ax[rot].errorbar(era, vis.real, np.sqrt(vis_var/2), ls='')
                ax[rot].errorbar(era, vis.imag, np.sqrt(vis_var/2), ls='')

                ax[rot].plot(era, vis_fit.real, lw=0.5, color='k', marker='')
                ax[rot].plot(era, vis_fit.imag, lw=0.5, color='grey', marker='')

                ax[rot].set(xlim=(era_min, era_max), ylim=(-0.75, 0.75),
                            xlabel=r'ERA (deg)', ylabel='Vis')

            fig.tight_layout()

            figname = "fringe_fit_f_{:d}_{:d}-{:d}.png".format(f, i, j)
            print("Saving", figname)
            fig.savefig(figname, dpi=600)
            plt.close(fig)

