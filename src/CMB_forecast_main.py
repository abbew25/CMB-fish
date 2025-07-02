import sys
import numpy as np
from configobj import ConfigObj
from fishercomputations import (
    Set_Bait,
    Fish,
)
from setup import CosmoResults, write_fisher
from rich.console import Console

if __name__ == "__main__":
    console = Console()
    # Read in the config file
    configfile = sys.argv[1]
    # print(sys.argv[1])
    pardict = ConfigObj(configfile)

    if "geff_fixed" not in pardict:
        pardict["geff_fixed"] = True
    if "neutrino_mass_fixed" not in pardict:
        pardict["neutrino_mass_fixed"] = True

    if not pardict.as_bool("neutrino_mass_fixed") and not pardict.as_bool("geff_fixed"):
        console.log(
            "You have set neutrino_mass_fixed to False and geff_fixed to False. This is not supported."
        )
        console.log("Please set one of these parameters to True.")
        sys.exit(1)

    # fracstepthetastar = 0.015  # good
    # fracstepomegab = 0.045
    # fracstepomegacdm = 0.045
    # fracstepAs = 0.01  # good
    # fracstepns = 0.001  # good
    # fracsteptau = 0.01  # good
    # fracstepmnu = 0.045  # good

    fracstepthetastar = 0.00002  # good
    fracstepomegab = 0.05
    fracstepomegacdm = 0.05
    fracstepAs = 0.01  # good
    fracstepns = 0.0002  # good
    fracsteptau = 0.001  # good
    fracstepmnu = 0.05  # good

    # Set up the linear power spectrum and derived parameters based on the input cosmology
    cosmo = CosmoResults(
        pardict,
        fracstepthetastar=fracstepthetastar,
        fracstepomegab=fracstepomegab,
        fracstepomegacdm=fracstepomegacdm,
        fracstepAs=fracstepAs,
        fracstepns=fracstepns,
        fracsteptau=fracsteptau,
        fracstepmnu=fracstepmnu,
    )

    console.log("computed CAMB CMB temperature-temperature power spectrum.")

    console.log("Fitting geff?")
    console.log((not pardict.as_bool("geff_fixed")))

    console.log("Fitting neutrino mass?")
    console.log((not pardict.as_bool("neutrino_mass_fixed")))

    # Precompute some things we might need for the Fisher matrix
    derivatives = Set_Bait(
        cosmo,
        geff_fixed=pardict.as_bool("geff_fixed"),
        neutrino_mass_fixed=pardict.as_bool("neutrino_mass_fixed"),
        fracstepthetastar=fracstepthetastar,
        fracstepomegab=fracstepomegab,
        fracstepomegacdm=fracstepomegacdm,
        fracstepAs=fracstepAs,
        fracstepns=fracstepns,
        fracsteptau=fracsteptau,
        fracstepmnu=fracstepmnu,
    )

    derClATT, derClAEE, derClATE, derClABB = derivatives[0]
    derClthetastarTT, derClthetastarEE, derClthetastarTE, derClthetastarBB = (
        derivatives[1]
    )
    derClOmegabTT, derClOmegabEE, derClOmegabTE, derCLOmegabBB = derivatives[2]
    derClOmegacdmTT, derClOmegacdmEE, derClOmegacdmTE, derClOmegamcdmBB = derivatives[3]
    derClAsTT, derClAsEE, derClAsTE, derClAsBB = derivatives[4]
    derClnsTT, derClnsEE, derClnsTE, derClnsBB = derivatives[5]
    derCltauTT, derCltauEE, derCltauTE, derCltauBB = derivatives[6]
    derClgeffTT, derClgeffEE, derClgeffTE, derClgeffBB = None, None, None, None
    derClmnuTT, derClmnuEE, derClmnuTE, derClmnuBB = None, None, None, None
    if not pardict.as_bool("geff_fixed"):
        derClgeffTT, derClgeffEE, derClgeffTE, derClgeffBB = derivatives[7]
    if not pardict.as_bool("neutrino_mass_fixed"):
        derClmnuTT, derClmnuEE, derClmnuTE, derClmnuBB = derivatives[7]

    # import matplotlib.pyplot as plt
    # plt.plot(cosmo.ell, derClA(cosmo.ell))
    # plt.show()
    # exit()

    # import matplotlib.pyplot as plt
    # plt.plot(cosmo.ell, splev(cosmo.ell, cosmo.clTT)*cosmo.ell*(cosmo.ell+1)/(2*np.pi), label=r"$C_\ell^{TT}$")
    # plt.show()
    # exit()

    # import matplotlib.pyplot as plt
    # plt.plot(cosmo.ell, derClOmegab(cosmo.ell), label=r"$\Omega_b$")

    # plt.plot(cosmo.ell, derClOmegacdm(cosmo.ell), label=r"$\Omega_{\mathrm{cdm}}$")

    # plt.plot(cosmo.ell, derClthetastar(cosmo.ell), label=r"$100\theta_*$")
    # plt.show()
    # exit()

    console.log(
        "Computed derivatives of the power spectrum w.r.t. forecast parameters."
    )

    # Loop over redshifts and compute the Fisher matrix and output the 3x3 matrix

    if pardict.as_bool("geff_fixed") and pardict.as_bool("neutrino_mass_fixed"):
        console.log(
            "#  100theta_star  100theta_star_err(%)  A(Neff)  A(Neff)_err(%)  100Omegab  100Omegab_err(%)  Omegacdm  Omegacdm_err(%)  As10  As10_err(%)  ns  ns_err(%)  tau  tau_err(%)"
        )
        # console.log(
        #     "#  100theta_star  100theta_star_err(%)  A(Neff)  A(Neff)_err(%)  Omegacdm  Omegacdm_err(%)  As10  As10_err(%)  ns  ns_err(%)  tau  tau_err(%)"
        # )

    elif not pardict.as_bool("geff_fixed") and pardict.as_bool("neutrino_mass_fixed"):
        console.log(
            "#  100theta_star  100theta_star_err(%)  A(Neff)  A(Neff)_err(%)  100Omegab  100Omegab_err(%)  Omegacdm  Omegacdm_err(%)  As10  As10_err(%)  ns  ns_err(%)  tau  tau_err(%)  log10Geff  geff_err(%)"
        )
        # console.log(
        #     "#  100theta_star  100theta_star_err(%)  A(Neff)  A(Neff)_err(%)  Omegacdm  Omegacdm_err(%)  As10  As10_err(%)  ns  ns_err(%)  tau  tau_err(%)  log10Geff  geff_err(%)"
        # )

    elif pardict.as_bool("geff_fixed") and not pardict.as_bool("neutrino_mass_fixed"):
        console.log(
            "#  100theta_star  100theta_star_err(%)  A(Neff)  A(Neff)_err(%)  100Omegab  100Omegab_err(%)  Omegacdm  Omegacdm_err(%)  As10  As10_err(%)  ns  ns_err(%)  tau  tau_err(%)  mnu  mnu_err(%)"
        )

    Catch = Fish(
        cosmo,
        [derClthetastarTT, derClthetastarEE, derClthetastarTE, derClthetastarBB],
        [derClATT, derClAEE, derClATE, derClABB],
        [derClOmegabTT, derClOmegabEE, derClOmegabTE, derCLOmegabBB],
        [derClOmegacdmTT, derClOmegacdmEE, derClOmegacdmTE, derClOmegamcdmBB],
        [derClAsTT, derClAsEE, derClAsTE, derClAsBB],
        [derClnsTT, derClnsEE, derClnsTE, derClnsBB],
        [derCltauTT, derCltauEE, derCltauTE, derCltauBB],
        [derClgeffTT, derClgeffEE, derClgeffTE, derClgeffBB],
        [derClmnuTT, derClmnuEE, derClmnuTE, derClmnuBB],
        pardict.as_bool("geff_fixed"),
        pardict.as_bool("neutrino_mass_fixed"),
    )

    # add prior on tau
    # Catch[-1][-1] += 1.0 / (0.01**2)

    Catch[0, :] *= 100.0  # 10000 theta_star -> 100 theta_star
    Catch[:, 0] *= 100.0  # 10000 theta_star -> 100 theta_star
    Catch[3, :] *= 100.0  # 100 Omegacdm -> Omegacdm
    Catch[:, 3] *= 100.0  # 100 Omegacdm -> Omegacdm

    # cov = np.linalg.inv(Catch)
    # console.log('matrix condition number: ', np.linalg.cond(Catch))

    # console.log('matrix condition number without A_phi: ', np.linalg.cond(Catch2))

    # exit()

    # exit()

    # import matplotlib.pyplot as plt
    # plt.imshow(np.log10(np.abs(Catch)), cmap="viridis")
    # plt.colorbar(
    # plt.show()

    # Catch = np.delete(Catch, 1, axis=0)  # remove A_phi
    # Catch = np.delete(Catch, 1, axis=1)

    # # plt.imshow(np.log10(np.abs(Catch)), cmap="viridis")
    # # plt.colorbar()
    # # plt.show()

    # console.log('matrix condition number: ', np.linalg.cond(Catch))

    # console.log(Catch - Catch.T)

    # console.log(Catch @ np.linalg.inv(Catch) - np.eye(len(Catch)))

    # print(Catch)
    # for i in range(len(Catch)):
    #     console.log(Catch[i, :])

    # Catch[1][1] += 1.0 / (0.1**2)  # add prior on A_phi

    # Catch[2][2] += 1.0 / (0.002**2)  # add prior on Omegab
    import copy

    Catch2 = copy.deepcopy(Catch)
    Catch2 = np.delete(Catch2, 1, axis=0)  # remove A_phi
    Catch2 = np.delete(Catch2, 1, axis=1)

    covCatch2 = np.linalg.inv(Catch2)

    # covCatch2[0, :] /= 100.0
    # covCatch2[:, 0] /= 100.0  # A_phi
    # covCatch2[2, :] /= 100.0
    # covCatch2[:, 2] /= 100.0  # Omegab

    errs2 = np.sqrt(np.diag(covCatch2))
    print(errs2)

    cov = np.linalg.inv(Catch)

    errs = np.sqrt(np.diag(cov))

    print(errs)

    means = np.array(
        [
            cosmo.theta_star,
            cosmo.A_phi,  # * 100.0,
            cosmo.Omegab * 100.0,
            cosmo.Omega_cdm,
            cosmo.lnAs10,
            cosmo.ns,
            cosmo.tau,
            cosmo.log10Geff,
        ]
    )
    if pardict.as_bool("geff_fixed"):
        means = np.array(
            [
                cosmo.theta_star,
                cosmo.A_phi,  # * 100.0,
                cosmo.Omegab * 100.0,
                cosmo.Omega_cdm,
                cosmo.lnAs10,
                cosmo.ns,
                cosmo.tau,
            ]
        )

    if not pardict.as_bool("neutrino_mass_fixed"):
        means = np.append(means, cosmo.mnu)

    if not pardict.as_bool("geff_fixed") or not pardict.as_bool("neutrino_mass_fixed"):
        txt = "{:.5f}    {:.5f}    {:.3f}    {:.3f}    {:.5f}    {:.5f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}".format(
            # txt = "{:.5f}    {:.5f}    {:.3f}    {:.3f}    {:.5f}    {:.5f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}".format(
            means[0],
            errs[0] / means[0] * 100.0,
            means[1],
            errs[1] / means[1] * 100.0,
            means[2],
            errs[2] / means[2] * 100.0,
            means[3],
            errs[3] / means[3] * 100.0,
            means[4],
            errs[4] / means[4] * 100.0,
            means[5],
            errs[5] / means[5] * 100.0,
            means[6],
            errs[6] / means[6] * 100.0,
            means[7],
            errs[7] / means[7] * 100.0,
        )
    else:
        txt = "{:.5f}    {:.5f}    {:.3f}    {:.3f}    {:.5f}    {:.5f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}".format(
            # txt = "{:.5f}    {:.5f}    {:.3f}    {:.3f}    {:.5f}    {:.5f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}".format(
            means[0],
            errs[0] / means[0] * 100.0,
            means[1],
            errs[1] / means[1] * 100.0,
            means[2],
            errs[2] / means[2] * 100.0,
            means[3],
            errs[3] / means[3] * 100.0,
            means[4],
            errs[4] / means[4] * 100.0,
            means[5],
            errs[5] / means[5] * 100.0,
            means[6],
            errs[6] / abs(means[6]) * 100.0,
        )

    console.log(txt)

    # Output the fisher matrix for the redshift bin
    write_fisher(
        pardict,
        cov,
        means,
    )

    # Catch_standard = copy.deepcopy(Catch)
    # Catch_standard = np.delete(Catch_standard, 1, axis=0)
    # Catch_standard = np.delete(Catch_standard, 1, axis=1)

    # print(np.sqrt(np.diag(np.linalg.inv(Catch_standard))))

    # console.log(Catch)

    # print(Catch @ np.linalg.inv(Catch) - np.eye(len(Catch)))

    # import matplotlib.pyplot as plt
    # plt.imshow(Catch)
    # plt.show()

    alpha_nu = 8.0 / 7.0 * (11.0 / 4.0) ** (4.0 / 3.0)
    eps_neff1 = 1.0 / (1.0 + alpha_nu)
    eps_neffstandard = 3.044 / (3.044 + alpha_nu)

    A_upper = cosmo.A_phi + errs[1]
    A_lower = cosmo.A_phi - errs[1]

    eps_upper = A_upper * (eps_neff1 - eps_neffstandard) + eps_neffstandard
    eps_lower = A_lower * (eps_neff1 - eps_neffstandard) + eps_neffstandard

    eps_centre = cosmo.A_phi * (eps_neff1 - eps_neffstandard) + eps_neffstandard

    neffLower = alpha_nu / (1.0 / eps_upper - 1.0)
    neffUpper = alpha_nu / (1.0 / eps_lower - 1.0)
    neff_phi = alpha_nu / (1.0 / eps_centre - 1.0)

    console.log(f"Neff from A_phi: {neff_phi:.6f}")
    console.log(f"Neff upper limit: {neffUpper:.6f}")
    console.log(f"Neff lower limit: {neffLower:.6f}")

    eps1 = 1.0 / (1.0 + alpha_nu)
    eps2 = 3.044 / (3.044 + alpha_nu)

    dA_dneff = 1.0 / (3.044 + alpha_nu) / (eps1 - eps2) - (
        3.044 / (3.044 + alpha_nu) ** 2
    ) / (eps1 - eps2)
    console.log("dneff /dA * sigmaA = sigmaNeff: ", abs(errs[1] / dA_dneff))
    console.log("sigmaA: ", errs[1])
    # exit()

    # make some pretty contour plots
    if pardict.as_bool("geff_fixed") and pardict.as_bool("neutrino_mass_fixed"):
        from chainconsumer import ChainConsumer, Chain
        import matplotlib.pyplot as plt

        c = ChainConsumer()
        c.add_chain(
            Chain.from_covariance(
                means,
                cov,
                columns=[
                    r"$100\theta_*$",
                    r"$A_{\phi}$",
                    r"$100\Omega_bh^2$",
                    r"$\Omega_{\mathrm{cdm}}h^2$",
                    r"$\ln(A_s10^{10})$",
                    r"$n_s$",
                    r"$\tau$",
                ],
                name="cov",
            )
        )

        c.plotter.plot()
        plt.show()

    elif not pardict.as_bool("geff_fixed") and pardict.as_bool("neutrino_mass_fixed"):
        # plot the contour for beta_phi and alpha_iso
        from chainconsumer import ChainConsumer, Chain
        import matplotlib.pyplot as plt

        c = ChainConsumer()
        c.add_chain(
            Chain.from_covariance(
                means,
                cov,
                columns=[
                    r"$100\theta_*$",
                    r"$A_{\phi}$",
                    r"$100\Omega_bh^2$",
                    r"$\Omega_{\mathrm{cdm}}h^2$",
                    r"$\ln(A_s10^{10})$",
                    r"$n_s$",
                    r"$\tau$",
                    r"$\log_{10}G_{\mathrm{eff}}$",
                ],
                name="cov",
            )
        )
        c.plotter.plot()
        plt.show()

    elif pardict.as_bool("geff_fixed") and not pardict.as_bool("neutrino_mass_fixed"):
        # plot the contour for beta_phi and alpha_iso
        from chainconsumer import ChainConsumer, Chain
        import matplotlib.pyplot as plt

        c = ChainConsumer()
        c.add_chain(
            Chain.from_covariance(
                means,
                cov,
                columns=[
                    r"$100\theta_*$",
                    r"$A_{\phi}$",
                    r"$100\Omega_bh^2$",
                    r"$\Omega_{\mathrm{cdm}}h^2$",
                    r"$\ln(A_s10^{10})$",
                    r"$n_s$",
                    r"$\tau$",
                    r"$m_\nu$",
                ],
                name="cov",
            )
        )
        c.plotter.plot()
        plt.show()
