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

    fracstepthetastar = 0.0001
    fracstepomegab = 0.0001
    fracstepomegacdm = 0.0001
    fracstepAs = 0.0001
    fracstepns = 0.0001
    fracsteptau = 0.00001

    # Set up the linear power spectrum and derived parameters based on the input cosmology
    cosmo = CosmoResults(
        pardict,
        fracstepthetastar=fracstepthetastar,
        fracstepomegab=fracstepomegab,
        fracstepomegacdm=fracstepomegacdm,
        fracstepAs=fracstepAs,
        fracstepns=fracstepns,
        fracsteptau=fracsteptau,
    )

    console.log("computed CAMB CMB temperature-temperature power spectrum.")

    console.log("Fitting geff?")
    console.log((not pardict.as_bool("geff_fixed")))

    # Precompute some things we might need for the Fisher matrix
    derivatives = Set_Bait(
        cosmo,
        geff_fixed=pardict.as_bool("geff_fixed"),
        fracstepthetastar=fracstepthetastar,
        fracstepomegab=fracstepomegab,
        fracstepomegacdm=fracstepomegacdm,
        fracstepAs=fracstepAs,
        fracstepns=fracstepns,
        fracsteptau=fracsteptau,
    )

    derClATT, derClAEE, derClATE = derivatives[0]
    derClthetastarTT, derClthetastarEE, derClthetastarTE = derivatives[1]
    derClOmegabTT, derClOmegabEE, derClOmegabTE = derivatives[2]
    derClOmegacdmTT, derClOmegacdmEE, derClOmegacdmTE = derivatives[3]
    derClAsTT, derClAsEE, derClAsTE = derivatives[4]
    derClnsTT, derClnsEE, derClnsTE = derivatives[5]
    derCltauTT, derCltauEE, derCltauTE = derivatives[6]
    derClgeff = []
    if not pardict.as_bool("geff_fixed"):
        derClgeffTT, derClgeffEE, derClgeffTE = derivatives[7]

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

    if pardict.as_bool("geff_fixed"):
        console.log(
            "#  100theta_star  100theta_star_err(%)  A(Neff)  A(Neff)_err(%)  Omegab   Omegab_err(%)  Omegacdm  Omegacdm_err(%)  As10  As10_err(%)  ns  ns_err(%)  tau  tau_err(%)"
        )

    else:
        console.log(
            "#  100theta_star  100theta_star_err(%)  A(Neff)  A(Neff)_err(%)  Omegab   Omegab_err(%)  Omegacdm  Omegacdm_err(%)  As10  As10_err(%)  ns  ns_err(%)  tau  tau_err(%)  log10Geff  geff_err(%)"
        )

    Catch = Fish(
        cosmo,
        [derClthetastarTT, derClthetastarEE, derClthetastarTE],
        [derClATT, derClAEE, derClATE],
        [derClOmegabTT, derClOmegabEE, derClOmegabTE],
        [derClOmegacdmTT, derClOmegacdmEE, derClOmegacdmTE],
        [derClAsTT, derClAsEE, derClAsTE],
        [derClnsTT, derClnsEE, derClnsTE],
        [derCltauTT, derCltauEE, derCltauTE],
        derClgeff,
        pardict.as_bool("geff_fixed"),
    )

    # add prior on tau
    # Catch[-1][-1] += 1.0 / (0.01**2)

    # cov = np.linalg.inv(Catch)
    # print(np.linalg.cond(Catch))

    # import matplotlib.pyplot as plt
    # plt.imshow(np.log10(np.abs(Catch)), cmap="viridis")
    # plt.colorbar()
    # plt.show()

    # for i in range(len(Catch)):
    #     for j in range(len(Catch)):
    #         if i < j:
    #             Catch[i, j] = Catch[j, i]

    # Catch[1][1] += 1.0 / (0.1**2)  # add prior on A_phi

    # Catch[2][2] += 1.0 / (0.002**2)  # add prior on Omegab

    cov = np.linalg.inv(Catch)

    errs = np.sqrt(np.diag(cov))
    means = np.array(
        [
            cosmo.theta_star,
            cosmo.A_phi,
            cosmo.Omegab,
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
                cosmo.A_phi,
                cosmo.Omegab,
                cosmo.Omega_cdm,
                cosmo.lnAs10,
                cosmo.ns,
                cosmo.tau,
            ]
        )

    if not pardict.as_bool("geff_fixed"):
        txt = "{:.5f}    {:.5f}    {:.3f}    {:.3f}    {:.5f}    {:.5f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}".format(
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

    # console.log(
    #     Catch
    # )

    # import matplotlib.pyplot as plt
    # plt.imshow(Catch)

    # plt.show()

    # for i in range(len(Catch)):
    #     for j in range(len(Catch)):
    #         if i < j:
    #             print(i, j)
    #             print(Catch[i,j] - Catch[j,i])

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

    # make some pretty contour plots
    if pardict.as_bool("geff_fixed"):
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
                    r"$\Omega_bh^2$",
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

    else:
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
                    r"$\Omega_b$h^2",
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
