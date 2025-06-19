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

    fracstepthetastar = 0.01
    fracstepomegab = 0.01
    fracstepomegacdm = 0.01
    fracstepAs = 0.01
    fracstepns = 0.01
    fracsteptau = 0.01

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
    derClthetastar = derivatives[0]
    derClA = derivatives[1]
    derClOmegab = derivatives[2]
    derClOmegacdm = derivatives[3]
    derClAs = derivatives[4]
    derClns = derivatives[5]
    derCltau = derivatives[6]

    derClgeff = None
    if not pardict.as_bool("geff_fixed"):
        derClgeff = derivatives[7]

    # import matplotlib.pyplot as plt
    # plt.plot(cosmo.ell, derClA(cosmo.ell))
    # plt.show()
    # exit()

    # import matplotlib.pyplot as plt
    # plt.plot(cosmo.ell, splev(cosmo.ell, cosmo.clTT))
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
            "#  100theta_star  100theta_star_err(%)  A(Neff)  A(Neff)_err(%)  Omegab   Omegab_err(%)  Omegacdm  Omegacdm_err(%)  lnAs10  lnAs10_err(%)  ns  ns_err(%)  tau  tau_err(%)"
        )

    else:
        console.log(
            "#  100theta_star  100theta_star_err(%)  A(Neff)  A(Neff)_err(%)  Omegab   Omegab_err(%)  Omegacdm  Omegacdm_err(%)  lnAs10  lnAs10_err(%)  ns  ns_err(%)  tau  tau_err(%)  log10Geff  geff_err(%)"
        )

    Catch = Fish(
        cosmo,
        float(pardict["lmin"]),
        float(pardict["lmax"]),
        derClthetastar,
        derClA,
        derClOmegab,
        derClOmegacdm,
        derClAs,
        derClns,
        derCltau,
        derClgeff,
        pardict.as_bool("geff_fixed"),
    )

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
        txt = "{:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}".format(
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
        txt = "{:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}".format(
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
                    r"$\Omega_b$",
                    r"$\Omega_{\mathrm{cdm}}$",
                    r"$ln(A_s10^{10})$",
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
                    r"$\Omega_b$",
                    r"$\Omega_{\mathrm{cdm}}$",
                    r"$ln(A_s10^{10})$",
                    r"$n_s$",
                    r"$\tau$",
                    r"$\log_{10}G_{\mathrm{eff}}$",
                ],
                name="cov",
            )
        )
        c.plotter.plot()
        plt.show()
