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

    # Set up the linear power spectrum and derived parameters based on the input cosmology
    cosmo = CosmoResults(pardict)

    console.log("computed CAMB CMB temperature-temperature power spectrum.")

    console.log("Fitting geff?")
    console.log((not pardict.as_bool("geff_fixed")))

    # Precompute some things we might need for the Fisher matrix
    derivatives = Set_Bait(
        cosmo,
        geff_fixed=pardict.as_bool("geff_fixed"),
    )
    derClthetastar = derivatives[0]
    derClA = derivatives[1]

    derClgeff = None
    if not pardict.as_bool("geff_fixed"):
        derClgeff = derivatives[2]

    console.log(
        "Computed derivatives of the power spectrum w.r.t. forecast parameters."
    )

    # Loop over redshifts and compute the Fisher matrix and output the 3x3 matrix
    FullCatch = np.zeros((3, 3))

    if pardict.as_bool("geff_fixed"):
        FullCatch = np.zeros((2, 2))

        console.log("#  theta_star  theta_star_err(%)  A(Neff)  A(Neff)_err(%)")

    else:
        console.log(
            "#  theta_star  theta_star_err(%)  A(Neff)  A(Neff)_err(%)  log10Geff  geff_err(%)"
        )

    Catch = Fish(
        cosmo,
        float(pardict["lmin"]),
        float(pardict["lmax"]),
        derClthetastar,
        derClA,
        derClgeff,
        pardict.as_bool("geff_fixed"),
    )

    cov = np.linalg.inv(Catch)
    errs = np.sqrt(np.diag(cov))
    means = np.array([cosmo.theta_star, cosmo.A_phi, cosmo.log10Geff])
    if pardict.as_bool("geff_fixed"):
        means = np.array([cosmo.theta_star, cosmo.A_phi])

    if not pardict.as_bool("geff_fixed"):
        txt = f" {0:.2f}    {0:.2f}     {0:.2f}       {0:.2f}         {0:.2f}       {0:.2f} ".format(
            means[0] / 100.0,
            errs[0] / means[0],
            means[1],
            errs[1] / means[1] * 100.0,
            means[2],
            errs[2] / means[2] * 100.0,
        )
    else:
        txt = f" {0:.2f}    {0:.2f}     {0:.2f}       {0:.2f} ".format(
            means[0],
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
                    r"$\theta_*$",
                    r"$A_{\phi}$",
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
                    r"$\theta_*$",
                    r"$A_{\phi}$",
                    r"$\log_{10}G_{\mathrm{eff}}$",
                ],
                name="cov",
            )
        )
        c.plotter.plot()
        plt.show()
