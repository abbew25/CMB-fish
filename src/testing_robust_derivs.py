import sys
from configobj import ConfigObj
from fishercomputations import Set_Bait
from setup import CosmoResults
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

    pardict["neutrino_mass_fixed"] = False

    fracstepthetastar = 0.00002  # good
    fracstepomegab = 0.05
    fracstepomegacdm = 0.05
    fracstepAs = 0.01  # good
    fracstepns = 0.0002  # good
    fracsteptau = 0.001  # good
    fracstepmnu = 0.05  # good

    # fracstepthetastar = 0.03
    # fracstepomegab = 0.03
    # fracstepomegacdm = 0.03
    # fracstepAs = 0.03
    # fracstepns = 0.03
    # fracsteptau = 0.03

    import matplotlib.pyplot as plt

    f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(5, 10))

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

    # derClATT, derClAEE, derClATE = compute_deriv_phiamplitude(cosmo, dl=0.01)

    # derClATT2, derClAEE2, derClATE2 = compute_deriv_phiamplitude(cosmo, dl=0.001)

    # derClATT3, derClAEE3, derClATE3 = compute_deriv_phiamplitude(cosmo, dl=0.0005)

    # plt.plot(
    #     cosmo.ell,
    #     derClATT2(cosmo.ell) / derClATT(cosmo.ell),
    #     label=r"$\partial C_\ell^{TT}/\partial \phi_{amplitude}$",
    #     color="red",
    # )
    # plt.plot(
    #     cosmo.ell,
    #     derClAEE2(cosmo.ell) / derClAEE(cosmo.ell),
    #     label=r"$\partial C_\ell^{EE}/\partial \phi_{amplitude}$",
    #     color="red",
    #     linestyle="--",
    # )
    # plt.plot(
    #     cosmo.ell,
    #     derClATE2(cosmo.ell) / derClATE(cosmo.ell),
    #     label=r"$\partial C_\ell^{TE}/\partial \phi_{amplitude}$",
    #     color="red",
    #     linestyle=":",
    # )

    # plt.plot(
    #     cosmo.ell,
    #     derClATT3(cosmo.ell) / derClATT(cosmo.ell),
    #     label=r"$\partial C_\ell^{TT}/\partial \phi_{amplitude}$",
    #     color="blue",
    # )
    # plt.plot(
    #     cosmo.ell,
    #     derClAEE3(cosmo.ell) / derClAEE(cosmo.ell),
    #     label=r"$\partial C_\ell^{EE}/\partial \phi_{amplitude}$",
    #     color="blue",
    #     linestyle="--",
    # )
    # plt.plot(
    #     cosmo.ell,
    #     derClATE3(cosmo.ell) / derClATE(cosmo.ell),
    #     label=r"$\partial C_\ell^{TE}/\partial \phi_{amplitude}$",
    #     color="blue",
    #     linestyle=":",
    # )

    # plt.show()

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
    derClOmegacdmTT, derClOmegacdmEE, derClOmegacdmTE, derClOmegacdmBB = derivatives[3]
    derClAsTT, derClAsEE, derClAsTE, derClAsBB = derivatives[4]
    derClnsTT, derClnsEE, derClnsTE, derClnsBB = derivatives[5]
    derCltauTT, derCltauEE, derCltauTE, derCltauBB = derivatives[6]
    derClgeffTT, derClgeffEE, derClgeffTE, derClgeffBB = None, None, None, None
    derClmnuTT, derClmnuEE, derClmnuTE, derClmnuBB = None, None, None, None
    if not pardict.as_bool("geff_fixed"):
        derClgeffTT, derClgeffEE, derClgeffTE, derClgeffBB = derivatives[7]
    if not pardict.as_bool("neutrino_mass_fixed"):
        derClmnuTT, derClmnuEE, derClmnuTE, derClmnuBB = derivatives[7]

    # ax1.plot(cosmo.ell, derClthetastarTT(cosmo.ell), label=r"$\partial C_\ell^{TT}/\partial 100\theta_*$", color='red')
    # ax1.plot(cosmo.ell, derClthetastarEE(cosmo.ell), label=r"$\partial C_\ell^{EE}/\partial 100\theta_*$",  color='red', linestyle='--')
    # ax1.plot(cosmo.ell, derClthetastarTE(cosmo.ell), label=r"$\partial C_\ell^{TE}/\partial 100\theta_*$", color='red', linestyle=':')

    # ax2.plot(cosmo.ell, derCltauTT(cosmo.ell), label=r"$\partial C_\ell^{TT}/\partial tau$", color='red')
    # ax2.plot(cosmo.ell, derCltauEE(cosmo.ell), label=r"$\partial C_\ell^{EE}/\partial tau$", color='red', linestyle='--')
    # ax2.plot(cosmo.ell, derCltauTE(cosmo.ell), label=r"$\partial C_\ell^{TE}/\partial tau$", color='red', linestyle=':')

    # ax3.plot(
    #     cosmo.ell,
    #     derClOmegabTT(cosmo.ell),
    #     label=r"$\partial C_\ell^{TT}/\partial \Omega_b$",
    #     color="red",
    # )
    # ax3.plot(
    #     cosmo.ell,
    #     derClthetastarEE(cosmo.ell),
    #     label=r"$\partial C_\ell^{EE}/\partial \Omega_b$",
    #     color="red",
    #     linestyle="--",
    # )
    # ax3.plot(
    #     cosmo.ell,
    #     derClOmegabTE(cosmo.ell),
    #     label=r"$\partial C_\ell^{TE}/\partial \Omega_b$",
    #     color="red",
    #     linestyle=":",
    # )

    # ax4.plot(
    #     cosmo.ell,
    #     derClOmegacdmTT(cosmo.ell),
    #     label=r"$\partial C_\ell^{TT}/\partial \Omega_{\mathrm{cdm}}$",
    #     color="red",
    # )
    # ax4.plot(
    #     cosmo.ell,
    #     derClnsEE(cosmo.ell),
    #     label=r"$\partial C_\ell^{EE}/\partial \Omega_{\mathrm{cdm}}$",
    #     color="red",
    #     linestyle="--",
    # )
    # ax4.plot(
    #     cosmo.ell,
    #     derClOmegacdmTE(cosmo.ell),
    #     label=r"$\partial C_\ell^{TE}/\partial \Omega_{\mathrm{cdm}}$",
    #     color="red",
    #     linestyle=":",
    # )

    # plt.show()
    # exit()

    # ax5.plot(cosmo.ell, derClAsTT(cosmo.ell), label=r"$\partial C_\ell^{TT}/\partial \ln A_s10^{10}$", color='red')
    # ax5.plot(cosmo.ell, derClAsEE(cosmo.ell), label=r"$\partial C_\ell^{EE}/\partial \ln A_s10^{10}$", color='red', linestyle='--')
    # ax5.plot(cosmo.ell, derClAsTE(cosmo.ell), label=r"$\partial C_\ell^{TE}/\partial \ln A_s10^{10}$", color='red', linestyle=':')

    # ax6.plot(cosmo.ell, derClnsTT(cosmo.ell), label=r"$\partial C_\ell^{TT}/\partial n_s$", color='red')
    # ax6.plot(cosmo.ell, derClnsEE(cosmo.ell), label=r"$\partial C_\ell^{EE}/\partial n_s$", color='red', linestyle='--')
    # ax6.plot(cosmo.ell, derClnsTE(cosmo.ell), label=r"$\partial C_\ell^{TE}/\partial n_s$", color='red', linestyle=':')

    # ax1.set_title('theta*')
    # ax2.set_title('tau')
    # ax3.set_title('Omegab')
    # ax4.set_title('Omegacdm')
    # ax5.set_title('As')
    # ax6.set_title('ns')

    # plt.show()
    # fracstepthetastar = 0.005
    # fracstepomegab = 0.01
    # fracstepomegacdm = 0.01
    # fracstepAs = 0.005
    # fracstepns = 0.005
    # fracsteptau = 0.005
    # fracstepmnu = 0.005

    fracstepthetastar = 0.00001  # good
    fracstepomegab = 0.04
    fracstepomegacdm = 0.04
    fracstepAs = 0.0002  # good
    fracstepns = 0.0001  # good
    fracsteptau = 0.002  # good
    fracstepmnu = 0.04  # good

    cosmo2 = CosmoResults(
        pardict,
        fracstepthetastar=fracstepthetastar,
        fracstepomegab=fracstepomegab,
        fracstepomegacdm=fracstepomegacdm,
        fracstepAs=fracstepAs,
        fracstepns=fracstepns,
        fracsteptau=fracsteptau,
        fracstepmnu=fracstepmnu,
    )

    derivatives2 = Set_Bait(
        cosmo2,
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

    derClthetastarTT2, derClthetastarEE2, derClthetastarTE2, derClthetastarBB2 = (
        derivatives2[1]
    )
    derClATT2, derClAEE2, derClATE2, derClABB2 = derivatives2[0]
    derClOmegabTT2, derClOmegabEE2, derClOmegabTE2, derCLOmegabBB2 = derivatives2[2]
    derClOmegacdmTT2, derClOmegacdmEE2, derClOmegacdmTE2, derClOmegacdmBB2 = (
        derivatives2[3]
    )
    derClAsTT2, derClAsEE2, derClAsTE2, derClAsBB2 = derivatives2[4]
    derClnsTT2, derClnsEE2, derClnsTE2, derClnsBB2 = derivatives2[5]
    derCltauTT2, derCltauEE2, derCltauTE2, derCltauBB2 = derivatives2[6]
    derClmnuTT2, derClmnuEE2, derClmnuTE2, derClmnuBB2 = derivatives2[7]

    ax1.plot(
        cosmo.ell,
        derClthetastarTT2 / derClthetastarTT,
        label=r"$\partial C_\ell^{TT}/\partial 100\theta_*$",
        color="blue",
    )
    ax1.plot(
        cosmo.ell,
        derClthetastarEE2 / derClthetastarEE,
        label=r"$\partial C_\ell^{EE}/\partial 100\theta_*$",
        color="blue",
        linestyle="--",
    )
    ax1.plot(
        cosmo.ell,
        derClthetastarTE2 / derClthetastarTE,
        label=r"$\partial C_\ell^{TE}/\partial 100\theta_*$",
        color="blue",
        linestyle=":",
    )
    ax2.plot(
        cosmo.ell,
        derClthetastarBB2 / derClthetastarBB,
        label=r"$\partial C_\ell^{TT}/\partial \tau$",
        color="blue",
        linestyle="-.",
    )

    ax2.plot(
        cosmo.ell,
        derCltauTT2 / derCltauTT,
        label=r"$\partial C_\ell^{TT}/\partial A$",
        color="blue",
    )
    ax2.plot(
        cosmo.ell,
        derCltauEE2 / derCltauEE,
        label=r"$\partial C_\ell^{EE}/\partial A$",
        color="blue",
        linestyle="--",
    )
    ax2.plot(
        cosmo.ell,
        derCltauTE2 / derCltauTE,
        label=r"$\partial C_\ell^{TE}/\partial A$",
        color="blue",
        linestyle=":",
    )
    ax2.plot(
        cosmo.ell,
        derCltauBB2 / derCltauBB,
        label=r"$\partial C_\ell^{BB}/\partial A$",
        color="blue",
        linestyle="-.",
    )

    ax3.plot(
        cosmo.ell,
        derClOmegabTT2 / derClOmegabTT,
        label=r"$\partial C_\ell^{TT}/\partial \Omega_b$",
        color="blue",
    )
    ax3.plot(
        cosmo.ell,
        derClOmegabEE2 / derClOmegabEE,
        label=r"$\partial C_\ell^{EE}/\partial \Omega_b$",
        color="blue",
        linestyle="--",
    )
    ax3.plot(
        cosmo.ell,
        derClOmegabTE2 / derClOmegabTE,
        label=r"$\partial C_\ell^{TE}/\partial \Omega_b$",
        color="blue",
        linestyle=":",
    )
    ax3.plot(
        cosmo.ell,
        derCLOmegabBB2 / derCLOmegabBB,
        label=r"$\partial C_\ell^{BB}/\partial \Omega_b$",
        color="blue",
        linestyle="-.",
    )

    ax4.plot(
        cosmo.ell,
        derClmnuTT2 / derClmnuTT,
        label=r"$\partial C_\ell^{TT}/\partial \Omega_{\mathrm{cdm}}$",
        color="blue",
    )
    ax4.plot(
        cosmo.ell,
        derClmnuEE2 / derClmnuEE,
        label=r"$\partial C_\ell^{EE}/\partial \Omega_{\mathrm{cdm}}$",
        color="blue",
        linestyle="--",
    )
    ax4.plot(
        cosmo.ell,
        derClmnuTE2 / derClmnuTE,
        label=r"$\partial C_\ell^{TE}/\partial \Omega_{\mathrm{cdm}}$",
        color="blue",
        linestyle=":",
    )
    ax4.plot(
        cosmo.ell,
        derClmnuBB2 / derClmnuBB,
        label=r"$\partial C_\ell^{BB}/\partial \Omega_{\mathrm{cdm}}$",
        color="blue",
        linestyle="-.",
    )

    # ax3.plot(
    #     cosmo.ell,
    #     derClOmegabTT2(cosmo.ell),
    #     label=r"$\partial C_\ell^{TT}/\partial \Omega_b$",
    #     color="blue",
    # )
    # ax3.plot(
    #     cosmo.ell,
    #     derClOmegabEE2(cosmo.ell),
    #     label=r"$\partial C_\ell^{EE}/\partial \Omega_b$",
    #     color="blue",
    #     linestyle="--",
    # )
    # ax3.plot(
    #     cosmo.ell,
    #     derClOmegabTE2(cosmo.ell),
    #     label=r"$\partial C_\ell^{TE}/\partial \Omega_b$",
    #     color="blue",
    #     linestyle=":",
    # )

    # ax4.plot(
    #     cosmo.ell,
    #     derClOmegacdmTT2(cosmo.ell),
    #     label=r"$\partial C_\ell^{TT}/\partial \Omega_{\mathrm{cdm}}$",
    #     color="blue",
    # )
    # ax4.plot(
    #     cosmo.ell,
    #     derClOmegacdmEE2(cosmo.ell),
    #     label=r"$\partial C_\ell^{EE}/\partial \Omega_{\mathrm{cdm}}$",
    #     color="blue",
    #     linestyle="--",
    # )
    # ax4.plot(
    #     cosmo.ell,
    #     derClOmegacdmTE2(cosmo.ell),
    #     label=r"$\partial C_\ell^{TE}/\partial \Omega_{\mathrm{cdm}}$",
    #     color="blue",
    #     linestyle=":",
    # )

    ax5.plot(
        cosmo.ell,
        derClAsTT2 / derClAsTT,
        label=r"$\partial C_\ell^{TT}/\partial \ln A_s10^{10}$",
        color="blue",
    )
    ax5.plot(
        cosmo.ell,
        derClAsEE2 / derClAsEE,
        label=r"$\partial C_\ell^{EE}/\partial \ln A_s10^{10}$",
        color="blue",
        linestyle="--",
    )
    ax5.plot(
        cosmo.ell,
        derClAsTE2 / derClAsTE,
        label=r"$\partial C_\ell^{TE}/\partial \ln A_s10^{10}$",
        color="blue",
        linestyle=":",
    )
    ax5.plot(
        cosmo.ell,
        derClAsBB2 / derClAsBB,
        label=r"$\partial C_\ell^{BB}/\partial \ln A_s10^{10}$",
        color="blue",
        linestyle="-.",
    )

    ax6.plot(
        cosmo.ell,
        derClnsTT2 / derClnsTT,
        label=r"$\partial C_\ell^{TT}/\partial n_s$",
        color="blue",
    )
    ax6.plot(
        cosmo.ell,
        derClnsEE2 / derClnsEE,
        label=r"$\partial C_\ell^{EE}/\partial n_s$",
        color="blue",
        linestyle="--",
    )
    ax6.plot(
        cosmo.ell,
        derClnsTE2 / derClnsTE,
        label=r"$\partial C_\ell^{TE}/\partial n_s$",
        color="blue",
        linestyle=":",
    )
    ax6.plot(
        cosmo.ell,
        derClnsBB2 / derClnsBB,
        label=r"$\partial C_\ell^{BB}/\partial n_s$",
        color="blue",
        linestyle="-.",
    )
    plt.show()

    # fracstepthetastar = 0.0005
    # fracstepomegab = 0.0005
    # fracstepomegacdm = 0.0005
    # fracstepAs = 0.0005
    # fracstepns = 0.0005
    # fracsteptau = 0.0005

    # cosmo3 = CosmoResults(
    #     pardict,
    #     fracstepthetastar=fracstepthetastar,
    #     fracstepomegab=fracstepomegab,
    #     fracstepomegacdm=fracstepomegacdm,
    #     fracstepAs=fracstepAs,
    #     fracstepns=fracstepns,
    #     fracsteptau=fracsteptau,
    # )

    # derivatives3 = Set_Bait(
    #     cosmo3,
    #     geff_fixed=pardict.as_bool("geff_fixed"),
    #     fracstepthetastar=fracstepthetastar,
    #     fracstepomegab=fracstepomegab,
    #     fracstepomegacdm=fracstepomegacdm,
    #     fracstepAs=fracstepAs,
    #     fracstepns=fracstepns,
    #     fracsteptau=fracsteptau,
    # )

    # derClthetastarTT3, derClthetastarEE3, derClthetastarTE3 = derivatives3[0]
    # derClATT3, derClAEE3, derClATE3 = derivatives3[1]
    # derClOmegabTT3, derClOmegabEE3, derClOmegabTE3 = derivatives3[2]
    # derClOmegacdmTT3, derClOmegacdmEE3, derClOmegacdmTE3 = derivatives3[3]
    # derClAsTT3, derClAsEE3, derClAsTE3 = derivatives3[4]
    # derClnsTT3, derClnsEE3, derClnsTE3 = derivatives3[5]
    # derCltauTT3, derCltauEE3, derCltauTE3 = derivatives3[6]

    # ax1.plot(cosmo.ell, derClthetastarTT3(cosmo3.ell)/derClthetastarTT(cosmo.ell), label=r"$\partial C_\ell^{TT}/\partial 100\theta_*$", color='green')
    # ax1.plot(cosmo.ell, derClthetastarEE3(cosmo3.ell)/derClthetastarEE(cosmo.ell), label=r"$\partial C_\ell^{EE}/\partial 100\theta_*$",  color='green', linestyle='--')
    # ax1.plot(cosmo.ell, derClthetastarTE3(cosmo3.ell)/derClthetastarTE(cosmo.ell), label=r"$\partial C_\ell^{TE}/\partial 100\theta_*$", color='green', linestyle=':')

    # ax2.plot(cosmo.ell, derCltauTT3(cosmo3.ell)/derCltauTT(cosmo.ell), label=r"$\partial C_\ell^{TT}/\partial tau$", color='green')
    # ax2.plot(cosmo.ell, derCltauEE3(cosmo3.ell)/derCltauEE(cosmo.ell), label=r"$\partial C_\ell^{EE}/\partial tau$", color='green', linestyle='--')
    # ax2.plot(cosmo.ell, derCltauTE3(cosmo3.ell)/derCltauTE(cosmo.ell), label=r"$\partial C_\ell^{TE}/\partial tau$", color='green', linestyle=':')

    # ax3.plot(cosmo.ell, derClOmegabTT3(cosmo.ell)/derClOmegabTT(cosmo.ell), label=r"$\partial C_\ell^{TT}/\partial \Omega_b$", color='green')
    # ax3.plot(cosmo.ell, derClOmegabEE3(cosmo.ell)/derClOmegabEE(cosmo.ell), label=r"$\partial C_\ell^{EE}/\partial \Omega_b$", color='green', linestyle='--')
    # ax3.plot(cosmo.ell, derClOmegabTE3(cosmo.ell)/derClOmegabTE(cosmo.ell), label=r"$\partial C_\ell^{TE}/\partial \Omega_b$", color='green', linestyle=':')

    # ax4.plot(cosmo.ell, derClOmegacdmTT3(cosmo.ell)/derClOmegacdmTT(cosmo.ell), label=r"$\partial C_\ell^{TT}/\partial \Omega_{\mathrm{cdm}}$", color='green')
    # ax4.plot(cosmo.ell, derClOmegacdmEE3(cosmo.ell)/derClOmegacdmEE(cosmo.ell), label=r"$\partial C_\ell^{EE}/\partial \Omega_{\mathrm{cdm}}$", color='green', linestyle='--')
    # ax4.plot(cosmo.ell, derClOmegacdmTE3(cosmo.ell)/derClOmegacdmTE(cosmo.ell), label=r"$\partial C_\ell^{TE}/\partial \Omega_{\mathrm{cdm}}$", color='green', linestyle=':')

    # ax5.plot(cosmo.ell, derClAsTT3(cosmo3.ell)/derClAsTT(cosmo.ell), label=r"$\partial C_\ell^{TT}/\partial \ln A_s10^{10}$", color='green')
    # ax5.plot(cosmo.ell, derClAsEE3(cosmo3.ell)/derClAsEE(cosmo.ell), label=r"$\partial C_\ell^{EE}/\partial \ln A_s10^{10}$", color='green', linestyle='--')
    # ax5.plot(cosmo.ell, derClAsTE3(cosmo3.ell)/derClAsTE(cosmo.ell), label=r"$\partial C_\ell^{TE}/\partial \ln A_s10^{10}$", color='green', linestyle=':')

    # ax6.plot(cosmo.ell, derClnsTT3(cosmo3.ell)/derClnsTT(cosmo.ell), label=r"$\partial C_\ell^{TT}/\partial n_s$", color='green')
    # ax6.plot(cosmo.ell, derClnsEE3(cosmo3.ell)/derClnsEE(cosmo.ell), label=r"$\partial C_\ell^{EE}/\partial n_s$", color='green', linestyle='--')
    # ax6.plot(cosmo.ell, derClnsTE3(cosmo3.ell)/derClnsTE(cosmo.ell), label=r"$\partial C_\ell^{TE}/\partial n_s$", color='green', linestyle=':')

    # ax1.legend()
    # ax2.legend()
    # ax3.legend()
    # ax4.legend()
    # ax5.legend()
    # ax6.legend()
    # plt.show()
    # exit()
