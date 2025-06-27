import numpy as np
from findiff import FinDiff
from scipy.integrate import simpson as simps
from scipy.interpolate import splev
from setup import CosmoResults, fitting_formula_Montefalcone2025
import numpy.typing as npt
from scipy.interpolate import interp1d


def Set_Bait(
    cosmo: CosmoResults,
    geff_fixed: bool = True,
    fracstepthetastar: float = 0.002,
    fracstepomegab: float = 0.002,
    fracstepomegacdm: float = 0.002,
    fracstepAs: float = 0.002,
    fracstepns: float = 0.002,
    fracsteptau: float = 0.002,
):
    derClthetastar = compute_deriv_thetastar(cosmo, fracstep=fracstepthetastar)
    derClAphi = compute_deriv_phiamplitude(cosmo)
    derClOmegab = compute_deriv_omegab(cosmo, fracstep=fracstepomegab)
    derClOmegacdm = compute_deriv_omegacdm(cosmo, fracstep=fracstepomegacdm)
    derClAs = compute_deriv_As(cosmo, fracstep=fracstepAs)
    derClns = compute_deriv_ns(cosmo, fracstep=fracstepns)
    derCltau = compute_deriv_tau(cosmo, fracstep=fracsteptau)

    if geff_fixed:
        return (
            derClAphi,
            derClthetastar,
            derClOmegab,
            derClOmegacdm,
            derClAs,
            derClns,
            derCltau,
        )
    elif not geff_fixed:
        return None


def compute_deriv_thetastar(cosmo: CosmoResults, fracstep: float = 0.002):
    cl_before = splev(cosmo.ell, cosmo.clTTEETE_minthetastar[0])
    cl_after = splev(cosmo.ell, cosmo.clTTEETE_plusthetastar[0])
    cl_centre = splev(cosmo.ell, cosmo.clTT)
    cl_before2 = splev(cosmo.ell, cosmo.clTTEETE_minthetastar2[0])
    cl_after2 = splev(cosmo.ell, cosmo.clTTEETE_plusthetastar2[0])

    CLs = np.zeros((5, len(cosmo.ell)))
    CLs[0, :] = cl_before2
    CLs[1, :] = cl_before
    CLs[2, :] = cl_centre
    CLs[3, :] = cl_after
    CLs[4, :] = cl_after2

    d_dthetastar = FinDiff(0, fracstep * cosmo.theta_star, acc=2)
    derCl_thetastar = d_dthetastar(CLs)

    derCl_thetastar_interpTT = interp1d(cosmo.ell, derCl_thetastar[2], kind="cubic")

    cl_before = splev(cosmo.ell, cosmo.clTTEETE_minthetastar[1])
    cl_after = splev(cosmo.ell, cosmo.clTTEETE_plusthetastar[1])
    cl_centre = splev(cosmo.ell, cosmo.clEE)
    cl_before2 = splev(cosmo.ell, cosmo.clTTEETE_minthetastar2[1])
    cl_after2 = splev(cosmo.ell, cosmo.clTTEETE_plusthetastar2[1])

    CLs = np.zeros((5, len(cosmo.ell)))
    CLs[0] = cl_before2
    CLs[1] = cl_before
    CLs[2] = cl_centre
    CLs[3] = cl_after
    CLs[4] = cl_after2

    derCl_thetastar_EE = d_dthetastar(CLs)
    derCl_thetastar_interpEE = interp1d(cosmo.ell, derCl_thetastar_EE[2], kind="cubic")

    cl_before = splev(cosmo.ell, cosmo.clTTEETE_minthetastar[2])
    cl_after = splev(cosmo.ell, cosmo.clTTEETE_plusthetastar[2])
    cl_centre = splev(cosmo.ell, cosmo.clTE)
    cl_before2 = splev(cosmo.ell, cosmo.clTTEETE_minthetastar2[2])
    cl_after2 = splev(cosmo.ell, cosmo.clTTEETE_plusthetastar2[2])

    CLs = np.zeros((5, len(cosmo.ell)))
    CLs[0] = cl_before2
    CLs[1] = cl_before
    CLs[2] = cl_centre
    CLs[3] = cl_after
    CLs[4] = cl_after2

    derCl_thetastar_TE = d_dthetastar(CLs)
    derCl_thetastar_interpTE = interp1d(cosmo.ell, derCl_thetastar_TE[2], kind="cubic")

    return derCl_thetastar_interpTT, derCl_thetastar_interpEE, derCl_thetastar_interpTE


def compute_deriv_omegab(cosmo: CosmoResults, fracstep: float = 0.002):
    cl_before = splev(cosmo.ell, cosmo.clTTEETE_minOmegab[0])
    cl_after = splev(cosmo.ell, cosmo.clTTEETE_plusOmegab[0])
    cl_centre = splev(cosmo.ell, cosmo.clTT)
    cl_before2 = splev(cosmo.ell, cosmo.clTTEETE_minOmegab2[0])
    cl_after2 = splev(cosmo.ell, cosmo.clTTEETE_plusOmegab2[0])

    CLs = np.zeros((5, len(cosmo.ell)))
    CLs[0, :] = cl_before2
    CLs[1, :] = cl_before
    CLs[2, :] = cl_centre
    CLs[3, :] = cl_after
    CLs[4, :] = cl_after2

    d_domegab = FinDiff(0, fracstep * cosmo.Omegab, acc=2)
    derCl_omegab = d_domegab(CLs)

    derCl_omegab_interpTT = interp1d(cosmo.ell, derCl_omegab[2], kind="cubic")

    cl_before = splev(cosmo.ell, cosmo.clTTEETE_minOmegab[1])
    cl_after = splev(cosmo.ell, cosmo.clTTEETE_plusOmegab[1])
    cl_centre = splev(cosmo.ell, cosmo.clEE)
    cl_before2 = splev(cosmo.ell, cosmo.clTTEETE_minOmegab2[1])
    cl_after2 = splev(cosmo.ell, cosmo.clTTEETE_plusOmegab2[1])
    CLs = np.zeros((5, len(cosmo.ell)))
    CLs[0, :] = cl_before2
    CLs[1, :] = cl_before
    CLs[2, :] = cl_centre
    CLs[3, :] = cl_after
    CLs[4, :] = cl_after2

    derCl_omegab_EE = d_domegab(CLs)
    derCl_omegab_interpEE = interp1d(cosmo.ell, derCl_omegab_EE[2], kind="cubic")

    cl_before = splev(cosmo.ell, cosmo.clTTEETE_minOmegab[2])
    cl_after = splev(cosmo.ell, cosmo.clTTEETE_plusOmegab[2])
    cl_centre = splev(cosmo.ell, cosmo.clTE)
    cl_before2 = splev(cosmo.ell, cosmo.clTTEETE_minOmegab2[2])
    cl_after2 = splev(cosmo.ell, cosmo.clTTEETE_plusOmegab2[2])
    CLs = np.zeros((5, len(cosmo.ell)))
    CLs[0, :] = cl_before2
    CLs[1, :] = cl_before
    CLs[2, :] = cl_centre
    CLs[3, :] = cl_after
    CLs[4, :] = cl_after2

    derCl_omegab_TE = d_domegab(CLs)
    derCl_omegab_interpTE = interp1d(cosmo.ell, derCl_omegab_TE[2], kind="cubic")

    return derCl_omegab_interpTT, derCl_omegab_interpEE, derCl_omegab_interpTE


def compute_deriv_omegacdm(cosmo: CosmoResults, fracstep: float = 0.002):
    cl_before = splev(cosmo.ell, cosmo.clTTEETE_minOmegacdm[0])
    cl_after = splev(cosmo.ell, cosmo.clTTEETE_plusOmegacdm[0])
    cl_centre = splev(cosmo.ell, cosmo.clTT)
    cl_before2 = splev(cosmo.ell, cosmo.clTTEETE_minOmegacdm2[0])
    cl_after2 = splev(cosmo.ell, cosmo.clTTEETE_plusOmegacdm2[0])

    CLs = np.zeros((5, len(cosmo.ell)))
    CLs[0, :] = cl_before2
    CLs[1, :] = cl_before
    CLs[2, :] = cl_centre
    CLs[3, :] = cl_after
    CLs[4, :] = cl_after2

    d_domegacdm = FinDiff(0, fracstep * cosmo.Omega_cdm, acc=2)
    derCl_omegacdm = d_domegacdm(CLs)

    derCl_omegacdm_interpTT = interp1d(cosmo.ell, derCl_omegacdm[2], kind="cubic")

    cl_before = splev(cosmo.ell, cosmo.clTTEETE_plusOmegacdm[1])
    cl_after = splev(cosmo.ell, cosmo.clTTEETE_minOmegacdm[1])
    cl_centre = splev(cosmo.ell, cosmo.clEE)
    cl_before2 = splev(cosmo.ell, cosmo.clTTEETE_minOmegacdm2[1])
    cl_after2 = splev(cosmo.ell, cosmo.clTTEETE_plusOmegacdm2[1])

    CLs = np.zeros((5, len(cosmo.ell)))
    CLs[0, :] = cl_before2
    CLs[1, :] = cl_before
    CLs[2, :] = cl_centre
    CLs[3, :] = cl_after
    CLs[4, :] = cl_after2

    derCl_omegacdm_EE = d_domegacdm(CLs)
    derCl_omegacdm_interpEE = interp1d(cosmo.ell, derCl_omegacdm_EE[2], kind="cubic")

    cl_before = splev(cosmo.ell, cosmo.clTTEETE_minOmegacdm[2])
    cl_after = splev(cosmo.ell, cosmo.clTTEETE_plusOmegacdm[2])
    cl_centre = splev(cosmo.ell, cosmo.clTE)
    cl_before2 = splev(cosmo.ell, cosmo.clTTEETE_minOmegacdm2[2])
    cl_after2 = splev(cosmo.ell, cosmo.clTTEETE_plusOmegacdm2[2])

    CLs = np.zeros((5, len(cosmo.ell)))
    CLs[0, :] = cl_before2
    CLs[1, :] = cl_before
    CLs[2, :] = cl_centre
    CLs[3, :] = cl_after
    CLs[4, :] = cl_after2

    derCl_omegacdm_TE = d_domegacdm(CLs)

    derCl_omegacdm_interpTE = interp1d(cosmo.ell, derCl_omegacdm_TE[2], kind="cubic")

    return derCl_omegacdm_interpTT, derCl_omegacdm_interpEE, derCl_omegacdm_interpTE


def compute_deriv_As(cosmo: CosmoResults, fracstep: float = 0.002):
    cl_before = splev(cosmo.ell, cosmo.clTTEETE_minAs[0])
    cl_after = splev(cosmo.ell, cosmo.clTTEETE_plusAs[0])
    cl_centre = splev(cosmo.ell, cosmo.clTT)
    cl_before2 = splev(cosmo.ell, cosmo.clTTEETE_minAs2[0])
    cl_after2 = splev(cosmo.ell, cosmo.clTTEETE_plusAs2[0])

    CLs = np.zeros((5, len(cosmo.ell)))
    CLs[0, :] = cl_before2
    CLs[1, :] = cl_before
    CLs[2, :] = cl_centre
    CLs[3, :] = cl_after
    CLs[4, :] = cl_after2

    d_dAs = FinDiff(0, fracstep * cosmo.lnAs10, acc=2)
    derCl_As = d_dAs(CLs)

    derCl_As_interpTT = interp1d(cosmo.ell, derCl_As[2], kind="cubic")

    cl_before = splev(cosmo.ell, cosmo.clTTEETE_minAs[1])
    cl_after = splev(cosmo.ell, cosmo.clTTEETE_plusAs[1])
    cl_centre = splev(cosmo.ell, cosmo.clEE)
    cl_before2 = splev(cosmo.ell, cosmo.clTTEETE_minAs2[1])
    cl_after2 = splev(cosmo.ell, cosmo.clTTEETE_plusAs2[1])

    CLs = np.zeros((5, len(cosmo.ell)))
    CLs[0, :] = cl_before2
    CLs[1, :] = cl_before
    CLs[2, :] = cl_centre
    CLs[3, :] = cl_after
    CLs[4, :] = cl_after2

    derCl_As_EE = d_dAs(CLs)
    derCl_As_interpEE = interp1d(cosmo.ell, derCl_As_EE[2], kind="cubic")

    cl_before = splev(cosmo.ell, cosmo.clTTEETE_minAs[2])
    cl_after = splev(cosmo.ell, cosmo.clTTEETE_plusAs[2])
    cl_centre = splev(cosmo.ell, cosmo.clTE)
    cl_before2 = splev(cosmo.ell, cosmo.clTTEETE_minAs2[2])
    cl_after2 = splev(cosmo.ell, cosmo.clTTEETE_plusAs2[2])

    CLs = np.zeros((5, len(cosmo.ell)))
    CLs[0, :] = cl_before2
    CLs[1, :] = cl_before
    CLs[2, :] = cl_centre
    CLs[3, :] = cl_after
    CLs[4, :] = cl_after2
    derCl_As_TE = d_dAs(CLs)

    derCl_As_interpTE = interp1d(cosmo.ell, derCl_As_TE[2], kind="cubic")

    return derCl_As_interpTT, derCl_As_interpEE, derCl_As_interpTE


def compute_deriv_ns(cosmo: CosmoResults, fracstep: float = 0.002):
    cl_before = splev(cosmo.ell, cosmo.clTTEETE_minns[0])
    cl_after = splev(cosmo.ell, cosmo.clTTEETE_plusns[0])
    cl_centre = splev(cosmo.ell, cosmo.clTT)
    cl_before2 = splev(cosmo.ell, cosmo.clTTEETE_minns2[0])
    cl_after2 = splev(cosmo.ell, cosmo.clTTEETE_plusns2[0])

    CLs = np.zeros((5, len(cosmo.ell)))
    CLs[0, :] = cl_before2
    CLs[1, :] = cl_before
    CLs[2, :] = cl_centre
    CLs[3, :] = cl_after
    CLs[4, :] = cl_after2

    d_dns = FinDiff(0, fracstep * cosmo.ns, acc=2)
    derCl_ns = d_dns(CLs)

    derCl_ns_interpTT = interp1d(cosmo.ell, derCl_ns[2], kind="cubic")

    cl_before = splev(cosmo.ell, cosmo.clTTEETE_minns[1])
    cl_after = splev(cosmo.ell, cosmo.clTTEETE_plusns[1])
    cl_centre = splev(cosmo.ell, cosmo.clEE)
    cl_before2 = splev(cosmo.ell, cosmo.clTTEETE_minns2[1])
    cl_after2 = splev(cosmo.ell, cosmo.clTTEETE_plusns2[1])

    CLs = np.zeros((5, len(cosmo.ell)))
    CLs[0, :] = cl_before2
    CLs[1, :] = cl_before
    CLs[2, :] = cl_centre
    CLs[3, :] = cl_after
    CLs[4, :] = cl_after2

    derCl_ns_EE = d_dns(CLs)

    derCl_ns_interpEE = interp1d(cosmo.ell, derCl_ns_EE[2], kind="cubic")

    cl_before = splev(cosmo.ell, cosmo.clTTEETE_minns[2])
    cl_after = splev(cosmo.ell, cosmo.clTTEETE_plusns[2])
    cl_centre = splev(cosmo.ell, cosmo.clTE)
    cl_before2 = splev(cosmo.ell, cosmo.clTTEETE_minns2[2])
    cl_after2 = splev(cosmo.ell, cosmo.clTTEETE_plusns2[2])

    CLs = np.zeros((5, len(cosmo.ell)))
    CLs[0, :] = cl_before2
    CLs[1, :] = cl_before
    CLs[2, :] = cl_centre
    CLs[3, :] = cl_after
    CLs[4, :] = cl_after2

    derCl_ns_TE = d_dns(CLs)

    derCl_ns_interpTE = interp1d(cosmo.ell, derCl_ns_TE[2], kind="cubic")

    return derCl_ns_interpTT, derCl_ns_interpEE, derCl_ns_interpTE


def compute_deriv_tau(cosmo: CosmoResults, fracstep: float = 0.002):
    cl_before = splev(cosmo.ell, cosmo.clTTEETE_mintau[0])
    cl_after = splev(cosmo.ell, cosmo.clTTEETE_plustau[0])
    cl_centre = splev(cosmo.ell, cosmo.clTT)
    cl_before2 = splev(cosmo.ell, cosmo.clTTEETE_mintau2[0])
    cl_after2 = splev(cosmo.ell, cosmo.clTTEETE_plustau2[0])

    CLs = np.zeros((5, len(cosmo.ell)))
    CLs[0, :] = cl_before2
    CLs[1, :] = cl_before
    CLs[2, :] = cl_centre
    CLs[3, :] = cl_after
    CLs[4, :] = cl_after2

    d_dtau = FinDiff(0, fracstep * cosmo.tau, acc=2)
    derCl_tau = d_dtau(CLs)

    derCl_tau_interpTT = interp1d(cosmo.ell, derCl_tau[2], kind="cubic")

    cl_before = splev(cosmo.ell, cosmo.clTTEETE_mintau[1])
    cl_after = splev(cosmo.ell, cosmo.clTTEETE_plustau[1])
    cl_centre = splev(cosmo.ell, cosmo.clEE)
    cl_before2 = splev(cosmo.ell, cosmo.clTTEETE_mintau2[1])
    cl_after2 = splev(cosmo.ell, cosmo.clTTEETE_plustau2[1])

    CLs = np.zeros((5, len(cosmo.ell)))
    CLs[0, :] = cl_before2
    CLs[1, :] = cl_before
    CLs[2, :] = cl_centre
    CLs[3, :] = cl_after
    CLs[4, :] = cl_after2

    derCl_tau_EE = d_dtau(CLs)

    derCl_tau_interpEE = interp1d(cosmo.ell, derCl_tau_EE[2], kind="cubic")

    cl_before = splev(cosmo.ell, cosmo.clTTEETE_mintau[2])
    cl_after = splev(cosmo.ell, cosmo.clTTEETE_plustau[2])
    cl_centre = splev(cosmo.ell, cosmo.clTE)
    cl_before2 = splev(cosmo.ell, cosmo.clTTEETE_mintau2[2])
    cl_after2 = splev(cosmo.ell, cosmo.clTTEETE_plustau2[2])

    CLs = np.zeros((5, len(cosmo.ell)))
    CLs[0, :] = cl_before2
    CLs[1, :] = cl_before
    CLs[2, :] = cl_centre
    CLs[3, :] = cl_after
    CLs[4, :] = cl_after2

    derCl_tau_TE = d_dtau(CLs)

    derCl_tau_interpTE = interp1d(cosmo.ell, derCl_tau_TE[2], kind="cubic")

    return derCl_tau_interpTT, derCl_tau_interpEE, derCl_tau_interpTE


def compute_deriv_phiamplitude(cosmo: CosmoResults, dl: float = 0.05):
    order = 4
    ClarrayTT = np.empty((2 * order + 1, len(cosmo.ell)))
    ClarrayEE = np.empty((2 * order + 1, len(cosmo.ell)))
    ClarrayTE = np.empty((2 * order + 1, len(cosmo.ell)))

    for i in range(-order, order + 1):
        linterp = cosmo.ell + i * dl
        ClarrayTT[i + order] = splev(linterp, cosmo.clTT, ext=1)
        ClarrayEE[i + order] = splev(linterp, cosmo.clEE, ext=1)
        ClarrayTE[i + order] = splev(linterp, cosmo.clTE, ext=1)

    derClTT = FinDiff(0, dl, acc=4)(ClarrayTT)[order]
    derClEE = FinDiff(0, dl, acc=4)(ClarrayEE)[order]
    derClTE = FinDiff(0, dl, acc=4)(ClarrayTE)[order]

    dl_dA = 1.0 * fitting_formula_Montefalcone2025(cosmo.ell)

    derClTT_A = derClTT * dl_dA
    derClEE_A = derClEE * dl_dA
    derClTE_A = derClTE * dl_dA
    derClTT_A = interp1d(cosmo.ell, derClTT_A, kind="cubic")
    derClEE_A = interp1d(cosmo.ell, derClEE_A, kind="cubic")
    derClTE_A = interp1d(cosmo.ell, derClTE_A, kind="cubic")
    return derClTT_A, derClEE_A, derClTE_A


# def compute_derive_geff(cosmo: CosmoResults):
#     from scipy.interpolate import RegularGridInterpolator

#     order = 4  # interpolating power spectrum at multiple different ks to get a precise derivative from findiff
#     nmu = 100
#     dk = 0.0001
#     mu = np.linspace(0.0, 1.0, nmu)

#     pkarray = np.empty((2 * order + 1, len(cosmo.k)))
#     for i in range(-order, order + 1):
#         kinterp = cosmo.k + i * dk

#         pkarray[i + order] = splev(kinterp, cosmo.pk[0]) / splev(
#             kinterp, cosmo.pksmooth[0]
#         )

#     derPk = FinDiff(0, dk, acc=4)(pkarray)[order]
#     dk_dgeff = derivk_geff(cosmo.k, cosmo.log10Geff, cosmo.r_d, cosmo.beta_phi)
#     derPgeff = np.outer(
#         derPk * dk_dgeff, np.ones(len(mu))
#     )  # dP(k')/dgeff = dP/dk' * dk'/dgeff
#     derPgeff_interp = [RegularGridInterpolator([cosmo.k, mu], derPgeff)]
#     return derPgeff_interp


def Fish(
    cosmo: CosmoResults,
    derClthetastar: list,
    derClbeta: list,
    derClOmegab: list,
    derClOmegacdm: list,
    derClAs: list,
    derClns: list,
    derCltau: list,
    derClgeff: list,
    geff_fixed: bool = True,
):
    """Computes the Fisher information on cosmological parameters theta_star, A_phi (phase shift amplitude due to standard model neutrinos,
    log10Geff).

    Parameters
    ----------
    cosmo: CosmoResults object
        An instance of the CosmoResults class.
    data: InputData object
        An instance of the InputData class.
    Returns
    -------
    ManyFish: np.ndarray

    """
    # Uses Simpson's rule or adaptive quadrature to integrate over all k and mu.
    # mu and k values for Simpson's rule
    lvec = np.arange(
        np.min([cosmo.lminTT, cosmo.lminTE, cosmo.lminEE]),
        np.max([cosmo.lmaxTT, cosmo.lmaxTE, cosmo.lmaxEE]),
    )

    # 2D integration
    ManyFish = simps(
        CastNet(
            lvec,
            cosmo,
            derClthetastar,
            derClbeta,
            derClOmegab,
            derClOmegacdm,
            derClAs,
            derClns,
            derCltau,
            derClgeff,
            geff_fixed,
        ),
        x=lvec,
        axis=2,
    )

    return ManyFish


def CastNet(
    ll: npt.NDArray,
    cosmo: CosmoResults,
    derClthetastar: list,
    derClA: list,
    derClOmegab: list,
    derClOmegacdm: list,
    derClAs: list,
    derClns: list,
    derCltau: list,
    derClgeff: list,
    geff_fixed: bool = True,
):
    """Compute the Fisher matrix for a vector of ll.

    Parameters
    ----------
    ll: np.ndarray
        The particular ll value(s) to consider
    data: InputData object
        An instance of the InputData class.
    cosmo: CosmoResults object
        An instance of the CosmoResults class.
    derCltheta_star: list
    derClA: list
    derClgeff: list
    geff_fixed: bool

    Returns
    -------
    Shoal: np.ndarray
        An array containing the Fisher information for each parameter of interest.
    """

    Shoal = np.empty((8, 8, len(ll)))
    if geff_fixed:
        Shoal = np.empty((7, 7, len(ll)))

    derClAval = np.array([derClA[0](ll), derClA[1](ll), derClA[2](ll)])
    derClgeffval = (
        np.array([derClgeff[0](ll), derClgeff[1](ll), derClgeff[2](ll)])
        if not geff_fixed
        else np.array([])
    )
    derClthetastarval = np.array(
        [derClthetastar[0](ll), derClthetastar[1](ll), derClthetastar[2](ll)]
    )
    derClOmegabval = np.array(
        [derClOmegab[0](ll), derClOmegab[1](ll), derClOmegab[2](ll)]
    )
    derClOmegacdmval = np.array(
        [derClOmegacdm[0](ll), derClOmegacdm[1](ll), derClOmegacdm[2](ll)]
    )
    derClAsval = np.array([derClAs[0](ll), derClAs[1](ll), derClAs[2](ll)])
    derClnsval = np.array([derClns[0](ll), derClns[1](ll), derClns[2](ll)])
    derCltauval = np.array([derCltau[0](ll), derCltau[1](ll), derCltau[2](ll)])

    Cl_arr = [cosmo.clTT, cosmo.clEE, cosmo.clTE]

    # Loop over each k and mu value and compute the Fisher information for the cosmological parameters
    for i, lval in enumerate(ll):
        derCl = np.array(
            [
                np.array(
                    [derClthetastarval[j][i] for j in range(len(derClthetastarval))]
                ),
                np.array([derClAval[j][i] for j in range(len(derClAval))]),
                np.array([derClOmegabval[j][i] for j in range(len(derClOmegabval))]),
                np.array(
                    [derClOmegacdmval[j][i] for j in range(len(derClOmegacdmval))]
                ),
                np.array([derClAsval[j][i] for j in range(len(derClAsval))]),
                np.array([derClnsval[j][i] for j in range(len(derClnsval))]),
                np.array([derCltauval[j][i] for j in range(len(derCltauval))]),
            ]
        )

        if not geff_fixed:
            derCl = np.vstack(
                (derCl, ([derClgeffval[0][i], derClgeffval[1][i], derClgeffval[2][i]]))
            )

        if lval < cosmo.lminTT or lval > cosmo.lmaxTT:
            derCl[:, 0] = 0.0
        if lval < cosmo.lminEE or lval > cosmo.lmaxEE:
            derCl[:, 1] = 0.0
        if lval < cosmo.lminTE or lval > cosmo.lmaxTE:
            derCl[:, 2] = 0.0

        covCl = compute_cov(
            np.array([splev(lval, Cl_arr[j]) for j in range(len(Cl_arr))]),
            lval,
        )
        covCl_inv = np.linalg.inv(covCl)

        if not cosmo.use_TE and not cosmo.use_EE:
            covCl = covCl[:1, :1]
            covCl_inv = np.linalg.inv(covCl)

        elif not cosmo.use_TE:
            covCl = covCl[:2, :2]
            covCl_inv = np.linalg.inv(covCl)

        elif not cosmo.use_EE:
            covCl = covCl[[0, 2], :][:, [0, 2]]
            covCl_inv = np.linalg.inv(covCl)

        for theta1 in range(derCl.shape[0]):
            for theta2 in range(derCl.shape[0]):
                derCltheta1 = derCl[theta1, :]
                derCltheta2 = derCl[theta2, :]

                indices = [0, 1, 2]
                if not cosmo.use_TE and not cosmo.use_EE:
                    indices = [0]
                    covCl = covCl[:1, :1]
                elif not cosmo.use_EE:
                    indices.remove(1)
                elif not cosmo.use_TE:
                    indices.remove(2)

                for index in indices:
                    for index2 in indices:
                        Shoal[theta1, theta2, i] += (
                            (2.0 * lval + 1.0)
                            * 0.5
                            * cosmo.area
                            / (4.0 * np.pi)
                            * derCltheta1[index]
                            * covCl_inv[index][index2]
                            * derCltheta2[index2]
                        )

    return Shoal


def compute_cov(cosmoClval: npt.NDArray, lval: float):
    """Computes the covariance matrix of the auto and cross-power spectra for a given
        ell, as well as its inverse.

    Returns
    -------
    covariance: np.ndarray
    cov_inv: np.ndarray
    """

    deltabT = np.array([33.0, 23.0, 14.0, 10.0, 7.0, 5.0, 5.0]) / 3437.75
    deltabE = np.array([14.0, 10.0, 7.0, 5.0, 5.0]) / 3437.75
    deltaT = np.array([145.0, 149.0, 137.0, 65.0, 43.0, 66.0, 200.0]) / 3437.75
    deltaE = np.array([450.0, 103.0, 81.0, 134.0, 406.0]) / 3437.75
    Nl_freqT = (
        1.0
        / (deltaT**2)
        * np.exp(-1.0 * lval * (lval + 1.0) * deltabT**2 / (8.0 * np.log(2.0)))
    )
    Nl_freqE = (
        1.0
        / (deltaE**2)
        * np.exp(-1.0 * lval * (lval + 1.0) * deltabE**2 / (8.0 * np.log(2.0)))
    )
    Nl_DeltaT = 1.0 / np.sum(Nl_freqT)
    Nl_DeltaE = 1.0 / np.sum(Nl_freqE)

    # Nl_DeltaT = 0.0
    # Nl_DeltaE = 0.0

    covariance = np.array(
        (
            np.array(
                [
                    (Nl_DeltaT + cosmoClval[0]) ** 2,
                    cosmoClval[1] ** 2,
                    (Nl_DeltaT + cosmoClval[0]) * cosmoClval[1],
                ]
            ),
            np.array(
                [
                    cosmoClval[1] ** 2,
                    (Nl_DeltaE + cosmoClval[2]) ** 2,
                    (Nl_DeltaE + cosmoClval[2]) * cosmoClval[1],
                ]
            ),
            np.array(
                [
                    (Nl_DeltaT + cosmoClval[0]) * cosmoClval[1],
                    (Nl_DeltaE + cosmoClval[2]) * cosmoClval[1],
                    0.5
                    * (
                        cosmoClval[1] ** 2
                        + (Nl_DeltaT + cosmoClval[0]) * (Nl_DeltaE + cosmoClval[2])
                    ),
                ]
            ),
        )
    )

    return covariance
