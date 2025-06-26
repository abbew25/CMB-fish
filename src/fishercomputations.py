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
    cl_before = splev(cosmo.ell, cosmo.clTT_minthetastar)
    cl_after = splev(cosmo.ell, cosmo.clTT_plusthetastar)
    deltathetastar = fracstep * cosmo.theta_star
    derCl_thetastar = (cl_after - cl_before) / (2.0 * deltathetastar)  # * 1.0e-2)
    derCl_thetastar_interpTT = interp1d(cosmo.ell, derCl_thetastar)

    cl_before = splev(cosmo.ell, cosmo.clEE_minthetastar)
    cl_after = splev(cosmo.ell, cosmo.clEE_plusthetastar)
    derCl_thetastar_EE = (cl_after - cl_before) / (2.0 * deltathetastar)  # * 1.0e-2)
    derCl_thetastar_interpEE = interp1d(cosmo.ell, derCl_thetastar_EE)

    cl_before = splev(cosmo.ell, cosmo.clTE_minthetastar)
    cl_after = splev(cosmo.ell, cosmo.clTE_plusthetastar)
    derCl_thetastar_TE = (cl_after - cl_before) / (2.0 * deltathetastar)  # * 1.0e-2)
    derCl_thetastar_interpTE = interp1d(cosmo.ell, derCl_thetastar_TE)

    return derCl_thetastar_interpTT, derCl_thetastar_interpEE, derCl_thetastar_interpTE


def compute_deriv_omegab(cosmo: CosmoResults, fracstep: float = 0.002):
    cl_before = splev(cosmo.ell, cosmo.clTT_minOmegab)
    cl_after = splev(cosmo.ell, cosmo.clTT_plusOmegab)
    deltaomegab = fracstep * cosmo.Omegab
    derCl_omegab = (cl_after - cl_before) / (2.0 * deltaomegab)
    derCl_omegab_interpTT = interp1d(cosmo.ell, derCl_omegab)

    cl_before = splev(cosmo.ell, cosmo.clEE_minOmegab)
    cl_after = splev(cosmo.ell, cosmo.clEE_plusOmegab)
    derCl_omegab_EE = (cl_after - cl_before) / (2.0 * deltaomegab)
    derCl_omegab_interpEE = interp1d(cosmo.ell, derCl_omegab_EE)

    cl_before = splev(cosmo.ell, cosmo.clTE_minOmegab)
    cl_after = splev(cosmo.ell, cosmo.clTE_plusOmegab)
    derCl_omegab_TE = (cl_after - cl_before) / (2.0 * deltaomegab)
    derCl_omegab_interpTE = interp1d(cosmo.ell, derCl_omegab_TE)

    return derCl_omegab_interpTT, derCl_omegab_interpEE, derCl_omegab_interpTE


def compute_deriv_omegacdm(cosmo: CosmoResults, fracstep: float = 0.002):
    cl_before = splev(cosmo.ell, cosmo.clTT_minOmegacdm)
    cl_after = splev(cosmo.ell, cosmo.clTT_plusOmegacdm)
    deltaomegacdm = fracstep * cosmo.Omega_cdm
    derCl_omegacdm = (cl_after - cl_before) / (2.0 * deltaomegacdm)
    derCl_omegacdm_interpTT = interp1d(cosmo.ell, derCl_omegacdm)

    cl_before = splev(cosmo.ell, cosmo.clEE_minOmegacdm)
    cl_after = splev(cosmo.ell, cosmo.clEE_plusOmegacdm)
    derCl_omegacdm_EE = (cl_after - cl_before) / (2.0 * deltaomegacdm)
    derCl_omegacdm_interpEE = interp1d(cosmo.ell, derCl_omegacdm_EE)

    cl_before = splev(cosmo.ell, cosmo.clTE_minOmegacdm)
    cl_after = splev(cosmo.ell, cosmo.clTE_plusOmegacdm)
    derCl_omegacdm_TE = (cl_after - cl_before) / (2.0 * deltaomegacdm)
    derCl_omegacdm_interpTE = interp1d(cosmo.ell, derCl_omegacdm_TE)

    return derCl_omegacdm_interpTT, derCl_omegacdm_interpEE, derCl_omegacdm_interpTE


def compute_deriv_As(cosmo: CosmoResults, fracstep: float = 0.002):
    cl_before = splev(cosmo.ell, cosmo.clTT_minAs)
    cl_after = splev(cosmo.ell, cosmo.clTT_plusAs)
    deltaAs = fracstep * cosmo.lnAs10
    derCl_As = (cl_after - cl_before) / (2.0 * deltaAs)
    derCl_As_interpTT = interp1d(cosmo.ell, derCl_As)

    cl_before = splev(cosmo.ell, cosmo.clEE_minAs)
    cl_after = splev(cosmo.ell, cosmo.clEE_plusAs)
    derCl_As_EE = (cl_after - cl_before) / (2.0 * deltaAs)
    derCl_As_interpEE = interp1d(cosmo.ell, derCl_As_EE)

    cl_before = splev(cosmo.ell, cosmo.clTE_minAs)
    cl_after = splev(cosmo.ell, cosmo.clTE_plusAs)
    derCl_As_TE = (cl_after - cl_before) / (2.0 * deltaAs)
    derCl_As_interpTE = interp1d(cosmo.ell, derCl_As_TE)

    return derCl_As_interpTT, derCl_As_interpEE, derCl_As_interpTE


def compute_deriv_ns(cosmo: CosmoResults, fracstep: float = 0.002):
    cl_before = splev(cosmo.ell, cosmo.clTT_minns)
    cl_after = splev(cosmo.ell, cosmo.clTT_plusns)
    deltans = fracstep * cosmo.ns
    derCl_ns = (cl_after - cl_before) / (2.0 * deltans)
    derCl_ns_interpTT = interp1d(cosmo.ell, derCl_ns)

    cl_before = splev(cosmo.ell, cosmo.clEE_minns)
    cl_after = splev(cosmo.ell, cosmo.clEE_plusns)
    derCl_ns_EE = (cl_after - cl_before) / (2.0 * deltans)
    derCl_ns_interpEE = interp1d(cosmo.ell, derCl_ns_EE)

    cl_before = splev(cosmo.ell, cosmo.clTE_minns)
    cl_after = splev(cosmo.ell, cosmo.clTE_plusns)
    derCl_ns_TE = (cl_after - cl_before) / (2.0 * deltans)
    derCl_ns_interpTE = interp1d(cosmo.ell, derCl_ns_TE)

    return derCl_ns_interpTT, derCl_ns_interpEE, derCl_ns_interpTE


def compute_deriv_tau(cosmo: CosmoResults, fracstep: float = 0.002):
    cl_before = splev(cosmo.ell, cosmo.clTT_mintau)
    cl_after = splev(cosmo.ell, cosmo.clTT_plustau)
    deltatau = fracstep * cosmo.tau
    derCl_tau = (cl_after - cl_before) / (2.0 * deltatau)
    derCl_tau_interpTT = interp1d(cosmo.ell, derCl_tau)

    cl_before = splev(cosmo.ell, cosmo.clEE_mintau)
    cl_after = splev(cosmo.ell, cosmo.clEE_plustau)
    derCl_tau_EE = (cl_after - cl_before) / (2.0 * deltatau)
    derCl_tau_interpEE = interp1d(cosmo.ell, derCl_tau_EE)

    cl_before = splev(cosmo.ell, cosmo.clTE_mintau)
    cl_after = splev(cosmo.ell, cosmo.clTE_plustau)
    derCl_tau_TE = (cl_after - cl_before) / (2.0 * deltatau)
    derCl_tau_interpTE = interp1d(cosmo.ell, derCl_tau_TE)

    return derCl_tau_interpTT, derCl_tau_interpEE, derCl_tau_interpTE


def compute_deriv_phiamplitude(cosmo: CosmoResults, dl: float = 0.01):
    clTT = splev(cosmo.ell, cosmo.clTT)
    clEE = splev(cosmo.ell, cosmo.clEE)
    clTE = splev(cosmo.ell, cosmo.clTE)
    derClTT = FinDiff(0, dl, acc=4)(clTT)
    derClEE = FinDiff(0, dl, acc=4)(clEE)
    derClTE = FinDiff(0, dl, acc=4)(clTE)

    order = 4
    ClarrayTT = np.empty((2 * order + 1, len(cosmo.ell)))
    ClarrayEE = np.empty((2 * order + 1, len(cosmo.ell)))
    ClarrayTE = np.empty((2 * order + 1, len(cosmo.ell)))

    for i in range(-order, order + 1):
        linterp = cosmo.ell + i * dl
        ClarrayTT[i + order] = splev(linterp, cosmo.clTT)
        ClarrayEE[i + order] = splev(linterp, cosmo.clEE)
        ClarrayTE[i + order] = splev(linterp, cosmo.clTE)

    derClTT = FinDiff(0, dl, acc=4)(ClarrayTT)[order]
    derClEE = FinDiff(0, dl, acc=4)(ClarrayEE)[order]
    derClTE = FinDiff(0, dl, acc=4)(ClarrayTE)[order]

    dl_dA = (
        1.0 * fitting_formula_Montefalcone2025(cosmo.ell)  #  / cosmo.theta_star * 100.0
    )

    derClTT_A = derClTT * dl_dA
    derClEE_A = derClEE * dl_dA
    derClTE_A = derClTE * dl_dA
    derClTT_A = interp1d(cosmo.ell, derClTT_A)
    derClEE_A = interp1d(cosmo.ell, derClEE_A)
    derClTE_A = interp1d(cosmo.ell, derClTE_A)
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

        covCl, covCl_inv = compute_inv_cov(
            np.array([splev(lval, Cl_arr[j]) for j in range(len(Cl_arr))]),
            lval,
            use_TE=cosmo.use_TE,
            use_EE=cosmo.use_EE,
        )

        for theta1 in range(derCl.shape[0]):
            for theta2 in range(derCl.shape[0]):
                derCltheta1 = derCl[theta1, :]
                derCltheta2 = derCl[theta2, :]

                if not cosmo.use_EE and not cosmo.use_TE:
                    Shoal[theta1, theta2, i] += (
                        (2.0 * lval + 1.0)
                        * 0.5
                        * cosmo.area
                        / (4.0 * np.pi)
                        * derCltheta1[0]
                        * covCl_inv
                        * derCltheta2[0]
                    )

                elif not cosmo.use_EE:
                    Shoal[theta1, theta2, i] += (
                        (2.0 * lval + 1.0)
                        * 0.5
                        * cosmo.area
                        / (4.0 * np.pi)
                        * derCltheta1[[0, 2]]
                        @ covCl_inv
                        @ derCltheta2[[0, 2]]
                    )

                elif not cosmo.use_TE:
                    Shoal[theta1, theta2, i] += (
                        (2.0 * lval + 1.0)
                        * 0.5
                        * cosmo.area
                        / (4.0 * np.pi)
                        * derCltheta1[:2]
                        @ covCl_inv
                        @ derCltheta2[:2]
                    )

                else:
                    Shoal[theta1, theta2, i] += (
                        (2.0 * lval + 1.0)
                        * 0.5
                        * cosmo.area
                        / (4.0 * np.pi)
                        * derCltheta1
                        @ covCl_inv
                        @ derCltheta2
                    )

                # derCovtheta1 = compute_deriv_cov(
                #     np.array([splev(lval, Cl_arr[j]) for j in range(len(Cl_arr))]),
                #     derCltheta1,
                #     use_TE=cosmo.use_TE,
                #     use_EE=cosmo.use_EE,
                # )[0]

                # derCovtheta2 = compute_deriv_cov(
                #     np.array([splev(lval, Cl_arr[j]) for j in range(len(Cl_arr))]),
                #     derCltheta2,
                #     use_TE=cosmo.use_TE,
                #     use_EE=cosmo.use_EE,
                # )[0]

    return Shoal


# def compute_deriv_cov(
#     cosmoClval: npt.NDArray, derivcosmoClval: npt.NDArray, use_TE: bool = True, use_EE: bool = True):

#     derivcovariance = np.array(
#         (
#             np.array(
#                 [
#                     2.0 * (cosmoClval[0]) * derivcosmoClval[0],
#                     2.0 * cosmoClval[1] * derivcosmoClval[1],
#                     derivcosmoClval[1] * cosmoClval[0] + derivcosmoClval[0] * cosmoClval[1],
#                 ]
#             ),
#             np.array(
#                 [
#                     2.0 * cosmoClval[1] * derivcosmoClval[1],
#                     2.0 * cosmoClval[2] * derivcosmoClval[2],
#                     derivcosmoClval[1] * cosmoClval[2] + derivcosmoClval[2] * cosmoClval[1],
#                 ]
#             ),
#             np.array(
#                 [
#                     cosmoClval[0] * derivcosmoClval[1] +  cosmoClval[1] * derivcosmoClval[0],
#                     derivcosmoClval[1] * cosmoClval[2] + derivcosmoClval[2] * cosmoClval[1],
#                     cosmoClval[1] * derivcosmoClval[1]  +  0.5 * cosmoClval[2] * derivcosmoClval[0] + 0.5 * cosmoClval[0] * derivcosmoClval[2],
#                 ]
#             ),
#         )
#     )

#     if not use_TE and not use_EE:
#         derivcovariance = derivcovariance[:1, :1]
#     elif not use_EE:
#         derivcovariance = derivcovariance[[0, 2], :][:, [0, 2]]
#     elif not use_TE:
#         derivcovariance = derivcovariance[:2, :2]

#     return derivcovariance


def compute_inv_cov(
    cosmoClval: npt.NDArray, lval: float, use_TE: bool = True, use_EE: bool = True
):
    """Computes the covariance matrix of the auto and cross-power spectra for a given
        ell, as well as its inverse.

    Returns
    -------
    covariance: np.ndarray
    cov_inv: np.ndarray
    """

    deltabT = np.array([33.0, 23.0, 14.0, 10.0, 7.0, 5.0, 5.0]) / 3437.75
    deltabE = np.array([14, 10, 7, 5, 5]) / 3437.75
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
                    0.5 * (cosmoClval[1] ** 2 + cosmoClval[2] * cosmoClval[0]),
                ]
            ),
        )
    )

    if not use_TE and not use_EE:
        covariance = covariance[:1, :1]
    elif not use_EE:
        covariance = covariance[[0, 2], :][:, [0, 2]]
    elif not use_TE:
        covariance = covariance[:2, :2]

    cov_inv = np.linalg.inv(covariance)

    return covariance, cov_inv
