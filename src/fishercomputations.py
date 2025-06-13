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
):
    derPthetastar = compute_deriv_thetastar(cosmo)
    derPbetaphi = compute_deriv_betaphiamplitude(cosmo)

    if geff_fixed:
        return derPbetaphi, derPthetastar
    elif not geff_fixed:
        return None


def compute_deriv_thetastar(cosmo: CosmoResults):
    cl_before = splev(cosmo.ell, cosmo.clTT_minthetastar)
    cl_after = splev(cosmo.ell, cosmo.clTT_plusthetastar)
    deltathetastar = 0.05 * cosmo.theta_star
    derCl_thetastar = (cl_before - cl_after) / (2.0 * deltathetastar / 100.0)
    derCl_thetastar_interp = interp1d(cosmo.ell, derCl_thetastar)
    return derCl_thetastar_interp


def compute_deriv_betaphiamplitude(cosmo: CosmoResults):
    dl = 0.001
    clTT = splev(cosmo.ell, cosmo.clTT)
    derCl = FinDiff(0, dl, acc=4)(clTT)
    order = 4
    Clarray = np.empty((2 * order + 1, len(cosmo.ell)))
    for i in range(-order, order + 1):
        linterp = cosmo.ell + i * dl
        Clarray[i + order] = splev(linterp, cosmo.clTT)
    derCl = FinDiff(0, dl, acc=4)(Clarray)[order]
    dl_dA = fitting_formula_Montefalcone2025(cosmo.ell) / cosmo.theta_star / 100.0

    derCl_A = derCl * dl_dA
    derCl_A = interp1d(cosmo.ell, derCl_A)
    return derCl_A


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
    lmin: float,
    lmax: float,
    derClthetastar: interp1d,
    derClbeta: interp1d,
    derClgeff: interp1d,
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
    lvec = np.arange(lmin, lmax)

    # 2D integration
    ManyFish = simps(
        CastNet(
            lvec,
            cosmo,
            derClthetastar,
            derClbeta,
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
    derClthetastar: interp1d,
    derClA: interp1d,
    derClgeff: interp1d,
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

    Shoal = np.empty((3, 3, len(ll)))
    if geff_fixed:
        Shoal = np.empty((2, 2, len(ll)))

    derClAval = derClA(ll)
    derClgeffval = derClgeff(ll) if not geff_fixed else []
    derClthetastarval = derClthetastar(ll)

    # Loop over each k and mu value and compute the Fisher information for the cosmological parameters
    for i, lval in enumerate(ll):
        derCl = np.array([derClthetastarval[i], derClAval[i]])
        if not geff_fixed:
            derCl.append(derClgeffval[i])

        covCl, covCl_inv = compute_inv_cov(
            splev(lval, cosmo.clTT),
            lval,
        )

        Shoal[:, :, i] = (
            (2.0 * lval + 1.0)
            * cosmo.area
            / (4.0 * np.pi)
            * np.outer(derCl * covCl_inv, derCl)
        )

    return Shoal


def compute_inv_cov(cosmoClval: float, lval: float):
    """Computes the covariance matrix of the auto and cross-power spectra for a given
        ell, as well as its inverse.

    Returns
    -------
    covariance: np.ndarray
    cov_inv: np.ndarray
    """

    deltab = np.array([33, 23, 14, 10, 7, 5, 5]) / 3437.75
    deltaT = np.array([145, 149, 137, 65, 43, 66, 200]) * 1.0e6 / 3437.75
    Nl_freq = deltaT**2 * np.exp(lval * (lval + 1.0) * deltab**2 / (8.0 * np.log(2.0)))
    Nl_DeltaT = 1.0 / np.sum(
        1.0 / Nl_freq
    )  # 1/Nl_freq is the variance of the noise in the power spectrum
    # if not geff_fixed:
    #     covariance = np.empty((3, 3))
    # Loop over power spectra of different samples P_12
    # for ps1, pair1 in enumerate(combinations_with_replacement(range(npop), 2)):
    #     n1, n2 = pair1
    #     # Loop over power spectra of different samples P_34
    #     for ps2, pair2 in enumerate(combinations_with_replacement(range(npop), 2)):
    #         n3, n4 = pair2
    #         # Cov(P_12,P_34)
    #         pk13, pk24 = kaiser[n1] * kaiser[n3] * pk, kaiser[n2] * kaiser[n4] * pk
    #         pk14, pk23 = kaiser[n1] * kaiser[n4] * pk, kaiser[n2] * kaiser[n3] * pk
    #         if n1 == n3:
    #             pk13 += 1.0 / nbar[n1]
    #         if n1 == n4:
    #             pk14 += 1.0 / nbar[n1]
    #         if n2 == n3:
    #             pk23 += 1.0 / nbar[n2]
    #         if n2 == n4:
    #             pk24 += 1.0 / nbar[n2]
    #         covariance[ps1, ps2] = pk13 * pk24 + pk14 * pk23

    covariance = 2.0 * (Nl_DeltaT + cosmoClval) ** 2

    cov_inv = 1.0 / covariance

    return covariance, cov_inv


# def shrink_sqr_matrix(sqr_matrix_obj: npt.NDArray, flags: npt.NDArray = np.array([])):
#     """
#     Function that removed the rows and columns of a square matrix (numpy matrix) if the rows
#     and columns that a diagonal element of the matrix coincides with is zero.
#     e.g. 1 2 3 4
#          2 1 9 0   ----- >     1 2 4
#          4 5 0 9               2 1 0
#          4 3 2 1               4 3 1

#     The third row and column has been removed since M_(2, 2) <= 1e-7
#     """
#     a = 0
#     new_obj = sqr_matrix_obj.copy()

#     if len(flags) >= 1:
#         new_obj = np.delete(new_obj, flags, 0)
#         new_obj = np.delete(new_obj, flags, 1)

#     else:
#         for i in (np.arange(sqr_matrix_obj.shape[0]))[::-1]:
#             if abs(sqr_matrix_obj[i][i]) <= 1e-13:
#                 a = i
#                 new_obj = np.delete(new_obj, a, 0)
#                 new_obj = np.delete(new_obj, a, 1)

#     return new_obj


# def compute_full_deriv(
#     npop: int,
#     npk: int,
#     kaiser: npt.NDArray,
#     pk: float,
#     pksmooth: float,
#     mu: float,
#     derPalpha: list,
#     derPbeta: list,
#     derPgeff: list,
#     f: float,
#     sigma8: float,
#     BAO_only: bool,
#     beta_phi_fixed: bool = True,
#     geff_fixed: bool = True,
# ):
#     """Computes the derivatives of the power spectrum as a function of
#         biases*sigma8, fsigma8, alpha_perp and alpha_par (in that order)
#         at a given k, mu and redshift

#     Parameters
#     ----------
#     npop: int
#         The number of different galaxy populations to consider. This is the number of different bias
#         parameters we need to take the derivatives with respect to.
#     npk: int
#         The number of different auto and cross power spectra to take to derivative of.
#         Equivalent to npop*(npop+1)/2, but passed in to avoid recomputing for each k/mu value.
#     kaiser: np.ndarray
#         The kaiser factors for each galaxy population at a fixed mu and redshift. Has length npop.
#     pk: float
#         The power spectrum value at the given k, mu and redshift values.
#     pksmooth: float
#         The smoothed power spectrum value at the given k, mu and redshift values.
#     mu: float
#         The mu value for the current call.
#     derPalpha: list
#         The precomputed derivatives of dP(k')/dalpha_perp and dP(k')/dalpha_par at the specific
#         value of k, mu and redshift. Contains 2 values, the first is the derivative w.r.t. alpha_perp,
#         the second is the derivative w.r.t. alpha_par.
#     f: float
#         The growth rate of structure at the current redshift.
#     sigma8: float
#         The value of sigma8 at the current redshift.
#     BAO_only: logical
#         If True compute derivatives w.r.t. to alpha_perp and alpha_par using only the BAO feature in the
#         power spectra. Otherwise use the full power spectrum and the kaiser factor. The former matches a standard
#         BAO analysis, the latter is more akin to a 'full-shape' analysis.

#     Returns
#     -------
#     derP: np.ndarray
#         The derivatives of all the auto and cross power spectra w.r.t. biases*sigma8, fsigma8, alpha_perp and alpha_par.
#         A 2D array where the first dimension corresponds to whichever parameter the derivative is w.r.t. in the following
#         order [b_0*sigma8 ... b_npop*sigma8], fsigma8, alpha_perp, alpha_par. The second dimension corresponds to the auto
#         or cross-power spectrum under consideration in the order P_00 , P_01, ... , P_0npop, P_11, P_1npop, ..., P_npopnpop.
#         The power spectrum order matches the covariance matrix order to allow for easy multiplication.
#     """

#     derP = np.zeros((npop + 3, npk))
#     if beta_phi_fixed and geff_fixed:
#         derP = np.zeros((npop + 3, npk))
#     elif not beta_phi_fixed and not geff_fixed:
#         derP = np.zeros((npop + 5, npk))
#     else:
#         derP = np.zeros((npop + 4, npk))

#     # Derivatives of all power spectra w.r.t to the bsigma8 of each population
#     for i in range(npop):
#         derP[i, int(i * (npop + (1 - i) / 2))] = 2.0 * kaiser[i] * pk / sigma8
#         derP[i, int(i * (npop + (1 - i) / 2)) + 1 : int((i + 1) * (npop - i / 2))] = (
#             kaiser[i + 1 :] * pk / sigma8
#         )
#         for j in range(0, i):
#             derP[i, i + int(j * (npop - (1 + j) / 2))] = kaiser[j] * pk / sigma8

#     # Derivatives of all power spectra w.r.t fsigma8
#     derP[npop, :] = [
#         (kaiser[i] + kaiser[j]) * mu**2 * pk / sigma8
#         for i in range(npop)
#         for j in range(i, npop)
#     ]

#     if not beta_phi_fixed and geff_fixed:
#         # Derivative of beta_phi amplitude w.r.t. alpha_perp and alpha_par
#         derP[npop + 3, :] = [
#             kaiser[i] * kaiser[j] * derPbeta[0] * pksmooth
#             for i in range(npop)
#             for j in range(i, npop)
#         ]

#     if not geff_fixed and beta_phi_fixed:
#         # Derivative of geff amplitude w.r.t. alpha_perp and alpha_par
#         derP[npop + 3, :] = [
#             kaiser[i] * kaiser[j] * derPgeff[0] * pksmooth
#             for i in range(npop)
#             for j in range(i, npop)
#         ]
#     if not beta_phi_fixed and not geff_fixed:
#         # Derivative of beta_phi amplitude w.r.t. alpha_perp and alpha_par
#         derP[npop + 3, :] = [
#             kaiser[i] * kaiser[j] * derPbeta[0] * pksmooth
#             for i in range(npop)
#             for j in range(i, npop)
#         ]
#         # Derivative of geff amplitude w.r.t. alpha_perp and alpha_par
#         derP[npop + 4, :] = [
#             kaiser[i] * kaiser[j] * derPgeff[0] * pksmooth
#             for i in range(npop)
#             for j in range(i, npop)
#         ]

#     # Derivatives of all power spectra w.r.t the alphas centred on alpha_per = alpha_par = 1.0
#     if BAO_only:
#         # For BAO_only we only include information on the alpha parameters
#         # from the BAO wiggles, and not the Kaiser factor
#         derP[npop + 1, :] = [
#             kaiser[i] * kaiser[j] * derPalpha[0] * pksmooth
#             for i in range(npop)
#             for j in range(i, npop)
#         ]
#         derP[npop + 2, :] = [
#             kaiser[i] * kaiser[j] * derPalpha[1] * pksmooth
#             for i in range(npop)
#             for j in range(i, npop)
#         ]

#     else:
#         # Derivative of mu'**2 w.r.t alpha_perp. Derivative w.r.t. alpha_par is -dmudalpha
#         dmudalpha = 2.0 * mu**2 * (1.0 - mu**2)

#         # We then just need use to the product rule as we already precomputed dP(k')/dalpha
#         derP[npop + 1, :] = [
#             (kaiser[i] + kaiser[j]) * f * pk * dmudalpha
#             + kaiser[i] * kaiser[j] * derPalpha[0]
#             for i in range(npop)
#             for j in range(i, npop)
#         ]
#         derP[npop + 2, :] = [
#             -(kaiser[i] + kaiser[j]) * f * pk * dmudalpha
#             + kaiser[i] * kaiser[j] * derPalpha[1]
#             for i in range(npop)
#             for j in range(i, npop)
#         ]

#     return derP
