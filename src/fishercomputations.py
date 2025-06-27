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
    derClthetastar = compute_deriv(
        cosmo,
        cosmo.theta_star,
        [
            cosmo.clTTEETE_minthetastar2,
            cosmo.clTTEETE_minthetastar,
            cosmo.clTTEETE_plusthetastar,
            cosmo.clTTEETE_plusthetastar2,
        ],
        fracstep=fracstepthetastar,
    )
    derClAphi = compute_deriv_phiamplitude(cosmo)
    derClOmegab = compute_deriv(
        cosmo,
        cosmo.Omegab * 100.0,
        [
            cosmo.clTTEETE_minOmegab2,
            cosmo.clTTEETE_minOmegab,
            cosmo.clTTEETE_plusOmegab,
            cosmo.clTTEETE_plusOmegab2,
        ],
        fracstep=fracstepomegab,
    )
    derClOmegacdm = compute_deriv(
        cosmo,
        cosmo.Omega_cdm,
        [
            cosmo.clTTEETE_minOmegacdm2,
            cosmo.clTTEETE_minOmegacdm,
            cosmo.clTTEETE_plusOmegacdm,
            cosmo.clTTEETE_plusOmegacdm2,
        ],
        fracstep=fracstepomegacdm,
    )
    derClAs = compute_deriv(
        cosmo,
        cosmo.lnAs10,
        [
            cosmo.clTTEETE_minAs2,
            cosmo.clTTEETE_minAs,
            cosmo.clTTEETE_plusAs,
            cosmo.clTTEETE_plusAs2,
        ],
        fracstep=fracstepAs,
    )
    derClns = compute_deriv(
        cosmo,
        cosmo.ns,
        [
            cosmo.clTTEETE_minns2,
            cosmo.clTTEETE_minns,
            cosmo.clTTEETE_plusns,
            cosmo.clTTEETE_plusns2,
        ],
        fracstep=fracstepns,
    )
    derCltau = compute_deriv(
        cosmo,
        cosmo.tau,
        [
            cosmo.clTTEETE_mintau2,
            cosmo.clTTEETE_mintau,
            cosmo.clTTEETE_plustau,
            cosmo.clTTEETE_plustau2,
        ],
        fracstep=fracsteptau,
    )

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


def compute_deriv(
    cosmo: CosmoResults, paramcentre: float, spectra: list, fracstep: float = 0.002
):
    centre = [
        splev(cosmo.ell, cosmo.clTT),
        splev(cosmo.ell, cosmo.clEE),
        splev(cosmo.ell, cosmo.clTE),
    ]

    d_dthetastar = FinDiff(0, fracstep * paramcentre, acc=2)

    s1, s2, s3, s4 = spectra

    derivs = []

    for i in range(3):
        CLs = np.zeros((5, len(cosmo.ell)))
        CLs[0, :] = splev(cosmo.ell, s1[i])
        CLs[1, :] = splev(cosmo.ell, s2[i])
        CLs[2, :] = centre[i]
        CLs[3, :] = splev(cosmo.ell, s3[i])
        CLs[4, :] = splev(cosmo.ell, s4[i])

        derCl_thetastar = d_dthetastar(CLs)
        derivs.append(interp1d(cosmo.ell, derCl_thetastar[2], kind="cubic"))

    return derivs[0], derivs[1], derivs[2]


def compute_deriv_phiamplitude(cosmo: CosmoResults, dl: float = 0.05):
    order = 4
    ClarrayTT = np.zeros((2 * order + 1, len(cosmo.ell)))
    ClarrayEE = np.zeros((2 * order + 1, len(cosmo.ell)))
    ClarrayTE = np.zeros((2 * order + 1, len(cosmo.ell)))

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

    Shoal = np.zeros((8, 8, len(ll)))
    if geff_fixed:
        Shoal = np.zeros((7, 7, len(ll)))

    Cl_arr = [cosmo.clTT, cosmo.clEE, cosmo.clTE]

    # Loop over each k and mu value and compute the Fisher information for the cosmological parameters
    for i, lval in enumerate(ll):
        derCl = np.array(
            [
                np.array([derClthetastar[j](lval) for j in range(len(derClthetastar))]),
                np.array([derClA[j](lval) for j in range(len(derClA))]),
                np.array([derClOmegab[j](lval) for j in range(len(derClOmegab))]),
                np.array([derClOmegacdm[j](lval) for j in range(len(derClOmegacdm))]),
                np.array([derClAs[j](lval) for j in range(len(derClAs))]),
                np.array([derClns[j](lval) for j in range(len(derClns))]),
                np.array([derCltau[j](lval) for j in range(len(derCltau))]),
            ]
        )

        if not geff_fixed:
            derCl = np.vstack(
                (derCl, ([derClgeff[0](lval), derClgeff[1](lval), derClgeff[2](lval)]))
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

        indices = 3
        select_index = [0, 1, 2]
        if not cosmo.use_TE and not cosmo.use_EE:
            covCl = covCl[:1, :1]
            covCl_inv = np.linalg.inv(covCl)
            indices = 1
            select_index = [0]

        elif not cosmo.use_TE:
            covCl = covCl[:2, :2]
            covCl_inv = np.linalg.inv(covCl)
            indices = 2
            select_index = [0, 1]

        elif not cosmo.use_EE:
            covCl = covCl[[0, 2], :][:, [0, 2]]
            covCl_inv = np.linalg.inv(covCl)
            indices = 2
            select_index = [0, 2]

        for theta1 in range(derCl.shape[0]):
            for theta2 in range(derCl.shape[0]):
                derCltheta1 = derCl[theta1, select_index]
                derCltheta2 = derCl[theta2, select_index]

                for index in np.arange(indices):
                    for index2 in np.arange(indices):
                        val = (
                            (2.0 * lval + 1.0)
                            * 0.5
                            * cosmo.area
                            / (4.0 * np.pi)
                            * derCltheta1[index]
                            * covCl_inv[index][index2]
                            * derCltheta2[index2]
                        )

                        Shoal[theta1, theta2, i] += val

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

    covariance = covariance

    return covariance
