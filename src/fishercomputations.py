import numpy as np
from setup import CosmoResults, fitting_formula_Montefalcone2025
import numpy.typing as npt
from scipy.interpolate import CubicSpline
from setup import derivell_geff


def Set_Bait(
    cosmo: CosmoResults,
    geff_fixed: bool = True,
    neutrino_mass_fixed: bool = True,
    fracstepthetastar: float = 0.002,
    fracstepomegab: float = 0.002,
    fracstepomegacdm: float = 0.002,
    fracstepAs: float = 0.002,
    fracstepns: float = 0.002,
    fracsteptau: float = 0.002,
    fracstepmnu: float = 0.002,
):
    derClthetastar = compute_deriv(
        cosmo,
        cosmo.theta_star * 100.0,
        cosmo.clTTEETE_variations_thetastar,
        fracstep=fracstepthetastar,
    )
    derClAphi = compute_deriv_phiamplitude(cosmo)
    derClgeff = compute_derive_geff(cosmo)
    derClOmegab = compute_deriv(
        cosmo,
        cosmo.Omegab * 100.0,
        cosmo.clTTEETE_variations_omegab,
        fracstep=fracstepomegab,
    )
    derClOmegacdm = compute_deriv(
        cosmo,
        cosmo.Omega_cdm * 100.0,
        cosmo.clTTEETE_variations_omegacdm,
        fracstep=fracstepomegacdm,
    )
    derClAs = compute_deriv(
        cosmo,
        cosmo.lnAs10,
        cosmo.clTTEETE_variations_As,
        fracstep=fracstepAs,
    )
    derClns = compute_deriv(
        cosmo,
        cosmo.ns,
        cosmo.clTTEETE_variations_ns,
        fracstep=fracstepns,
    )
    derCltau = compute_deriv(
        cosmo,
        cosmo.tau,
        cosmo.clTTEETE_variations_tau,
        fracstep=fracsteptau,
    )

    derClmnu = []
    if not neutrino_mass_fixed:
        # If neutrino mass is not fixed, compute the derivative for mnu
        derClmnu = compute_deriv(
            cosmo,
            cosmo.mnu,
            cosmo.clTTEETE_variations_mnu,
            fracstep=fracstepmnu,
        )

    if geff_fixed and neutrino_mass_fixed:
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
        return (
            derClAphi,
            derClthetastar,
            derClOmegab,
            derClOmegacdm,
            derClAs,
            derClns,
            derCltau,
            derClgeff,
        )

    elif not neutrino_mass_fixed:
        return (
            derClAphi,
            derClthetastar,
            derClOmegab,
            derClOmegacdm,
            derClAs,
            derClns,
            derCltau,
            derClmnu,
        )


def compute_deriv(
    cosmo: CosmoResults, paramcentre: float, spectra: list, fracstep: float = 0.002
):
    centre = [
        cosmo.clTT,  # splev(cosmo.ell, cosmo.clTT),
        cosmo.clEE,  # splev(cosmo.ell, cosmo.clEE),
        cosmo.clTE,  # splev(cosmo.ell, cosmo.clTE),
    ]

    # d_dthetastar = FinDiff(0, fracstep * paramcentre, acc=2)
    step = fracstep * paramcentre

    s1, s2, s3, s4, s5, s6, s7, s8 = (
        spectra[0],
        spectra[1],
        spectra[2],
        spectra[3],
        spectra[4],
        spectra[5],
        spectra[6],
        spectra[7],
    )

    derivs = []

    for i in range(3):
        CLs = np.zeros((9, len(cosmo.ell)))
        CLs[0, :] = s1[i] * 1.0 / 280.0
        CLs[1, :] = s2[i] * -4.0 / 105.0
        CLs[2, :] = s3[i] * 1.0 / 5.0
        CLs[3, :] = s4[i] * -4.0 / 5.0
        CLs[4, :] = centre[i] * 0.0
        CLs[5, :] = s5[i] * 4.0 / 5.0
        CLs[6, :] = s6[i] * -1.0 / 5.0
        CLs[7, :] = s7[i] * 4.0 / 105.0
        CLs[8, :] = s8[i] * -1.0 / 280.0

        # derCl_thetastar = d_dthetastar(CLs)
        dfdx = np.sum(CLs, axis=0) / step

        derivs.append(dfdx)  # derCl_thetastar[4]))

    return derivs[0], derivs[1], derivs[2]


def compute_deriv_phiamplitude(cosmo: CosmoResults, dl: float = 0.1):
    order = 4
    ClarrayTT = np.zeros((2 * order + 1, len(cosmo.ell)))
    ClarrayEE = np.zeros((2 * order + 1, len(cosmo.ell)))
    ClarrayTE = np.zeros((2 * order + 1, len(cosmo.ell)))

    coefficients = [
        1.0 / 280.0,
        -4.0 / 105.0,
        1.0 / 5.0,
        -4.0 / 5.0,
        0.0,
        4.0 / 5.0,
        -1.0 / 5.0,
        4.0 / 105.0,
        -1.0 / 280.0,
    ]

    for i in range(-order, order + 1):
        linterp = cosmo.ell + i * dl
        ClarrayTT[i + order] = (
            CubicSpline(cosmo.ell, cosmo.clTT)(linterp) * coefficients[i + 4]
        )  # splev(linterp, cosmo.clTT, ext=1)
        ClarrayEE[i + order] = (
            CubicSpline(cosmo.ell, cosmo.clEE)(linterp) * coefficients[i + 4]
        )  # splev(linterp, cosmo.clEE, ext=1)
        ClarrayTE[i + order] = (
            CubicSpline(cosmo.ell, cosmo.clTE)(linterp) * coefficients[i + 4]
        )  # splev(linterp, cosmo.clTE, ext=1)

    # derClTT = FinDiff(0, dl, acc=6)(ClarrayTT)[order]
    # derClEE = FinDiff(0, dl, acc=6)(ClarrayEE)[order]
    # derClTE = FinDiff(0, dl, acc=6)(ClarrayTE)[order]

    derClTT = np.sum(ClarrayTT, axis=0) / dl
    derClEE = np.sum(ClarrayEE, axis=0) / dl
    derClTE = np.sum(ClarrayTE, axis=0) / dl

    dl_dA = 1.0 * fitting_formula_Montefalcone2025(cosmo.ell)

    derClTT_A = derClTT * dl_dA
    derClEE_A = derClEE * dl_dA
    derClTE_A = derClTE * dl_dA
    # derClTT_A = CubicSpline(cosmo.ell, derClTT_A)
    # derClEE_A = CubicSpline(cosmo.ell, derClEE_A)
    # derClTE_A = CubicSpline(cosmo.ell, derClTE_A)
    return derClTT_A, derClEE_A, derClTE_A


def compute_derive_geff(cosmo: CosmoResults, dl: float = 0.02):
    order = 4
    ClarrayTT = np.zeros((2 * order + 1, len(cosmo.ell)))
    ClarrayEE = np.zeros((2 * order + 1, len(cosmo.ell)))
    ClarrayTE = np.zeros((2 * order + 1, len(cosmo.ell)))

    coefficients = [
        1.0 / 280.0,
        -4.0 / 105.0,
        1.0 / 5.0,
        -4.0 / 5.0,
        0.0,
        4.0 / 5.0,
        -1.0 / 5.0,
        4.0 / 105.0,
        -1.0 / 280.0,
    ]

    for i in range(-order, order + 1):
        linterp = cosmo.ell + i * dl
        ClarrayTT[i + order] = (
            CubicSpline(cosmo.ell, cosmo.clTT)(linterp) * coefficients[i + 4]
        )  # splev(linterp, cosmo.clTT, ext=1)
        ClarrayEE[i + order] = (
            CubicSpline(cosmo.ell, cosmo.clEE)(linterp) * coefficients[i + 4]
        )  # splev(linterp, cosmo.clEE, ext=1)
        ClarrayTE[i + order] = (
            CubicSpline(cosmo.ell, cosmo.clTE)(linterp) * coefficients[i + 4]
        )  # splev(linterp, cosmo.clTE, ext=1)

    # derClTT = FinDiff(0, dl, acc=6)(ClarrayTT)[order]
    # derClEE = FinDiff(0, dl, acc=6)(ClarrayEE)[order]
    # derClTE = FinDiff(0, dl, acc=6)(ClarrayTE)[order]

    derClTT = np.sum(ClarrayTT, axis=0) / dl
    derClEE = np.sum(ClarrayEE, axis=0) / dl
    derClTE = np.sum(ClarrayTE, axis=0) / dl

    dll_dgeff = derivell_geff(cosmo.ell, cosmo.log10Geff, cosmo.theta_star, cosmo.A_phi)

    derClTT_geff = derClTT * dll_dgeff
    derClEE_geff = derClEE * dll_dgeff
    derClTE_geff = derClTE * dll_dgeff
    # derClTT_interp = CubicSpline(cosmo.ell, derClTT_geff)
    # derClEE_interp = CubicSpline(cosmo.ell, derClEE_geff)
    # derClTE_interp = CubicSpline(cosmo.ell, derClTE_geff)
    return derClTT_geff, derClEE_geff, derClTE_geff


def Fish(
    cosmo: CosmoResults,
    derClthetastar: list,
    derClA: list,
    derClOmegab: list,
    derClOmegacdm: list,
    derClAs: list,
    derClns: list,
    derCltau: list,
    derClgeff: list,
    derClmnu: list = None,
    geff_fixed: bool = True,
    mnu_fixed: bool = True,
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
    # lvec = np.arange(
    #     np.min([cosmo.lminTT, cosmo.lminTE, cosmo.lminEE]),
    #     np.max([cosmo.lmaxTT, cosmo.lmaxTE, cosmo.lmaxEE]) + 1,
    # )

    lvec = cosmo.ell

    # 2D integration
    ManyFish = np.sum(
        CastNet(
            lvec,
            cosmo,
            derClthetastar,
            derClA,
            derClOmegab,
            derClOmegacdm,
            derClAs,
            derClns,
            derCltau,
            derClgeff,
            derClmnu,
            geff_fixed,
            mnu_fixed,
        ),
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
    derClmnu: list,
    geff_fixed: bool = True,
    mnu_fixed: bool = True,
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
    if geff_fixed and mnu_fixed:
        Shoal = np.zeros((7, 7, len(ll)))

    Cl_arr = [cosmo.clTT, cosmo.clEE, cosmo.clTE]

    # Loop over each k and mu value and compute the Fisher information for the cosmological parameters
    for i, lval in enumerate(ll):
        derCl = np.array(
            [
                np.array([derClthetastar[j][i] for j in range(len(derClthetastar))]),
                np.array([derClA[j][i] for j in range(len(derClA))]),
                np.array([derClOmegab[j][i] for j in range(len(derClOmegab))]),
                np.array([derClOmegacdm[j][i] for j in range(len(derClOmegacdm))]),
                np.array([derClAs[j][i] for j in range(len(derClAs))]),
                np.array([derClns[j][i] for j in range(len(derClns))]),
                np.array([derCltau[j][i] for j in range(len(derCltau))]),
            ]
        )

        if not geff_fixed:
            derCl = np.vstack(
                (derCl, ([derClgeff[0][i], derClgeff[1][i], derClgeff[2][i]]))
            )
        if not mnu_fixed:
            derCl = np.vstack(
                (derCl, ([derClmnu[0][i], derClmnu[1][i], derClmnu[2][i]]))
            )

        if lval < cosmo.lminTT or lval > cosmo.lmaxTT:
            derCl[:, 0] = 0.0
        if lval < cosmo.lminEE or lval > cosmo.lmaxEE:
            derCl[:, 1] = 0.0
        if lval < cosmo.lminTE or lval > cosmo.lmaxTE:
            derCl[:, 2] = 0.0

        covCl = compute_cov(
            np.array([Cl_arr[j][i] for j in range(len(Cl_arr))]),
            lval,
            noise_Planck=cosmo.noise_Planck,
        )

        covCl = covCl * 2.0 * (1.0 / (2.0 * lval + 1.0))
        covCl_inv = np.linalg.inv(covCl)

        select_index = [0, 1, 2]
        if (not cosmo.use_TE and not cosmo.use_EE) or (
            covCl[1][1] <= 0.0 and covCl[2][2] <= 0.0
        ):
            covCl = covCl[:1, :1]
            covCl_inv = np.linalg.inv(covCl)
            select_index = [0]

        elif not cosmo.use_TE or covCl[2][2] <= 0.0:
            covCl = covCl[:2, :2]
            covCl_inv = np.linalg.inv(covCl)
            select_index = [0, 1]

        elif not cosmo.use_EE or covCl[1][1] <= 0.0:
            covCl = covCl[[0, 2], :][:, [0, 2]]
            covCl_inv = np.linalg.inv(covCl)
            select_index = [0, 2]

        for theta1 in range(derCl.shape[0]):
            for theta2 in range(derCl.shape[0]):
                derCltheta1 = derCl[theta1, select_index]
                derCltheta2 = derCl[theta2, select_index]

                val = derCltheta1 @ covCl_inv @ derCltheta2.T

                if val == 0.0:
                    continue

                else:
                    if theta1 < theta2:
                        Shoal[theta1, theta2, i] += val
                        Shoal[theta2, theta1, i] += val

                    if theta1 == theta2:
                        Shoal[theta1, theta2, i] += val

                    else:
                        continue

    return Shoal


def compute_cov(cosmoClval: npt.NDArray, lval: float, noise_Planck: bool = True):
    """Computes the covariance matrix of the auto and cross-power spectra for a given
        ell, as well as its inverse.

    Returns
    -------
    covariance: np.ndarray
    cov_inv: np.ndarray
    """

    deltab_f = np.array([9.65, 7.25, 4.99]) / 3437.75
    deltaT_f = np.array([2.5, 2.2, 4.8e-6]) * deltab_f
    deltaE_f = np.array([6.7, 4.0, 9.8e-6]) * deltab_f

    # deltabE = np.array([14.0, 10.0, 7.0, 5.0, 5.0]) / 3437.75
    # deltabT = np.array([33.0, 23.0, 14.0, 10.0, 7.0, 5.0, 5.0]) / 3437.75
    # deltaT = np.array([145.0, 149.0, 137.0, 65.0, 43.0, 66.0, 200.0]) / 3437.75
    # deltaE = np.array([450.0, 103.0, 81.0, 134.0, 406.0]) / 3437.75
    Nl_freqT = (
        1.0
        / (deltaT_f**2)
        * np.exp(-1.0 * lval * (lval + 1.0) * deltab_f**2 / (8.0 * np.log(2.0)))
    )
    Nl_freqE = (
        1.0
        / (deltaE_f**2)
        * np.exp(-1.0 * lval * (lval + 1.0) * deltab_f**2 / (8.0 * np.log(2.0)))
    )
    Nl_DeltaT = 1.0 / np.sum(Nl_freqT)
    Nl_DeltaE = 1.0 / np.sum(Nl_freqE)

    if not noise_Planck:
        Nl_DeltaT = 0.0
        Nl_DeltaE = 0.0

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
