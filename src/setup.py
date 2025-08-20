import copy
import numpy as np
import numpy.typing as npt
from configobj import ConfigObj


# This class contains everything we might need to set up to compute the fisher matrix
class CosmoResults:
    def __init__(
        self,
        pardict: ConfigObj,
        fracstepthetastar: float = 0.002,
        fracstepomegab: float = 0.002,
        fracstepomegacdm: float = 0.002,
        fracstepAs: float = 0.002,
        fracstepns: float = 0.002,
        fracsteptau: float = 0.002,
        fracstepmnu: float = 0.002,
    ):
        (
            self.ell,
            self.clTT,
            self.clEE,
            self.clTE,
            self.clBB,
            self.theta_star,
            self.Omegab,
            self.Omega_cdm,
            self.lnAs10,
            self.ns,
            self.tau,
            self.A_phi,
            self.log10Geff,
            self.area,
            self.mnu,
        ) = self.run_class_CMB(pardict)
        # ) = self.run_camb(pardict)

        self.lminTT = int(pardict["lminTT"])
        self.lmaxTT = int(pardict["lmaxTT"])
        self.lminTE = int(pardict["lminTE"])
        self.lmaxTE = int(pardict["lmaxTE"])
        self.lminEE = int(pardict["lminEE"])
        self.lmaxEE = int(pardict["lmaxEE"])
        self.lminBB = int(pardict["lminBB"])
        self.lmaxBB = int(pardict["lmaxBB"])
        self.use_TE = True if pardict["use_TE"] == "True" else False
        self.use_EE = True if pardict["use_EE"] == "True" else False
        self.use_BB = True if pardict["use_BB"] == "True" else False
        self.noise_Planck = True if pardict["noise_Planck"] == "True" else False
        self.noise_Planck2 = True if pardict["noise_Planck2"] == "True" else False

        # for derivatives w.r.t. thetastar
        self.clTTEETE_variations_thetastar = []
        self.clTTEETE_variations_omegab = []
        self.clTTEETE_variations_omegacdm = []
        self.clTTEETE_variations_As = []
        self.clTTEETE_variations_ns = []
        self.clTTEETE_variations_tau = []
        self.clTTEETE_variations_mnu = []

        pardict["thetastar"] = self.theta_star / 100.0
        if "h" in pardict.keys():
            del pardict["h"]

        # print(self.theta_star)
        for i in [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]:
            pardictcopy = copy.deepcopy(pardict)
            pardictcopy["thetastar"] = (
                self.theta_star * (1.0 + i * fracstepthetastar) / 100.0
            )

            # var = self.run_camb(pardictcopy)
            var = self.run_class_CMB(pardictcopy)
            self.clTTEETE_variations_thetastar.append((var[1:5]))

            pardictcopy = copy.deepcopy(pardict)
            pardictcopy["omega_b"] = self.Omegab * (1.0 + i * fracstepomegab)
            # var = self.run_camb(pardictcopy)
            var = self.run_class_CMB(pardictcopy)
            self.clTTEETE_variations_omegab.append((var[1:5]))

            pardictcopy = copy.deepcopy(pardict)
            pardictcopy["omega_cdm"] = self.Omega_cdm * (1.0 + i * fracstepomegacdm)
            # var = self.run_camb(pardictcopy)
            var = self.run_class_CMB(pardictcopy)
            self.clTTEETE_variations_omegacdm.append((var[1:5]))

            pardictcopy = copy.deepcopy(pardict)
            pardictcopy["A_s"] = (
                np.exp((self.lnAs10) * (1.0 + i * fracstepAs)) * 1.0e-10
            )
            # var = self.run_camb(pardictcopy)
            var = self.run_class_CMB(pardictcopy)
            self.clTTEETE_variations_As.append((var[1:5]))

            pardictcopy = copy.deepcopy(pardict)
            pardictcopy["n_s"] = self.ns * (1.0 + i * fracstepns)
            # var = self.run_camb(pardictcopy)
            var = self.run_class_CMB(pardictcopy)
            self.clTTEETE_variations_ns.append((var[1:5]))

            pardictcopy = copy.deepcopy(pardict)
            pardictcopy["tau_reio"] = float(pardict["tau_reio"]) * (
                1.0 + i * fracsteptau
            )
            # var = self.run_camb(pardictcopy)
            var = self.run_class_CMB(pardictcopy)
            self.clTTEETE_variations_tau.append((var[1:5]))

            if not pardict.as_bool("neutrino_mass_fixed"):
                pardictcopy = copy.deepcopy(pardict)
                pardictcopy["Sum_mnu"] = float(pardict["Sum_mnu"]) * (
                    1.0 + i * fracstepmnu
                )
                # var = self.run_camb(pardictcopy)
                var = self.run_class_CMB(pardictcopy)
                self.clTTEETE_variations_mnu.append((var[1:5]))

    def run_camb(self, pardict: ConfigObj):
        """Runs an instance of CAMB given the cosmological parameters in pardict and redshift bins

        Parameters
        ----------
        pardict: configobj.ConfigObj
            A dictionary of parameters read from the config file
        """

        import camb
        from scipy.interpolate import CubicSpline

        parlinear = copy.deepcopy(pardict)

        # Set the CAMB parameters
        pars = camb.CAMBparams()

        # pars.set_accuracy(
        #     lSampleBoost=2.0,
        #     lAccuracyBoost=2.0,
        #     AccuracyBoost=2.0,
        #     #high_precision=True,

        # )

        if "A_s" not in parlinear.keys():
            if "ln10^{10}A_s" in parlinear.keys():
                parlinear["A_s"] = np.exp(float(parlinear["ln10^{10}A_s"])) / 1.0e10
            else:
                print("Error: Neither ln10^{10}A_s nor A_s given in config file")
                exit()
        if "H0" in parlinear.keys():
            parlinear["H0"] = float(parlinear["H0"])
            parlinear["thetastar"] = None
        elif "h" in parlinear.keys():
            parlinear["H0"] = 100.0 * float(parlinear["h"])
            parlinear["thetastar"] = None
        elif "thetastar" in parlinear.keys():
            parlinear["thetastar"] = float(parlinear["thetastar"])
            parlinear["H0"] = None
        else:
            print("Error: Neither H0 nor h nor theta_s given in config file")
            exit()
        if "w0_fld" in parlinear.keys():
            pars.set_dark_energy(
                w=float(parlinear["w0_fld"]), dark_energy_model="fluid"
            )
        elif "wa_fld" in parlinear.keys() and "w0_fld" in parlinear.keys():
            pars.set_dark_energy(
                w=float(parlinear["w0_fld"]),
                wa=float(parlinear["wa_fld"]),
                dark_energy_model="ppf",
            )
        if "Neff" in parlinear.keys():
            parlinear["Neff"] = float(parlinear["Neff"])
        else:
            parlinear["Neff"] = 3.044
        pars.InitPower.set_params(
            As=float(parlinear["A_s"]), ns=float(parlinear["n_s"])
        )
        pars.set_cosmology(
            H0=parlinear["H0"],
            omch2=float(parlinear["omega_cdm"]),
            ombh2=float(parlinear["omega_b"]),
            omk=float(parlinear["Omega_k"]),
            tau=float(parlinear["tau_reio"]),
            mnu=float(parlinear["Sum_mnu"]),
            neutrino_hierarchy=parlinear["nu_hierarchy"],
            thetastar=parlinear["thetastar"],
            nnu=float(parlinear["Neff"]),
            # YHe=0.2478,
            # TCMB=2.7255,
        )
        pars.NonLinear = camb.model.NonLinear_none
        pars.set_for_lmax(5000)
        # Run CAMB
        results = camb.get_results(pars)

        ll = np.arange(2, 5001)

        # print(results.get_cmb_power_spectra(
        #     pars, CMB_unit="muK", lmax=5000, raw_cl=True))

        CMBdat = results.get_cmb_power_spectra(
            pars, CMB_unit="muK", lmax=5000, raw_cl=True
        )["total"]

        clTT = np.array(CMBdat[:, 0][2:])
        clEE = np.array(CMBdat[:, 1][2:])
        clTE = np.array(CMBdat[:, 3][2:])
        clBB = np.array(CMBdat[:, 2][2:])

        # import matplotlib.pyplot as plt
        # plt.plot(ll, CMBdat[:, 2][2:], label="BB")
        # plt.show()
        # exit()

        # Get some derived quantities
        area = float(pardict["skyarea"]) * (np.pi / 180.0) ** 2
        theta_star = results.get_derived_params()["thetastar"]
        # print(results.bbn_predictions())
        Omegab = float(parlinear["omega_b"])
        Omegacdm = float(parlinear["omega_cdm"])
        lnAs10 = np.log(float(parlinear["A_s"]) * 1.0e10)
        ns = float(parlinear["n_s"])
        tau = float(parlinear["tau_reio"])
        mnu = float(parlinear["Sum_mnu"]) if "Sum_mnu" in parlinear.keys() else 0.0

        A_phi = float(parlinear["A_phi"]) if "A_phi" in parlinear.keys() else 0.0
        log10Geff = (
            pardict.as_float("log10Geff") if "log10Geff" in pardict.keys() else -np.inf
        )

        alphanu = 8.0 / 7.0 * (11.0 / 4.0) ** (4.0 / 3.0)
        eps = A_phi * (1.0 / (1.0 + alphanu) - 3.044 / (3.044 + alphanu)) + 3.044 / (
            3.044 + alphanu
        )
        beta = eps / (3.044 / (3.044 + alphanu))
        factor = beta / (1.0 / (1.0 + alphanu) - 3.044 / (3.044 + alphanu))

        ellshift = factor * fitting_formula_interactingneutrinos(
            ll, log10Geff, theta_star
        ) - factor * fitting_formula_Montefalcone2025(ll)

        # ellshift = (A_phi) * fitting_formula_Montefalcone2025(ll)

        # import matplotlib.pyplot as plt
        # plt.plot(ll, clTT, label="no shift")

        clTT = CubicSpline(ll - ellshift, clTT)(ll)
        clEE = CubicSpline(ll - ellshift, clEE)(ll)
        clTE = CubicSpline(ll - ellshift, clTE)(ll)

        # plt.plot(ll, clTT, label="shifted")
        # plt.show()

        return (
            ll,
            clTT,
            clEE,
            clTE,
            clBB,
            theta_star,
            Omegab,
            Omegacdm,
            lnAs10,
            ns,
            tau,
            A_phi,
            log10Geff,
            area,
            mnu,
        )

    def run_class_CMB(self, pardictin: ConfigObj):
        parlinear = copy.deepcopy(pardictin)
        if "A_s" not in parlinear.keys():
            if "ln10^{10}A_s" in parlinear.keys():
                parlinear["A_s"] = np.exp(float(parlinear["ln10^{10}A_s"])) / 1.0e10
            else:
                print("Error: Neither ln10^{10}A_s nor A_s given in config file")
                exit()
        if "H0" in parlinear.keys():
            parlinear["H0"] = float(parlinear["H0"])
            parlinear["thetastar"] = None
        elif "h" in parlinear.keys():
            parlinear["H0"] = 100.0 * float(parlinear["h"])
            parlinear["thetastar"] = None
        elif "thetastar" in parlinear.keys():
            parlinear["thetastar"] = float(parlinear["thetastar"])
            parlinear["H0"] = None
        else:
            print("Error: Neither H0 nor h nor theta_s given in config file")
            exit()
        if "Neff" in parlinear.keys():
            parlinear["Neff"] = float(parlinear["Neff"])
        else:
            parlinear["Neff"] = 3.044

        # Set the CLASS parameters
        from classy import Class

        M = Class()

        # M.set(
        #     {
        #         "omega_b": float(parlinear["omega_b"]),
        #         "omega_cdm": float(parlinear["omega_cdm"]),
        #         "theta_s_100": float(parlinear["thetastar"]) * 100.0,
        #         "A_s": float(parlinear["A_s"]),
        #         "N_ur": float(parlinear["Neff"])
        #         - 1.0132,  # This is the effective number of relativistic species
        #         "N_ncdm": 1,  # Number of massive neutrino species
        #         "m_ncdm": float(parlinear["Sum_mnu"]),
        #         "tau_reio": float(parlinear["tau_reio"]),
        #         "n_s": float(parlinear["n_s"]),
        #     }
        # )

        if parlinear["thetastar"] is not None:
            # print(parlinear['thetastar'])
            M.set(
                {
                    "omega_b": float(parlinear["omega_b"]),
                    "omega_cdm": float(parlinear["omega_cdm"]),
                    "theta_s_100": float(parlinear["thetastar"]) * 100.0,
                    "A_s": float(parlinear["A_s"]),
                    "N_ur": float(parlinear["Neff"])
                    - 1.0132,  # This is the effective number of relativistic species
                    "N_ncdm": 1,  # Number of massive neutrino species
                    "m_ncdm": float(parlinear["Sum_mnu"]),
                    "tau_reio": float(parlinear["tau_reio"]),
                    "n_s": float(parlinear["n_s"]),
                }
            )

        else:
            M.set(
                {
                    "omega_b": float(parlinear["omega_b"]),
                    "omega_cdm": float(parlinear["omega_cdm"]),
                    "h": float(parlinear["h"]),
                    "A_s": float(parlinear["A_s"]),
                    "N_ur": float(parlinear["Neff"])
                    - 1.0132,  # This is the effective number of relativistic species
                    "N_ncdm": 1,  # Number of massive neutrino species
                    "m_ncdm": float(parlinear["Sum_mnu"]),
                    "tau_reio": float(parlinear["tau_reio"]),
                    "n_s": float(parlinear["n_s"]),
                }
            )

        M.set({"output": "tCl,lCl,pCl", "l_max_scalars": 5000, "lensing": "yes"})

        M.compute()

        T_cmb = M.T_cmb()  # Temperature of the CMB in Kelvin
        cls = M.raw_cl(5000)
        # cls = M.lensed_cl(5000)  # Get the lensed power spectra
        # print(cls.keys())
        ll = cls["ell"][2:]
        ll = np.array(ll, dtype=float)
        clTT = cls["tt"][2:] * T_cmb**2 * 1.0e12  # Convert to microK^2
        clEE = cls["ee"][2:] * T_cmb**2 * 1.0e12  # Convert to microK^2
        clTE = cls["te"][2:] * T_cmb**2 * 1.0e12  # Convert to microK^2
        clBB = cls["bb"][2:] * T_cmb**2 * 1.0e12  # Convert to microK^2

        area = float(pardictin["skyarea"]) * (np.pi / 180.0) ** 2
        # print(results.bbn_predictions())
        Omegab = float(parlinear["omega_b"])
        Omegacdm = float(parlinear["omega_cdm"])
        lnAs10 = np.log(float(parlinear["A_s"]) * 1.0e10)
        ns = float(parlinear["n_s"])
        tau = float(parlinear["tau_reio"])
        mnu = float(parlinear["Sum_mnu"]) if "Sum_mnu" in parlinear.keys() else 0.0
        theta_star = M.get_current_derived_parameters(["100*theta_s"])[
            "100*theta_s"
        ]  # parlinear["thetastar"]*100.0 if "thetastar" in parlinear.keys() else None
        # print(theta_star)
        A_phi = float(parlinear["A_phi"]) if "A_phi" in parlinear.keys() else 0.0
        log10Geff = (
            pardictin.as_float("log10Geff")
            if "log10Geff" in pardictin.keys()
            else -np.inf
        )

        alphanu = 8.0 / 7.0 * (11.0 / 4.0) ** (4.0 / 3.0)
        eps = A_phi * (1.0 / (1.0 + alphanu) - 3.044 / (3.044 + alphanu)) + 3.044 / (
            3.044 + alphanu
        )
        beta = eps / (3.044 / (3.044 + alphanu))
        factor = beta / (1.0 / (1.0 + alphanu) - 3.044 / (3.044 + alphanu))

        ellshift = factor * fitting_formula_interactingneutrinos(
            ll, log10Geff, theta_star
        ) - factor * fitting_formula_Montefalcone2025(ll)

        # ellshift = (A_phi) * fitting_formula_Montefalcone2025(ll)

        from scipy.interpolate import CubicSpline

        clTT = CubicSpline(ll - ellshift, clTT)(ll)
        clEE = CubicSpline(ll - ellshift, clEE)(ll)
        clTE = CubicSpline(ll - ellshift, clTE)(ll)

        return (
            ll,
            clTT,
            clEE,
            clTE,
            clBB,
            theta_star,
            Omegab,
            Omegacdm,
            lnAs10,
            ns,
            tau,
            A_phi,
            log10Geff,
            area,
            mnu,
        )


def write_fisher(
    pardict: ConfigObj,
    cov_inv: npt.NDArray,
    parameter_means: list,
) -> None:
    """
    Write Fisher predictions to text files

    """

    cov_filename = pardict["outputfile"] + "_cov.txt"
    data_filename = pardict["outputfile"] + "_dat.txt"

    np.savetxt(cov_filename, cov_inv)
    np.savetxt(data_filename, parameter_means)


def fitting_formula_Montefalcone2025(ll: npt.NDArray) -> npt.NDArray:
    """Compute the fitting formula for the power spectrum phase shift (for standard model neutrinos) based on Montefalcone et. al., 2025"""
    ell_inf = 11.0
    ell_star = 483.0
    eps = -1.69

    return ell_inf / (1.0 + (ll / ell_star) ** eps)


def amplitude_modulation_geff(
    ellarr: npt.NDArray, log10Geff: float, thetas: float
) -> npt.NDArray:
    """Amplitude modulation based on Geff"""
    amplitude_modulation = 2.03359823e-05 * (log10Geff**6) + 5.36960127e-04 * (
        log10Geff**5
    )
    amplitude_modulation = (
        amplitude_modulation
        + 4.55360397e-03 * (log10Geff**4)
        + 9.73443600e-03 * (log10Geff**3)
    )
    amplitude_modulation = (
        amplitude_modulation
        + -5.52743545e-02 * (log10Geff**2)
        + -3.04729338e-01 * (log10Geff)
        + 5.89273173e-01
    )
    if log10Geff <= -2:
        amplitude_modulation = 1.0
    return amplitude_modulation


def exponential_damping_geff(
    ellarr: npt.NDArray, log10Geff: float, thetas: float
) -> npt.NDArray:
    """Compute the exponential damping based on Geff"""
    exponential_damp_modulation = (
        7.84726283e-06 * (log10Geff**6)
        + 2.33592405e-04 * (log10Geff**5)
        + 2.55941525e-03 * (log10Geff**4)
    )

    exponential_damp_modulation = (
        exponential_damp_modulation
        + 1.28825961e-02 * (log10Geff**3)
        + 2.80788885e-02 * (log10Geff**2)
    )
    exponential_damp_modulation = (
        exponential_damp_modulation + 1.09893067e-02 * (log10Geff) + -2.89929198e-02
    )

    if log10Geff <= -3.5:
        exponential_damp_modulation = 0.0
    exponential_damping = np.exp(
        ellarr * 1.0628324 / 50.0 * exponential_damp_modulation
    )  # thetas here is actually 100 * thetas
    return exponential_damping


def fitting_formula_interactingneutrinos(
    ellarr: npt.NDArray,
    log10Geff: float,
    thetas: float,  # actually 100 * thetas
) -> npt.NDArray:
    """Compute the fitting formula for the power spectrum phase shift (for interacting neutrinos) based on Baumann et. al., 2019
    and multiply by new parameters to capture impact of log10Geff on the phase shift."""

    standard_phase = fitting_formula_Montefalcone2025(ellarr)
    amplitude_modulation = amplitude_modulation_geff(ellarr, log10Geff, thetas)
    if log10Geff < -6:
        amplitude_modulation = 1.0
    exponential_damp_modulation = exponential_damping_geff(ellarr, log10Geff, thetas)
    return amplitude_modulation * standard_phase * exponential_damp_modulation


def deriv_amplitude_modulation_geff(
    ellarr: npt.NDArray, log10Geff: float, thetas: float
) -> npt.NDArray:
    """Amplitude modulation based on Geff"""

    amplitude_modulation_der = 6 * 2.03359823e-05 * (
        log10Geff**5
    ) + 5 * 5.36960127e-04 * (log10Geff**4)
    amplitude_modulation_der = (
        amplitude_modulation_der
        + 4 * 4.55360397e-03 * (log10Geff**3)
        + 3 * 9.734436e-03 * (log10Geff**2)
    )
    amplitude_modulation_der = (
        amplitude_modulation_der + 2 * -5.52743545e-02 * (log10Geff) + -3.04729338e-01
    )
    if log10Geff < -2.0:
        amplitude_modulation_der = 0.0
    return amplitude_modulation_der


def deriv_exponential_damping_geff(
    ellarr: npt.NDArray, log10Geff: float, thetas: float
) -> npt.NDArray:
    """Compute the exponential damping based on Geff"""
    exponential_damp_modulation_der = (
        6 * 7.8726283e-06 * (log10Geff**5)
        + 5 * 2.33592405e-04 * (log10Geff**4)
        + 4 * 2.55941525e-03 * (log10Geff**3)
    )
    exponential_damp_modulation_der = (
        exponential_damp_modulation_der
        + 3 * 1.28825961e-02 * (log10Geff**2)
        + 2 * 2.80788885e-02 * (log10Geff)
    )
    exponential_damp_modulation_der = exponential_damp_modulation_der + 1.09893067e-02

    if log10Geff < -3.5:
        exponential_damp_modulation_der = 0.0

    return exponential_damp_modulation_der


def derivell_geff(ellarr: npt.NDArray, log10Geff: float, thetas: float, A: float):
    # A = A_phi = (eps - eps3044) / (eps1.0 - eps3044)
    # lets compute beta from this
    alphanu = 8.0 / 7.0 * (11.0 / 4.0) ** (4.0 / 3.0)
    eps = A * (1.0 / (1.0 + alphanu) - 3.044 / (3.044 + alphanu)) + 3.044 / (
        3.044 + alphanu
    )
    beta = eps / (3.044 / (3.044 + alphanu))
    denom = 1.0 / (1.0 + alphanu) - 3.044 / (3.044 + alphanu)
    factor = beta / denom

    firstterm = (
        deriv_amplitude_modulation_geff(ellarr, log10Geff, thetas)
        * fitting_formula_Montefalcone2025(ellarr)
        * factor
        * (exponential_damping_geff(ellarr, log10Geff, thetas))
    )  # A'(Geff) * factor * f(ell) * exp(ll thetas B(Geff))
    secondterm = (
        amplitude_modulation_geff(ellarr, log10Geff, thetas)
        * deriv_exponential_damping_geff(ellarr, log10Geff, thetas)
        * (exponential_damping_geff(ellarr, log10Geff, thetas))
        * fitting_formula_Montefalcone2025(ellarr)
        * factor
        * 1.0628324
        / 50.0
        * ellarr
    )

    derivk_geff = firstterm + secondterm

    return derivk_geff


class CosmoResults_quick:
    def __init__(
        self,
        pardict: ConfigObj,
    ):
        (
            self.ell,
            self.clTT,
            self.clEE,
            self.clTE,
            self.clBB,
            self.theta_star,
            self.Omegab,
            self.Omega_cdm,
            self.lnAs10,
            self.ns,
            self.tau,
            self.A_phi,
            self.log10Geff,
            self.area,
            self.mnu,
        ) = self.run_class_CMB(pardict)
        # ) = self.run_camb(pardict)

        self.lminTT = int(pardict["lminTT"])
        self.lmaxTT = int(pardict["lmaxTT"])
        self.lminTE = int(pardict["lminTE"])
        self.lmaxTE = int(pardict["lmaxTE"])
        self.lminEE = int(pardict["lminEE"])
        self.lmaxEE = int(pardict["lmaxEE"])
        self.lminBB = int(pardict["lminBB"])
        self.lmaxBB = int(pardict["lmaxBB"])
        self.use_TE = True if pardict["use_TE"] == "True" else False
        self.use_EE = True if pardict["use_EE"] == "True" else False
        self.use_BB = True if pardict["use_BB"] == "True" else False
        self.noise_Planck = True if pardict["noise_Planck"] == "True" else False

    def run_camb(self, pardict: ConfigObj):
        """Runs an instance of CAMB given the cosmological parameters in pardict and redshift bins

        Parameters
        ----------
        pardict: configobj.ConfigObj
            A dictionary of parameters read from the config file
        """

        import camb
        from scipy.interpolate import CubicSpline

        parlinear = copy.deepcopy(pardict)

        # Set the CAMB parameters
        pars = camb.CAMBparams()

        # pars.set_accuracy(
        #     lSampleBoost=2.0,
        #     lAccuracyBoost=2.0,
        #     AccuracyBoost=2.0,
        #     #high_precision=True,

        # )

        if "A_s" not in parlinear.keys():
            if "ln10^{10}A_s" in parlinear.keys():
                parlinear["A_s"] = np.exp(float(parlinear["ln10^{10}A_s"])) / 1.0e10
            else:
                print("Error: Neither ln10^{10}A_s nor A_s given in config file")
                exit()
        if "H0" in parlinear.keys():
            parlinear["H0"] = float(parlinear["H0"])
            parlinear["thetastar"] = None
        elif "h" in parlinear.keys():
            parlinear["H0"] = 100.0 * float(parlinear["h"])
            parlinear["thetastar"] = None
        elif "thetastar" in parlinear.keys():
            parlinear["thetastar"] = float(parlinear["thetastar"])
            parlinear["H0"] = None
        else:
            print("Error: Neither H0 nor h nor theta_s given in config file")
            exit()
        if "w0_fld" in parlinear.keys():
            pars.set_dark_energy(
                w=float(parlinear["w0_fld"]), dark_energy_model="fluid"
            )
        elif "wa_fld" in parlinear.keys() and "w0_fld" in parlinear.keys():
            pars.set_dark_energy(
                w=float(parlinear["w0_fld"]),
                wa=float(parlinear["wa_fld"]),
                dark_energy_model="ppf",
            )
        if "Neff" in parlinear.keys():
            parlinear["Neff"] = float(parlinear["Neff"])
        else:
            parlinear["Neff"] = 3.044
        pars.InitPower.set_params(
            As=float(parlinear["A_s"]), ns=float(parlinear["n_s"])
        )
        pars.set_cosmology(
            H0=parlinear["H0"],
            omch2=float(parlinear["omega_cdm"]),
            ombh2=float(parlinear["omega_b"]),
            omk=float(parlinear["Omega_k"]),
            tau=float(parlinear["tau_reio"]),
            mnu=float(parlinear["Sum_mnu"]),
            neutrino_hierarchy=parlinear["nu_hierarchy"],
            thetastar=parlinear["thetastar"],
            nnu=float(parlinear["Neff"]),
            # YHe=0.2478,
            # TCMB=2.7255,
        )
        pars.NonLinear = camb.model.NonLinear_none
        pars.set_for_lmax(5000)
        # Run CAMB
        results = camb.get_results(pars)

        ll = np.arange(2, 5001)

        # print(results.get_cmb_power_spectra(
        #     pars, CMB_unit="muK", lmax=5000, raw_cl=True))

        CMBdat = results.get_cmb_power_spectra(
            pars, CMB_unit="muK", lmax=5000, raw_cl=True
        )["unlensed_total"]

        clTT = np.array(CMBdat[:, 0][2:])
        clEE = np.array(CMBdat[:, 1][2:])
        clTE = np.array(CMBdat[:, 3][2:])
        clBB = np.array(CMBdat[:, 2][2:])

        # import matplotlib.pyplot as plt
        # plt.plot(ll, CMBdat[:, 2][2:], label="BB")
        # plt.show()
        # exit()

        # Get some derived quantities
        area = float(pardict["skyarea"]) * (np.pi / 180.0) ** 2
        theta_star = results.get_derived_params()["thetastar"]
        # print(results.bbn_predictions())
        Omegab = float(parlinear["omega_b"])
        Omegacdm = float(parlinear["omega_cdm"])
        lnAs10 = np.log(float(parlinear["A_s"]) * 1.0e10)
        ns = float(parlinear["n_s"])
        tau = float(parlinear["tau_reio"])
        mnu = float(parlinear["Sum_mnu"]) if "Sum_mnu" in parlinear.keys() else 0.0

        A_phi = float(parlinear["A_phi"]) if "A_phi" in parlinear.keys() else 0.0
        log10Geff = (
            pardict.as_float("log10Geff") if "log10Geff" in pardict.keys() else -np.inf
        )

        alphanu = 8.0 / 7.0 * (11.0 / 4.0) ** (4.0 / 3.0)
        eps = A_phi * (1.0 / (1.0 + alphanu) - 3.044 / (3.044 + alphanu)) + 3.044 / (
            3.044 + alphanu
        )
        beta = eps / (3.044 / (3.044 + alphanu))
        factor = -1.0 * beta / (1.0 / (1.0 + alphanu) - 3.044 / (3.044 + alphanu))

        ellshift = factor * fitting_formula_interactingneutrinos(
            ll, log10Geff, theta_star
        ) - factor * fitting_formula_Montefalcone2025(ll)

        # ellshift = (A_phi) * fitting_formula_Montefalcone2025(ll)

        # import matplotlib.pyplot as plt
        # plt.plot(ll, clTT, label="no shift")

        clTT = CubicSpline(ll - ellshift, clTT)(ll)
        clEE = CubicSpline(ll - ellshift, clEE)(ll)
        clTE = CubicSpline(ll - ellshift, clTE)(ll)

        # plt.plot(ll, clTT, label="shifted")
        # plt.show()

        return (
            ll,
            clTT,
            clEE,
            clTE,
            clBB,
            theta_star,
            Omegab,
            Omegacdm,
            lnAs10,
            ns,
            tau,
            A_phi,
            log10Geff,
            area,
            mnu,
        )

    def run_class_CMB(self, pardict: ConfigObj):
        parlinear = copy.deepcopy(pardict)
        if "A_s" not in parlinear.keys():
            if "ln10^{10}A_s" in parlinear.keys():
                parlinear["A_s"] = np.exp(float(parlinear["ln10^{10}A_s"])) / 1.0e10
            else:
                print("Error: Neither ln10^{10}A_s nor A_s given in config file")
                exit()
        if "H0" in parlinear.keys():
            parlinear["H0"] = float(parlinear["H0"])
            parlinear["thetastar"] = None
        elif "h" in parlinear.keys():
            parlinear["H0"] = 100.0 * float(parlinear["h"])
            parlinear["thetastar"] = None
        elif "thetastar" in parlinear.keys():
            parlinear["thetastar"] = float(parlinear["thetastar"])
            parlinear["H0"] = None
        else:
            print("Error: Neither H0 nor h nor theta_s given in config file")
            exit()
        if "Neff" in parlinear.keys():
            parlinear["Neff"] = float(parlinear["Neff"])
        else:
            parlinear["Neff"] = 3.044

        # Set the CLASS parameters
        from classy import Class

        M = Class()

        if parlinear["thetastar"] is not None:
            M.set(
                {
                    "omega_b": float(parlinear["omega_b"]),
                    "omega_cdm": float(parlinear["omega_cdm"]),
                    "theta_s_100": float(parlinear["thetastar"]) * 100.0,
                    "A_s": float(parlinear["A_s"]),
                    "N_ur": float(parlinear["Neff"])
                    - 1.0132,  # This is the effective number of relativistic species
                    "N_ncdm": 1,  # Number of massive neutrino species
                    "m_ncdm": float(parlinear["Sum_mnu"]),
                    "tau_reio": float(parlinear["tau_reio"]),
                    "n_s": float(parlinear["n_s"]),
                }
            )

        else:
            M.set(
                {
                    "omega_b": float(parlinear["omega_b"]),
                    "omega_cdm": float(parlinear["omega_cdm"]),
                    "h": float(parlinear["h"]),
                    "A_s": float(parlinear["A_s"]),
                    "N_ur": float(parlinear["Neff"])
                    - 1.0132,  # This is the effective number of relativistic species
                    "N_ncdm": 1,  # Number of massive neutrino species
                    "m_ncdm": float(parlinear["Sum_mnu"]),
                    "tau_reio": float(parlinear["tau_reio"]),
                    "n_s": float(parlinear["n_s"]),
                }
            )

        M.set({"output": "tCl,lCl,pCl", "l_max_scalars": 5000, "lensing": "yes"})

        M.compute()

        T_cmb = M.T_cmb()  # Temperature of the CMB in Kelvin
        cls = M.raw_cl(5000)
        # cls = M.lensed_cl(5000)  # Get the lensed power spectra
        # print(cls.keys())
        ll = cls["ell"][2:]
        ll = np.array(ll, dtype=float)
        clTT = cls["tt"][2:] * T_cmb**2 * 1.0e12  # Convert to microK^2
        clEE = cls["ee"][2:] * T_cmb**2 * 1.0e12  # Convert to microK^2
        clTE = cls["te"][2:] * T_cmb**2 * 1.0e12  # Convert to microK^2
        clBB = cls["bb"][2:] * T_cmb**2 * 1.0e12  # Convert to microK^2

        area = float(pardict["skyarea"]) * (np.pi / 180.0) ** 2
        # print(results.bbn_predictions())
        Omegab = float(parlinear["omega_b"])
        Omegacdm = float(parlinear["omega_cdm"])
        lnAs10 = np.log(float(parlinear["A_s"]) * 1.0e10)
        ns = float(parlinear["n_s"])
        tau = float(parlinear["tau_reio"])
        mnu = float(parlinear["Sum_mnu"]) if "Sum_mnu" in parlinear.keys() else 0.0
        theta_star = M.get_current_derived_parameters(["100*theta_s"])[
            "100*theta_s"
        ]  # parlinear["thetastar"]*100.0 if "thetastar" in parlinear.keys() else None

        A_phi = float(parlinear["A_phi"]) if "A_phi" in parlinear.keys() else 0.0
        log10Geff = (
            pardict.as_float("log10Geff") if "log10Geff" in pardict.keys() else -np.inf
        )

        alphanu = 8.0 / 7.0 * (11.0 / 4.0) ** (4.0 / 3.0)
        eps = A_phi * (1.0 / (1.0 + alphanu) - 3.044 / (3.044 + alphanu)) + 3.044 / (
            3.044 + alphanu
        )
        beta = eps / (3.044 / (3.044 + alphanu))
        # beta = 0.0
        # log10Geff = -12.0
        factor = (
            -1.0
            * (3.044 / (3.044 + alphanu))
            / (1.0 / (1.0 + alphanu) - 3.044 / (3.044 + alphanu))
        )

        ellshift = beta * factor * fitting_formula_interactingneutrinos(
            ll, log10Geff, theta_star
        ) - factor * fitting_formula_Montefalcone2025(ll)

        # ellshift = (A_phi) * fitting_formula_Montefalcone2025(ll)

        # plt.plot(ll, clTT*ll*(ll+1.0), label="TT", color='blue', linestyle='-.')
        from scipy.interpolate import CubicSpline
        # clTT_compare = CubicSpline(ll - factor * fitting_formula_interactingneutrinos(ll, log10Geff, theta_star), clTT)(ll)
        # import matplotlib.pyplot as plt
        # plt.plot(ll, clTT, label="no shift")

        clTT = CubicSpline(ll - ellshift, clTT)(ll)
        clEE = CubicSpline(ll - ellshift, clEE)(ll)
        clTE = CubicSpline(ll - ellshift, clTE)(ll)

        # plt.plot(ll, clTT, label="shifted")
        # plt.legend()
        # plt.show()

        # plt.plot(ll, clTT*ll*(ll+1.0), label="TTshift", color='green', linestyle=':', lw=2)
        # plt.plot(ll, clTT_compare*ll*(ll+1.0), label="TTcompare", color='red', linestyle='--')
        # plt.show()
        # plt.plot(ll, clTT)
        # plt.show()

        # plt.plot(ll, ellshift)
        # plt.show()

        return (
            ll,
            clTT,
            clEE,
            clTE,
            clBB,
            theta_star,
            Omegab,
            Omegacdm,
            lnAs10,
            ns,
            tau,
            A_phi,
            log10Geff,
            area,
            mnu,
        )
