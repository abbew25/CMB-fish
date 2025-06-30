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
    ):
        (
            self.ell,
            self.clTT,
            self.clEE,
            self.clTE,
            self.theta_star,
            self.Omegab,
            self.Omega_cdm,
            self.lnAs10,
            self.ns,
            self.tau,
            self.A_phi,
            self.log10Geff,
            self.area,
        ) = self.run_camb(pardict)

        self.lminTT = int(pardict["lminTT"])
        self.lmaxTT = int(pardict["lmaxTT"])
        self.lminTE = int(pardict["lminTE"])
        self.lmaxTE = int(pardict["lmaxTE"])
        self.lminEE = int(pardict["lminEE"])
        self.lmaxEE = int(pardict["lmaxEE"])
        self.use_TE = True if pardict["use_TE"] == "True" else False
        self.use_EE = True if pardict["use_EE"] == "True" else False
        self.noise_Planck = True if pardict["noise_Planck"] == "True" else False

        # for derivatives w.r.t. thetastar
        self.clTTEETE_variations_thetastar = []
        self.clTTEETE_variations_omegab = []
        self.clTTEETE_variations_omegacdm = []
        self.clTTEETE_variations_As = []
        self.clTTEETE_variations_ns = []
        self.clTTEETE_variations_tau = []
        for i in [-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0]:
            pardictcopy = copy.deepcopy(pardict)
            pardictcopy["thetastar"] = (
                self.theta_star * (1.0 + i * fracstepthetastar) / 100.0
            )
            if "h" in pardictcopy.keys():
                del pardictcopy["h"]

            var = self.run_camb(pardictcopy)

            self.clTTEETE_variations_thetastar.append((var[1:4]))

            pardictcopy = copy.deepcopy(pardict)
            pardictcopy["omega_b"] = self.Omegab * (1.0 + i * fracstepomegab)
            var = self.run_camb(pardictcopy)

            self.clTTEETE_variations_omegab.append((var[1:4]))

            pardictcopy = copy.deepcopy(pardict)
            pardictcopy["omega_cdm"] = self.Omega_cdm * (1.0 + i * fracstepomegacdm)
            var = self.run_camb(pardictcopy)
            self.clTTEETE_variations_omegacdm.append((var[1:4]))

            pardictcopy = copy.deepcopy(pardict)
            pardictcopy["A_s"] = (
                np.exp((self.lnAs10) * (1.0 + i * fracstepAs)) * 1.0e-10
            )
            var = self.run_camb(pardictcopy)
            self.clTTEETE_variations_As.append((var[1:4]))

            pardictcopy = copy.deepcopy(pardict)
            pardictcopy["n_s"] = self.ns * (1.0 + i * fracstepns)
            var = self.run_camb(pardictcopy)
            self.clTTEETE_variations_ns.append((var[1:4]))

            pardictcopy = copy.deepcopy(pardict)
            pardictcopy["tau_reio"] = float(pardict["tau_reio"]) * (
                1.0 + i * fracsteptau
            )
            var = self.run_camb(pardictcopy)
            self.clTTEETE_variations_tau.append((var[1:4]))

    def run_camb(self, pardict: ConfigObj):
        """Runs an instance of CAMB given the cosmological parameters in pardict and redshift bins

        Parameters
        ----------
        pardict: configobj.ConfigObj
            A dictionary of parameters read from the config file
        """

        import camb
        from scipy.interpolate import splrep

        parlinear = copy.deepcopy(pardict)

        # Set the CAMB parameters
        pars = camb.CAMBparams()
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

        # Get some derived quantities
        area = float(pardict["skyarea"]) * (np.pi / 180.0) ** 2
        theta_star = results.get_derived_params()["thetastar"]
        Omegab = float(parlinear["omega_b"])
        Omegacdm = float(parlinear["omega_cdm"])
        lnAs10 = np.log(float(parlinear["A_s"]) * 1.0e10)
        ns = float(parlinear["n_s"])
        tau = float(parlinear["tau_reio"])

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

        clTT = splrep(ll + ellshift, clTT)
        clEE = splrep(ll + ellshift, clEE)
        clTE = splrep(ll + ellshift, clTE)

        return (
            ll,
            clTT,
            clEE,
            clTE,
            theta_star,
            Omegab,
            Omegacdm,
            lnAs10,
            ns,
            tau,
            A_phi,
            log10Geff,
            area,
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
    amplitude_modulation = 3.577e-05 * (log10Geff**6) + 1.410e-03 * (log10Geff**5)
    amplitude_modulation = (
        amplitude_modulation + 1.823e-02 * (log10Geff**4) + 9.582e-02 * (log10Geff**3)
    )
    amplitude_modulation = (
        amplitude_modulation
        + 1.575e-01 * (log10Geff**2)
        + -2.160e-01 * (log10Geff)
        + 3.536e-01
    )
    if log10Geff < -6:
        amplitude_modulation = 1.0
    return amplitude_modulation


def exponential_damping_geff(
    ellarr: npt.NDArray, log10Geff: float, thetas: float
) -> npt.NDArray:
    """Compute the exponential damping based on Geff"""
    exponential_damp_modulation = (
        8.259e-11 * (log10Geff**6)
        + 4.496e-03 * (log10Geff**5)
        + 1.627e-06 * (log10Geff**4)
    )
    exponential_damp_modulation = (
        exponential_damp_modulation
        + 1.946e-05 * (log10Geff**3)
        + 9.129e-05 * (log10Geff**2)
    )
    exponential_damp_modulation = (
        exponential_damp_modulation + 1.241e-04 * (log10Geff) + -1.068e-04
    )

    if log10Geff < -6:
        exponential_damp_modulation = 0.0
    exponential_damping = np.exp(
        ellarr * thetas * exponential_damp_modulation
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

    amplitude_modulation_der = 6 * 3.577e-05 * (log10Geff**5) + 5 * 1.410e-03 * (
        log10Geff**4
    )
    amplitude_modulation_der = (
        amplitude_modulation_der
        + 4 * 1.823e-02 * (log10Geff**3)
        + 3 * 9.582e-02 * (log10Geff**2)
    )
    amplitude_modulation_der = (
        amplitude_modulation_der + 2 * 1.575e-01 * (log10Geff) + -2.160e-01
    )
    if log10Geff < -6:
        amplitude_modulation_der = 0.0
    return amplitude_modulation_der


def deriv_exponential_damping_geff(
    ellarr: npt.NDArray, log10Geff: float, thetas: float
) -> npt.NDArray:
    """Compute the exponential damping based on Geff"""
    exponential_damp_modulation_der = (
        6 * 8.259e-11 * (log10Geff**5)
        + 5 * 4.496e-03 * (log10Geff**4)
        + 4 * 1.627e-06 * (log10Geff**3)
    )
    exponential_damp_modulation_der = (
        exponential_damp_modulation_der
        + 3 * 1.946e-05 * (log10Geff**2)
        + 2 * 9.129e-05 * (log10Geff)
    )
    exponential_damp_modulation_der = exponential_damp_modulation_der + 1.241e-04

    if log10Geff < -6:
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
        * thetas
        * ellarr
    )

    derivk_geff = firstterm + secondterm

    return derivk_geff
