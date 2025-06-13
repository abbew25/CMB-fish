import copy
import numpy as np
import numpy.typing as npt
from configobj import ConfigObj


class InputData:
    def __init__(self, pardict: ConfigObj):
        return None 

# This class contains everything we might need to set up to compute the fisher matrix
class CosmoResults:
    def __init__(self, pardict: ConfigObj):
        (
            self.ell,
            self.clTT,
            self.theta_star, 
            self.A_phi,
            self.log10Geff,
            area

        ) = self.run_camb(pardict)
        
        pardictbefore = pardict.copy() 
        pardictafter = pardict.copy() 
        pardictbefore['thetastar'] = self.theta_star* 0.95 
        pardictbefore.popitem('h')
        pardictafter['thetastar'] = self.theta_star * 1.05  
        pardictafter.popitem('h')
        
        self.clTT_before = self.run_camb(pardictbefore)[1] 
        self.clTT_after = self.run_camb(pardictafter)[1] 

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

        # Run CAMB
        results = camb.get_results(pars)

        # Get the power spectrum
        kin, zin, pklin = results.get_matter_power_spectrum(
            minkh=2.0e-5, maxkh=10.0, npoints=2000
        )
        
        ll = np.arange(1, 3000)
        CMBdat = results.get_total_cls(
            CMB_unit="muK", lmax=3000, raw_cl=True)
        clTT, clTE, clEE, clBB = (
            CMBdat[:, 0],
            CMBdat[:, 1],
            CMBdat[:, 2],
            CMBdat[:, 3],
        )

        # Get some derived quantities
        area = float(pardict["skyarea"]) * (np.pi / 180.0) ** 2
        theta_star = results.get_derived_params()["theta_s"]
        
        
        A_phi = (
            pardict.as_float("A_phi") if "A_phi" in parlinear.keys() else 1.0
        )
        log10Geff = (
            pardict.as_float("log10Geff") if "log10Geff" in pardict.keys() else -np.inf
        )

        # ellshift = (
        #     A_phi * fitting_formula_interactingneutrinos(kin, log10Geff, theta_star)
        #     - fitting_formula_Baumann19(kin)
        # ) / theta_star
        
        ellshift = A_phi * (1.0- fitting_formula_Montefalcone2025(ll)) / theta_star
        
        clTT = splrep(ll + ellshift, clTT)

        return (
            
            ll,
            clTT,
            theta_star,
            A_phi,
            log10Geff,
            area
            
        )

   
def write_fisher(
    pardict: ConfigObj,
    cov_inv: npt.NDArray,
    parameter_means: list,
) -> None:
    """
    Write Fisher predictions to text files  
    
    """

    cov_filename = (
        pardict["outputfile"] + "_cov.txt"
    )
    data_filename = (
        pardict["outputfile"] + "_dat.txt"
    )

    np.savetxt(cov_filename, cov_inv)
    np.savetxt(data_filename, parameter_means)


def fitting_formula_Montefalcone2025(ll: npt.NDArray) -> npt.NDArray:
    """Compute the fitting formula for the power spectrum phase shift (for standard model neutrinos) based on Montefalcone et. al., 2025"""
    ell_inf = 11.0
    ell_star = 483.0
    eps = -1.69 
    
    return (ell_inf/(1.0 + (ll/ell_star)**eps))



# def amplitude_modulation_geff(
#     ks: npt.NDArray, log10Geff: float, rs: float
# ) -> npt.NDArray:
#     """Amplitude modulation based on Geff"""
#     amplitude_modulation = 2.03359823e-05 * (log10Geff**6) + 5.36960127e-04 * (
#         log10Geff**5
#     )
#     amplitude_modulation = (
#         amplitude_modulation
#         + 4.55360397e-03 * (log10Geff**4)
#         + 9.73443600e-03 * (log10Geff**3)
#     )
#     amplitude_modulation = (
#         amplitude_modulation
#         + -5.52743545e-02 * (log10Geff**2)
#         + -3.04729338e-01 * (log10Geff)
#         + 5.89273173e-01
#     )
#     if log10Geff < -6:
#         amplitude_modulation = 1.0
#     return amplitude_modulation


# def exponential_damping_geff(
#     ks: npt.NDArray, log10Geff: float, rs: float
# ) -> npt.NDArray:
#     """Compute the exponential damping based on Geff"""
#     exponential_damp_modulation = (
#         7.84726283e-06 * (log10Geff**6)
#         + 2.33592405e-04 * (log10Geff**5)
#         + 2.55941525e-03 * (log10Geff**4)
#     )
#     exponential_damp_modulation = (
#         exponential_damp_modulation
#         + 1.28825961e-02 * (log10Geff**3)
#         + 2.80788885e-02 * (log10Geff**2)
#     )
#     exponential_damp_modulation = (
#         exponential_damp_modulation + 1.09893067e-02 * (log10Geff) + -2.89929198e-02
#     )
#     if log10Geff < -6:
#         exponential_damp_modulation = 0.0
#     exponential_damping = np.exp(ks * rs * exponential_damp_modulation)
#     return exponential_damping


# def fitting_formula_interactingneutrinos(
#     ks: npt.NDArray, log10Geff: float, rs: float
# ) -> npt.NDArray:
#     """Compute the fitting formula for the power spectrum phase shift (for interacting neutrinos) based on Baumann et. al., 2019
#     and multiply by new parameters to capture impact of log10Geff on the phase shift."""
#     standard_phase = fitting_formula_Baumann19(ks)
#     amplitude_modulation = amplitude_modulation_geff(ks, log10Geff, rs)
#     if log10Geff < -6:
#         amplitude_modulation = 1.0
#     exponential_damp_modulation = exponential_damping_geff(ks, log10Geff, rs)
#     return amplitude_modulation * standard_phase * exponential_damp_modulation


# def deriv_amplitude_modulation_geff(
#     ks: npt.NDArray, log10Geff: float, rs: float
# ) -> npt.NDArray:
#     """Amplitude modulation based on Geff"""
#     amplitude_modulation_der = 6 * 2.03359823e-05 * (
#         log10Geff**5
#     ) + 5 * 5.36960127e-04 * (log10Geff**4)
#     amplitude_modulation_der = (
#         amplitude_modulation_der
#         + 4 * 4.55360397e-03 * (log10Geff**3)
#         + 3 * 9.73443600e-03 * (log10Geff**2)
#     )
#     amplitude_modulation_der = (
#         amplitude_modulation_der + 2 * -5.52743545e-02 * (log10Geff) + -3.04729338e-01
#     )
#     if log10Geff < -6:
#         amplitude_modulation_der = 0.0
#     return amplitude_modulation_der


# def deriv_exponential_damping_geff(
#     ks: npt.NDArray, log10Geff: float, rs: float
# ) -> npt.NDArray:
#     """Compute the exponential damping based on Geff"""
#     exponential_damp_modulation_der = (
#         6 * 7.84726283e-06 * (log10Geff**5)
#         + 5 * 2.33592405e-04 * (log10Geff**4)
#         + 4 * 2.55941525e-03 * (log10Geff**3)
#     )
#     exponential_damp_modulation_der = (
#         exponential_damp_modulation_der
#         + 3 * 1.28825961e-02 * (log10Geff**2)
#         + 2 * 2.80788885e-02 * (log10Geff)
#     )
#     exponential_damp_modulation_der = exponential_damp_modulation_der + 1.09893067e-02
#     if log10Geff < -6:
#         exponential_damp_modulation_der = 0.0
#     return exponential_damp_modulation_der


# def derivk_geff(ks: npt.NDArray, log10Geff: float, rs: float, beta: float):
#     firstterm = (
#         deriv_amplitude_modulation_geff(ks, log10Geff, rs)
#         * fitting_formula_Baumann19(ks)
#         / rs
#         * beta
#         * (exponential_damping_geff(ks, log10Geff, rs))
#     )  # A'(Geff) * beta * f(k) / rs * exp(k rs B(Geff))
#     secondterm = (
#         amplitude_modulation_geff(ks, log10Geff, rs)
#         * deriv_exponential_damping_geff(ks, log10Geff, rs)
#         * (exponential_damping_geff(ks, log10Geff, rs))
#         * fitting_formula_Baumann19(ks)
#         * beta
#         * ks
#     )
#     # A(Geff) * B'(Geff) * exp(k rs B(Geff)) * f(k) * beta * k

#     derivk_geff = firstterm + secondterm

#     return derivk_geff
