# Description 

This code is based off the GoFish code, but I have adapted it to compute the Fisher information for the temperature fluctuations in the CMB. This script can produce Fisher matrix forecasts for the following cosmological parameters:

$$100\theta_s$$, $$A_{\phi}$$ (parameterizes the amplitude of the phase shift due to free-streaming standard model neutrinos), $$100\omega_b$$, $$\omega_{c}$$, $$\ln{(10^{10} A_s)}$$, $$n_s$$, $$\tau$$. It can also optionally do forecasts for $$\sum m_{\nu}$$, and $$G_{\mathrm{eff}}$$, the strength of neutrino self-interactions in a model with universal self-interactions, which has been included in a template for the phase shift. 

# Installing and running the code 

For usage with the cosmological boltzmann solver CAMB, this code can easily be installed and run using the makefile, just run 'make install'. 

Within the UV environment, it is simple to simply run 'uv sync' to install all the required packages for this script and run forecasts using CAMB, like

```uv run CMB_forecast_main.py config.ini``` where config.ini refers to the config file, which is used as the first command line argument. 

However when I wrote this script I wanted to be able to switch between using CLASS and CAMB in case they change the forecasts at all. Unfortunately, it is not easy to install CLASS within the uv environment. Therefore, if the user wants to run forecasts with CLASS, they should set up their own virtual environment and make sure CLASS and the install relevant packages specified in the pyproject.toml file. 

Switching between CLASS and CAMB also requires a few quick edits by the user. In the setup.py file in the src/ directory, there are references to the function ```self.run_class_CMB(pardict)``` with the object called ```CosmoResults```. These references should be commented out and replaced with  ```self.run_camb(pardict)``` (or vice versa). 


## Config file 

The config file allows the user to specify the cosmology and also set limits on what information is included in the Fisher forecasts (see example files in config/). One can specify the maximum and minimum ells and which power spectra to include (TT is always included, and optionally EE, TE and BB). The sky area is also set in the config file. 

The user can also specify whether to include Geff or Mnu as free parameters here (```geff_fixed = False```, ```neutrino_mass_fixed = True```). 

The parameter $$A_{\phi}$$ should be set to zero for the standard model neutrino phase shift (corresponding to $$N_{\mathrm{eff}} = 3.044$$. For standard model neutrinos, set $$\log_{10}{(G_{\mathrm{eff}})} = -12$$. 

If the user sets ```geff_fixed = False``` (thus including it as a parameter in the forecasts), one may find that for very small values of $$\log_{10}{(G_{\mathrm{eff}})} \leq -3$$, this will result in a singular fisher matrix. There is little to no information on $$\log_{10}{(G_{\mathrm{eff}})} \leq -3$$ from the phase shift alone for small values which is why this occurrs. 

The config file also specifies the prefix for output files (the covariance matrix) to be saved for the user to access later. 

In the source file, if the user sets ```noise_Planck = False``` the forecasts will reflect those of a CV limited survey. If you want to include/not include CMB lensing, edit the output for the C_ells in the functions ```self.run_class_CMB(pardict)```/ ```self.run_CAMB(pardict)``` in the ```CosmoResults``` object within the setup.py file in src/. At present, I haven't set up the code to be able to read in a file that specifies the noise of the CMB experiment. However, if one sets ```Plank_noise = True``` in the config file, it will assume the noise properties specified in Font-Ribera et al 2014 for Planck specific forecasts for the TT, EE and TE power spectra. 

## Other important info.

In the main source file, ```CMB_forecast_main.py```, the step sizes for numerical derivatives of the C_ells w.r.t. cosmological parameters are specified. The forecasts are very sensitive to these step sizes and can become unstable for step sizes that are not robust.


