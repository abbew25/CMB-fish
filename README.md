# Description 

This code is based off the GoFish code, but I have adapted it to compute the Fisher information for the temperature fluctuations in the CMB. This script can produce Fisher matrix forecasts for the following cosmological parameters:

$$100\theta_s$$, $$100\omega_b$$, $$\omega_{c}$$, $$\ln{(10^{10} A_s)}$$, $$n_s$$, $$\tau$$. It can also optionally do forecasts for $$\sum m_{\nu}$$, and for phase shift related parameters: $$A_{\phi}$$, which parameterizes the amplitude of the phase shift due to free-streaming standard model neutrinos, and $$G_{\mathrm{eff}}$$, the strength of neutrino self-interactions in a model with universal self-interactions, which has been included in a template for the phase shift. 


# Installing and running the code 

For usage with the cosmological boltzmann solver CAMB, this code can easily be installed and run using the makefile, just run 'make install'. 

Within the UV environment, it is simple to simply run 'uv sync' to install all the required packages for this script and run forecasts using CAMB, like

```uv run CMB_forecast_main.py config.ini``` where config.ini refers to the config file, which is used as the first command line argument. 

However when I wrote this script I wanted to be able to switch between using CLASS and CAMB in case they change the forecasts at all. Unfortunately, it is not easy to install CLASS within the uv environment. Therefore, if the user wants to run forecasts with CLASS, they should set up their own virtual environment and make sure CLASS and the install relevant packages specified in the pyproject.toml file. 

Switching between CLASS and CAMB also requires a few quick edits by the user. In the setup.py file in the src/ directory, there are references to the function ```self.run_class_CMB(pardict)``` with the object called ```CosmoResults```. These references should be commented out and replaced with  ```self.run_camb(pardict)```. 


## Config file 


## Output files 
