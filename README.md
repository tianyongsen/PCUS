# Parameter_Calibration_for_Urban_Underlying_Surface
This repository is our codebase for PCUS: Calibration of the underlying surface parameters for urban flood using latent variables and adjoint equation. Our paper is currently being submitted for publication. We will provide more detailed guide soon.

# Basic structure
```shell
The core calsses in our code and the interaction diagram among them. 
      ------------------------>|projector|------>
     |                                          |   
|data pool|--->|celldiled|--->  |model|  ------>|loss|--->|optim|  
                       ^                                     |         
                       <-----------------<-------------------

```


# Installation
```shell
# conda create -n pcus python=3.12
# conda activate pcus  # To keep Python environments separate
git clone https://github.com/tianyongsen/PCUS.git --depth 1
pip install -r requirements.txt
```

# Usage
```shell
Inputs：
  A.txt: the rule of each link.
  dem.txt: the elevation of each cell.
  infil.txt: the infiltration of each cell.
  init_h.txt: the initial water depth in each cell.
  n.txt: the Manning' coefficient of each cell.
  rain.txt: the intensity curves of rains
  rain_map.txt: the rain selection for each cell.
  z_true.txt: the true value of the latent variables.
  z_prior.txt: the prior value of the latent variables.
  obs.txt: the observation information.    
```  
