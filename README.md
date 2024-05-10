# Environment and Uncertainty-Aware Robust Motion Planning for Autonomous Driving
This is the Python code for the article
```
@article{zhou2024environment,
  title={Environment and Uncertainty-Aware Robust Motion Planning for Autonomous Driving},
  author={Zhou, Jian and Gao, Yulong and Olofsson, Bj{\"o}rn and Frisk, Erik},
  year={2024}}
```

Jian Zhou and Erik Frisk are with the Department of Electrical Engineering, Linköping University, Sweden. Yulong Gao is with the Department of Electrical and Electronic Engineering, Imperial College London, UK. Björn Olofsson is with both the Department of Automatic Control, Lund
University, Sweden, and the Department of Electrical Engineering, Linköping University, Sweden.

For any questions, feel free to contact me at: jian.zhou@liu.se or zjzb1212@qq.com
## Packages for running the code
To run the code you need to install the following key packages:

**Pytope**: https://pypi.org/project/pytope/

**CasADi**: https://web.casadi.org/

**HSL Solver**: https://licences.stfc.ac.uk/product/coin-hsl

Note: Installing the HSL package can be a bit comprehensive, but the solvers just speed up the solutions. You can comment out the places where the HSL solver is used (i.e., comment out the command "ipopt.linear_solver": "ma57"), and just use the default linear solver of ipopt in CasADi.

## Introduction to the files
In folder `1_roundabout_simulation`, you will find the implementation for comparing the proposed method, the deterministic method, the robust method, and the scenario method, in a single scenario. 

1. The file `main.ipynb` defines the main file for running the code; `Planner_P` is the method for the proposed approach, `Planner_D` is the method for the deterministic method, `Planner_R` is the method for the robust method, and `Planner_S` is the method for the scenario method. The surrounding vehicle is modeled by `ModelingSVTrue.py` with an NMPC controller. The file `Initial_Set_Estimation.py` contains the function for learning the intended control set of the surrounding vehicle at the initial time step.

2. The file `track_para.npy` contains the parameters for defining the geometry of the roundabout, there is no need to access and modify it.

3. After running the file `main.ipynb`, the data will be automatically saved in the folder `1_Data_Case_1`, where you can analyze the result by running `Analysis.ipynb`. In folder `1_Data_Case_1`, there is a function `Check_Safety.py` for computing the distance between the ego vehicle and the surrounding vehicle, as well as the distance to the round boundary, for assessing the safety of the planner.


The code for other case studies will be included soon.








