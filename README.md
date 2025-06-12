# Environment and Uncertainty-Aware Robust Motion Planning for Autonomous Driving
This is the Python code for the article
```
@article{zhou2025robust,
  title={Robust motion planning for autonomous vehicles based on environment and uncertainty-aware reachability prediction},
  author={Zhou, Jian and Gao, Yulong and Olofsson, Bj{\"o}rn and Frisk, Erik},
  journal={Control Engineering Practice},
  volume={160},
  pages={106319},
  year={2025},
  publisher={Elsevier}
}
```

Jian Zhou and Erik Frisk are with the Department of Electrical Engineering, Linköping University, Sweden. Yulong Gao is with the Department of Electrical and Electronic Engineering, Imperial College London, UK. Björn Olofsson is with both the Department of Automatic Control, Lund
University, Sweden, and the Department of Electrical Engineering, Linköping University, Sweden.

For any questions, feel free to contact me at: jian.zhou@liu.se or zjzb1212@qq.com
## Packages for running the code
To run the code you need to install the following key packages:

**Pytope**: https://pypi.org/project/pytope/

**CasADi**: https://web.casadi.org/

**HSL Solver**: https://licences.stfc.ac.uk/product/coin-hsl

**cvxpy**：https://www.cvxpy.org/

**mosek**: https://www.mosek.com/

Note: 

(1). Installing the HSL package can be a bit comprehensive, but the solvers just speed up the solutions. You can comment out the places where the HSL solver is used (i.e., comment out the command "ipopt.linear_solver": "ma57"), and just use the default linear solver of ipopt in CasADi.

(2). The package "Pytope" is used very few in the development. However, many people have reported issues with the installation. If you do not manage to install it, please email me then I will provide a solution that avoids this package (I uses this package in the development just for convenience since I already installed it).

(3). The solver "mosek" may need a license, and you need to apply for it.

## Introduction to the files
In folder `1_roundabout_simulation`, you will find the implementation for comparing the proposed method, the deterministic method, the robust method, and the scenario method, in a single scenario. 

1. The file `main.ipynb` defines the main file for running the code; `Planner_P` is the method for the proposed approach, `Planner_D` is the method for the deterministic method, `Planner_R` is the method for the robust method, and `Planner_S` is the method for the scenario method. The surrounding vehicle (SV) is modeled by `ModelingSVTrue.py` with a nonlinear model predictive control (NMPC). The file `Initial_Set_Estimation.py` contains the function for learning the intended control set of the SV at the initial time step.

2. The file `track_para.npy` contains the parameters for defining the geometry of the roundabout, there is no need to access and modify it.

3. After running the file `main.ipynb`, the data will be automatically saved in the folder `1_Data_Case_1`, where you can analyze the result by running `Analysis.ipynb`. In folder `1_Data_Case_1`, there is a function `Check_Safety.py` for computing the distance between the ego vehicle (EV) and the SV, as well as the distance to the round boundary, for assessing the safety of the planner.

4. The code for producing the animations has been removed as a result of possible compilations. However, the provided code can fully recover the results presented in the paper.


In folder `1_roundabout_simulation`, you will find the implementation for comparing the proposed method in a rounD dataset scenario [1].

1. The information of the EV and SVs has been saved in the files "EV_Data.npy", "SV0_Data.npy", "SV1_Data.npy", and "SV2_Data.npy". These files include the motion information (e.g., trajectory, velocity, etc.) from the dataset.

2. `Planner_P.npy` defines the proposed motion planner. The file `Initial_Set_Estimation.py` contains the function for learning the intended control set of the surrounding vehicle at the initial time step. The function `Check_Safety.py` computes the distance between the EV and the SVs, as well as the distance to the round boundary, for assessing the safety of the planner.

3. Run `main.ipynb` you will reproduce the results presented in the paper.

## Reference
Krajewski, Robert, et al. "The rounD dataset: A drone dataset of road user trajectories at roundabouts in Germany." IEEE International Conference on Intelligent Transportation Systems (ITSC), 2020.








