# OptimizationDER

Repository files for the Thesis Methodology for the formulation and solution of optimization problems regarding the operation of distribution networks with battery storage systems by Diego Mendoza.

The folder AlgorithmsPerformance includes some codes to analyzes performance in PF and OPF problems.

PF folder contains original power flow optimizations (Single-Period)

PFN folder contains power flow optimizations implementing mean demand coefficients for Bogotá, Jamundí and Popayan (Single-Period)

MPF folder contains power flow optimizations implementing hourly mean demand coefficients for Bogotá, Jamundí and Popayan (Multi-Period)

OPF-PV folder contains single-period optimal PV allocation problems implementing mean demand coefficients for Bogotá, Jamundí and Popayan

OPF-PV-D folder contains deterministic, multi-period, optimal PV allocation problems implementing mean demand coefficients for Bogotá, Jamundí and Popayan.

OPF-PV-S folder contains stochastic, multi-period, optimal PV allocation problems.

OPF-BESS-I folder contains deterministic, multi-period, optimal operation of ideal MBESS problems.

OPF-BESS-IS folder contains stochastic, multi-period, optimal operation of ideal MBESS problems.

OPF-BESS-E folder contains deterministic, multi-period, optimal operation of non-ideal MBESS problems.

OPF-BESS-ES folder contains stochastic, multi-period, optimal operation of non-ideal MBESS problems.

These repository not only includes every code implementatioon of every formulation appearing in the manuscript, but also results in .csv files and codes to process these results into LaTex tables, and .txt files with the tables.

The demand and irradiance data is not publicly available. To request this information contact PhD Javier Rosero (jaroserog@unal.edu.co). Therefore, most optimization problems available cannot be run if kept unmodified to read different datafiles.

To run optimizations, it is required a Python distribution with CVXPY, MOSEK, GUROBI, AMPL with IPOPT, BONMIN, HIGHS and SCIP and MATLAB. 
