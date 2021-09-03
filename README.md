# TimeResolved

Wrapper of Bayesian Maximum Entropy code () to perform time-dependent data reweighting. Currently implemented for SAXS data only. 

## Requirements
1. Python > 3.x;
2. Pandas;
3. Matplotlib; 
4. [Fplanck](https://github.com/johnaparker/fplanck) for numerical solution of the Fokker-Planck equation;
5. [Kneed](https://github.com/arvkevi/kneed) for automatic knee point location in reweighting curves;

## Usage
Class allocation requires: the path to the directory with experimental SAXS intensities, the path to an MD or Metadynamics simulation, the path to a topology file, the path to a BME file containing the SAXS intensities from the simulation, a file with the metadynamics weights (no file required for MD), the global name of intensity files. 
