# PDAF Instructions

This directory contains code to run PDAF with Lorenz96 from Fortran.

At the moment the instructions are for gfortran compiler in a Linux architecture,
though this may be expanded later.


## Dependencies

This code requires NetCDF and BLAS to be run.

From ubuntu these can be installed with `apt` as:
```
libblas-dev
liblapack-dev
libnetcdff-dev
```


## Building PDAF

For Linux from this directory:
```bash
make PDAF_ARCH=linux_gfortran
```


## Building the model

Open the `make.arch/linux_gfortran.h` file and comment out 
```makefile
CPP_DEFS = -DUSE_PDAF
```

and add:
```makefile
NC_LIB   = 
NC_INC   = 
```
set as the results of `nf-config -fflibs` and `nf-config -fflags` respectively.

Then navigate to the model file:
```bash
cd models/lorenz96/
```
and run:
```bash
make lorenz_96 PDAF_ARCH=linux_gfortran
```
to build the model.

This will generate the `lorenz_96` executable.

Note that you can set the environment variable to avoid specifying `PDAF_ARCH` for
every command:
```bash
export PDAF_ARCH=linux_gfortran
```

You will also need to provide the path to the FTorch build:
```bash
export FTORCH_BUILDDIR=/path/to/ftorch/build
```

## Running the model

The model can now be run for the desired number of timesteps with:
```bash
./lorenz_96 -total_steps 10
```


## Building the data assimilative model (Lorenz 96 + PDAF)

Now edit the file make.arch/linux_gfortran.h again and back to 
```bash
CPP_DEFS = -DUSE_PDAF
```
It will activate coupling of PDAF in the model code.

Navigate to the model directory:
```bash
cd models/lorenz96/
```

Now compile the Lorenz-96 model with activated coupling to PDAF. First remove the binary files from the previous compilation using
```bash
make clean PDAF_ARCH=linux_gfortran
```
  
Then build the assimilative model using
```bash
make pdaf_lorenz_96 PDAF_ARCH=linux_gfortran
```
Now you should have a executable file pdaf_lorenz_96 and can run the data assimilation

## Running the data assimilative model (Lorenz 96 + PDAF)

```bash
./pdaf_lorenz_96 -total_steps 5000 -step_null 1000 -dim_ens 30
```
So we run the model for 5000 time steps and we start assimilation at 1000 time steps for 30 ensemble members. You can change the configuration with these command line arguments or from a namelist. 
