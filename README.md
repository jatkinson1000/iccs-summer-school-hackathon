# Lorenz ‘96 data assimilation

This repository is a contribution to the 2025 ICCS summer school hackathon by
the following participants:
* Nabir Mamnun @nmamnun
* Niccolò Zanotti @niccolozanotti
* Viktoriia Hrytsyna @V-H-Reads
* Jack Atkinson @jatkinson1000
* Joe Wallwork @jwallwork23
* Sam Avis @sjavis

## Project overview

The repository uses the [PDAF](https://pdaf.awi.de) data assimilation framework
to perform data assimilation on the Lorenz ‘96 model in the context of a machine
learning based surrogate model.
Observations are generated using a Python implementation of the model.
The surrogate model is written in [PyTorch](https://pytorch.org/) and this is
ported to Fortran for use with PDAF using
[FTorch](https://cambridge-iccs.github.io/FTorch/).

## Installation

*Work in progress*

### Python dependencies

After having created a [virtual environment](https://docs.python.org/3/tutorial/venv.html), say at `.venv`, activate it with `source .venv/bin/activate` and install the python dependencies for the project with:
```shell
(.venv) pip install -e . 
```
or by preprending `uv` if using [uv](https://github.com/astral-sh/uv).

## Running the code

*Work in progress*

## Further details

*Work in progress*
