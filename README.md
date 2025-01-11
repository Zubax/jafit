# jafit

Jiles-Atherton system identification tool: Given a BH curve, finds the Jiles-Atherton model coefficients.
The JA formulation used here follows that of COMSOL Multiphysics,
allowing one to simulate (de)magnetization of anisotropic ferromagnetics there.

<img src="doc/image.png" width="800" alt="">

## Usage

The tool accepts a tab-separated file encoding the demagnetization BH curve as a command line argument,
where the first column contains the magnetic field strength H in ampere/meter,
and the second column contains the magnetic flux density B in tesla.
The rows should be ordered by increasing H.
The first row may or may not be the header row.

Ideally, the reference BH curve should cover the entire range from negative to positive (near-)saturation;
providing only the third quadrant is not sufficient for full system identification.

Install the package before using it:

```shell
pip install .
```

Derive parameters for a given BH curve as shown below.
Optionally, you can provide the initial guess for (some of) the coefficients like `c_r=0.07`, etc.
Be sure to launch the tool from a dedicated directory because it may generate a lot of intermediate output files.
All existing outputs in the current working directory are removed at startup.

```shell
jafit data/bh-lng37.tab
```

Solve the JA equation with the given coefficients:

```shell
jafit c_r=0.07 M_s=1578608 a=29639 k_p=96544 alpha=0.046
```

## Development

To run tests locally, simply say `nox`.

If you want to run PyTest only, you may want to `export NUMBA_DISABLE_JIT=1` beforehand, or uninstall Numba.

To profile, go like: `python3 -m cProfile -o out.prof -m jafit ../data/bh-lng37.tab`.
Then you can use `flameprof` to visualize the collected data.
