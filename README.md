# jafit

[![Discuss - on Zubax Forum](https://img.shields.io/static/v1?label=Discuss&message=on+Zubax+Forum&color=ff0000)](https://forum.zubax.com)

Jiles-Atherton system identification tool: Given a BH curve, finds the Jiles-Atherton model coefficients.

<img src="image.png" width="800" alt="">

## Usage

The tool accepts a tab-separated (TSV) or comma-separated (CSV) file encoding the reference BH curve
as a command line argument (the type of file is auto-detected).
The first column contains the magnetic field strength H \[ampere/meter],
and the second column contains the magnetic flux density B \[tesla].
The file may contain either the entire major hysteresis loop, or any part of it;
e.g., only a part of the descending branch.
The first row may or may not be the header row.

Install the package before using it:

```shell
pip install .
```

Derive parameters for a given BH curve as shown below.
Be sure to launch the tool from a dedicated directory because it may generate a lot of intermediate output files.
All pre-existing outputs in the current working directory are removed at startup.

```shell
jafit "data/B(H).AlNiCo_5.tab"
```

Optionally, you can provide the initial guess for (some of) the coefficients:

```shell
jafit "data/B(H).LNG37.ansys.tab" c_r=0.1 M_s=1.6e6 a=560 k_p=1200 alpha=0.0007
```

Solve the JA equation with the given coefficients without system identification:

```shell
jafit c_r=0.1 M_s=1.6e6 a=560 k_p=1200 alpha=0.0007
```

Add `H_max=30e3` to manually limit the maximum H-field strength to 30 kA/m instead of relying on heuristics.
Note that if the provided value is insufficient to reach saturation, the resulting hysteresis loop will be a minor loop!

## Helpful tips

For fetching the (approximate) data points from a third-party plot,
such as from a published paper or a material datasheet,
consider using [`trace_image.py`](https://gist.github.com/pavel-kirienko/0fcd509cd1d7c6dc2651981510badb99).

For the benefit of all mankind, please only use SI units. To convert data from a non-SI source:

- $1 \ \text{oersted} \approx 79.57747 \frac{\text{A}}{\text{m}}$
- $1 \ \frac{\text{emu}}{\text{cm}^3} = 10^3 \frac{\text{A}}{\text{m}}$

For more, refer to `papers/magnetic_units.pdf`.

## Development

To run tests locally, simply say `nox`.

If you want to run PyTest only, you may want to `export NUMBA_DISABLE_JIT=1` beforehand, or uninstall Numba.

To profile, go like: `python3 -m cProfile -o out.prof -m jafit ../data/bh-lng37.tab`.
Then you can use `flameprof` to visualize the collected data.

To evaluate the optimizer behaviors quickly, run the script in fast mode with `fast=1`.
This may render the results inaccurate, but it will be much faster.

## Validation

There is a COMSOL model in the `validation` directory that contains a bored steel cylinder with a copper wire passing
along its axis.
The wire carries a 1 Hz magnetizing current whose amplitude is chosen to be just high enough to push the
cylinder material into saturation, while the frequency is chosen to be low to avoid eddy currents.
The setup is used to obtain the BH curve and ascertain that it matches the predictions made by the tool.

<img src="validation/B(t).gif" width="600px" alt="">

To make the prediction, run the tool specifying the JA model coefficients copied from the material properties
assigned to the cylinder in the COMSOL model,
note the predicted $H_c$, $B_r$, and $BH_\text{max}$,
and compare them against the values seen in the COMSOL model.

### Specimen A

```shell
jafit c_r=0.1 M_s=1.6e6 a=560 k_p=1200 alpha=0.0007
```

<img src="validation/H(t),M(t),B(t) c_r=0.1 M_s=1.6e6 a=560 k_p=1200 alpha=0.0007.png" width="400px" alt=""><img src="validation/hysteresis c_r=0.1 M_s=1.6e6 a=560 k_p=1200 alpha=0.0007.png" width="400px" alt="">

### Specimen B

```shell
jafit c_r=0.8 M_s=1.6e6 a=56000 k_p=50000 alpha=0.001
```

<img src="validation/H(t),M(t),B(t) c_r=0.8 M_s=1.6e6 a=56000 k_p=50000 alpha=0.001.png" width="400px" alt=""><img src="validation/hysteresis c_r=0.8 M_s=1.6e6 a=56000 k_p=50000 alpha=0.001.png" width="400px" alt="">

### Specimen C

```shell
jafit c_r=0.5 M_s=1.6e6 a=56000 k_p=50000 alpha=0.001
```

<img src="validation/hysteresis c_r=0.5 M_s=1.6e6 a=56000 k_p=50000 alpha=0.001.png" width="400px" alt="">  

### Specimen D

```shell
jafit c_r=0.1 M_s=1191941.07155 a=65253 k_p=85677 alpha=0.19
```

<img src="validation/hysteresis c_r=0.1 M_s=1191941.07155 a=65253 k_p=85677 alpha=0.19.png" width="400px" alt="">  
