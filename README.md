# jafit

[![Discuss - on Zubax Forum](https://img.shields.io/static/v1?label=Discuss&message=on+Zubax+Forum&color=ff0000)](https://forum.zubax.com)

Jiles-Atherton system identification tool: Given a hysteresis loop, finds the Jiles-Atherton model coefficients.
Supports several JA model definitions.

<img src="image.png" width="800" alt="">

## Usage

### Install the tool

```shell
git clone https://github.com/Zubax/jafit
cd jafit
pip install .
```

The tool works on GNU/Linux and Windows. Probably also on macOS, but YMMV.

### Solve the JA equation

```shell
# Coefficients from the default COMSOL Jiles-Atherton material model:
jafit model=venk  c_r=0.1 M_s=1.6e6 a=560 k_p=1200 alpha=0.0007

# Coefficients from the Altair Flux example B(H) curve:
jafit model=venk  H_amp_max=1111  c_r=0.2107788 M_s=1306755.22 a=108.694943 k_p=177.625645 alpha=0.000294224757

# From "Modeling of permanent magnets: Interpretation of parameters obtained from the Jiles–Atherton hysteresis model":
jafit model=orig  H_amp_min=4774648 H_amp_max=4774648  c_r=0.885 M_s=1080000 a=1107718.3824 k_p=702271.17275 alpha=3.168
```

If the H amplitude is set manually and is insufficient to reach saturation,
the resulting hysteresis loop will be a minor loop.

### Find JA coefficients for a given reference B(H) curve

The fitting problem may take multiple hours to solve, depending on the curve shape and the performance of your computer.
Intermediate results and logs will be stored in the current working directory,
so it may be a good idea to create a dedicated directory for this purpose.

```shell
# Reproduce the model fitting utility from Altair Flux:
jafit model=venk ref="data/Altair_Flux_HystereticExample.csv"

# Find coefficients for isotropic AlNiCo 5:
jafit model=venk ref="data/B(H).AlNiCo_5.tab"
```

The curve file contains two columns: H \[A/m\] and B \[T\], either tab- or comma-separated.
The first row may or may not be the header row.

The reference curve may be either the entire hysteresis loop, or any part of it;
e.g., only a part of the descending branch.
If a full loop is provided, then that loop doesn't need to be the major loop;
the tool will simply use the H amplitude seen in the reference loop for solving the JA equation.

If the reference curve is only a part of the hysteresis loop,
then the tool will use simple heuristics to guess the reasonable H amplitude for solving the JA equation,
assuming that the loop is the major loop (i.e., it reaches saturation).
In this case, it is recommended to specify the `H_amp_max` parameter explicitly instead of relying on heuristics.

Optionally, you can provide the initial guess for (some of) the coefficients: `c_r`, `M_s`, `a`, `k_p`, `alpha`.

### Helpful tips

For fetching the (approximate) data points from a third-party plot,
such as from a published paper or a material datasheet,
consider using [`trace_image.py`](https://gist.github.com/pavel-kirienko/0fcd509cd1d7c6dc2651981510badb99).

For the benefit of all mankind, please only use SI units. To convert data from a non-SI source:

- $1 \ \text{oersted} \approx 79.57747 \frac{\text{A}}{\text{m}}$
- $1 \ \frac{\text{emu}}{\text{cm}^3} = 10^3 \frac{\text{A}}{\text{m}}$

For more, refer to `papers/magnetic_units.pdf`.

## Development

To run verification locally, simply say `nox`.

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
The setup is used to obtain the J(H) curve and ascertain that it matches the predictions made by the tool.

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
