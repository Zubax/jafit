# jafit

Jiles-Atherton system identification tool: Given a BH curve, finds the Jiles-Atherton model coefficients.
The JA formulation used here follows that of COMSOL Multiphysics,
allowing one to simulate (de)magnetization of anisotropic ferromagnetics there.

## Usage

The script accepts a tab-separated file encoding the second quadrant of the BH curve via stdin,
where the first column contains the magnetic field strength H in ampere/meter,
and the second column contains the magnetic flux density B in tesla.
The first row may or may not be the header row.

Install the dependencies before the first invocation: `pip install -r requirements.txt`;
then run:

```bash
cat test-data/bh-lng37.tab | python3 ./jafit.py
```

## Development

To run tests locally, simply say `nox`.
