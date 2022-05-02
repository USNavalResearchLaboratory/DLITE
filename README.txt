This distribution contains Python software for running a Deployable Low-band Ionosphere
and Transient Experiment (DLITE) array and working with the data (see this paper for
details: https://doi.org/10.1029/2021RS007298).  There are three main files:

1. dlite_npol_conf.py -- the Python/GNURadio-based correlator that runs the array;
requires GNURadio
2. dlite_squelch_conf.py -- a version of the correlator with clipping to mitigate radio
frequency interference (RFI)
3. dliteTools.py -- a Python module with functions for reading in and working with DLITE
data; requires NumPy and astropy

The two correlator scripts take a single argument, which is the name of a text file
containing parameters it needs to run.  The format of the file must look like this example:

BANDWIDTH = 100.e6/12.
NCHAN = 512
FREQ = 35.e6
TINT = 1.024
DURATION = 86164
NPOL = 2
OUTDIR = '/home/dlite/daily'
GAIN = 20
THRESH = [-56,-56,-66,-66,-53,-53,-63,-63]

The THRESH parameter is only required for dlite_squelch_conf.py and sets the clipping
threshold (in dB) per channel.

The dliteTools.py module can be installed with the setup.py script (i.e., python setup.py
install), or by simply copying it into a directory that is in your PYTHONPATH.
