# Neuropixels visualisation

Some scripts to got from raw data to psths in the Speech task

## Installation

Install the package in a virtual environment either with poetry (prefered) or pip (+ conda for pykilosort for example)

Then, change the paths in `neuropixels_visualisation/configs.py`

## Scripts

### spikesort_session.py

All spikesorters have to be installed separatly

Set the data path and output path before running the `spikesort_session.py` script
The script will save the spikesorting output in phy format, as well as a 'report' folder with different information about the isolated clusters

### plot_psths.py

Make sure you have the correct paths in `neuropixels_visualisation/configs.py`
The data format is based on [neo](https://neo.readthedocs.io/en/latest/)
The current script plot psths for all units, but feel free to add some filter to only keep good ones (as there can be lots of noise clusters)

To filter the spikesorting output:

    * either use [phy](https://github.com/cortex-lab/phy) (the phy manual clustering results will be loaded in the neo object)
    * Or use filters on your quality metrics of choice (all saved in the spiketrains object from neo)

More information about quality metrics on [spikeinterface](https://spikeinterface.readthedocs.io/en/latest/modules/qualitymetrics.html) doc and this [Allen institute doc](https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html).
