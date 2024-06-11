"""
Loading Digital Mobility Outcome (DMO) data
===========================================

Besides raw data, mobgap also has utilities to load pre-calculated DMO data in the format published by Mobilise-D.
Specifically, this is the data of the TVS (=technical validation study) and the
individual CVS (=clinical validation study) visits.
For each CVS visit data is available as a single CSV file, which contains the DMO data for all walking bouts of all
participants.

In addition, you might have access to the weartime reports, published by McRoberts.
They can be loaded optionally together with the DMO data (see below).

To get started, the data should be organised as follows:

1. The main dmo file with a name like ``cvs-T1-wb-dmo-27-11-2023.csv``.
   The important part is that the second element (separated by ``-``) indicates the visit-id (T1, T2, ...).
2. A file that contains the mapping from the p-id to the measurement site. This file should have at least the columns
    ``Local.Participant`` and ``Participant.Site``.
3. If you are planning to load the weartime reports, you need to have a folder with the individual weartime reports and
   a "compliance report" that contains the total weartime per day.
   The file should follow the naming schema ``CVS-wear-complicance-*.xlsx`` and should be placed in the same folder as
   the weartime reports.

If the data is organised as described above, you can load the data using the :class:`MobilisedCvsDmoDataset` class.
Below, we will use some example data that is included in the mobgap package containing the data from two participants.

Loading data using these classes handles a lot of common edgecases, in particular the correct handling of timezones and
is hence, the recommended way to load the data.

We will only show loading the data without the weartime reports, as no example weartime reports are included in the
package at the moment.
"""

from mobgap.data import MobilisedCvsDmoDataset, get_example_cvs_dmo_data_path

example_data_base_path = get_example_cvs_dmo_data_path()
dmo_data_path = example_data_base_path / "cvs-T1-test_data.csv"
mapping_path = example_data_base_path / "cvs-T1-test_data_mapping.csv"

dataset = MobilisedCvsDmoDataset(
    dmo_path=dmo_data_path, site_pid_map_path=mapping_path
)
dataset

# %%
# We can access all dmo data (i.e. all individual dmos per walking bout) of the entire dataset using the following
# line.
# This might take a second, as the data is loaded from the CSV file (in particular when using the full dataset instead
# of the example data).
dataset.data


# %%
# We can also access a ``data_mask`` that represents potential data quality issues in the data.
# If the value is ``False`` the specific value of the DMO is outside expert defined thresholds.
# Depending on the analysis, you might want to exclude these values from the analysis.
# Further methods like the :class:`MobilisedAggregator` allow to pass this data mask to exclude these values correctly
# from further analysis.
dataset.data_mask

# %%
# We can see in the index that each day of the recording is listed as a separate entry in the dataset index and hence
# can be easily accessed individually.
#
dataset

# %%
single_participant = dataset.get_subset(participant_id="10004")
single_participant

# %%
# This allows to access the measurement site and timezone of the participant.
# Note, that this is usually not that important, as the class handles timezone conversions internally and provides all
# time values (e.g. the start of a walking bout) in the local time of the measurement site.
single_participant.measurement_site

# %%
single_participant.timezone
