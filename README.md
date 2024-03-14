> [!CAUTION]
> **Warning:** Gaitlink is currently under development and not ready for production use. Do not use any of the algorithm results for actual research purposes. Most of them are not in their final state and are not properly validated yet.

# GaitLink - The Mobilise-D algorithm toolbox

A Python implementation of the Mobilise-D algorithm pipeline for gait analysis using IMU worn at the lower back.

## Installation

At the moment, the package is not available on PyPI.
To install the package use pip (or poetry) with the git repository URL

```bash
pip install "git+https://github.com/mobilise-d/gaitlink.git" --upgrade
```

You might need to set your git credentials to install the package.
If you run into problems, clone the repository and install the package locally.

```bash
git clone https://github.com/mobilise-d/gaitlink.git
cd gaitlink
pip install .
```

Or the equivalent commands of the python package manager you are using to install local dependencies.

## Usage Recommendation

The package is designed to be used in two modes:

1. Usage as a full end-to-end pipeline:

   We provide high level pipelines that take in raw sensor data and output final gait parameters on a walking bout
   level, and on various aggregation levels (e.g. per day or per week).
   These pipelines were validated as part of the Technical Validation Study of Mobilise-D and are the **recommended**
   way to obtain gait parameters according to the Mobilise-D algorithms.
   Depending on the clinical cohort and the amount of gait impairment, we recommend different pipelines.
   When using the pipelines in the recommended way, you can expect error ranges as reported in [[1]].
   Outside, this recommended use case, we cannot provide any supported evidence about the correctness of the results.

   If you are using the pipelines in this way, we recommend to cite [[1]] and [[2]] as follows:

   > Gait parameters were obtained using the Mobilise-D algorithm pipeline [[1], [2]] in its official implementation
   > provided with the gaitlink Python library version {insert version you used}.

   When appropriate, include the link to the gaitlink library as a footnote or as an "online resource" in the reference
   list.

   In general, we would like to ask you to be precise about the version of the gaitlink library you used and only
   use the term "Mobilise-D algorithm pipeline" if you used the pipelines as described in the technical validation
   study and not when you just use individual algorithms (see point 2) or use the pipelines with modified parameters.

   In the latter case, we recommend the following citation:

   > Gait parameters were obtained using an approach inspired by Mobilise-D algorithm pipeline [[1], [2]].
   > The algorithm pipeline was implemented based on {name of Pipeline class} available as part of the gaitlink Python
   > library version {insert version you used} with the following modifications:
   > {insert modifications you made}.
   
   [1]: https://doi.org/10.1038/s41598-024-51766-5
   [2]: https://doi.org/10.1186/s12984-023-01198-5
   
   ```
   [1] Kirk, C., Küderle, A., Micó-Amigo, M.E. et al. Mobilise-D insights to estimate real-world walking speed in 
   multiple conditions with a wearable device. Sci Rep 14, 1754 (2024). 
   https://doi.org/10.1038/s41598-024-51766-5
   
   [2] Micó-Amigo, M., Bonci, T., Paraschiv-Ionescu, A. et al. Assessing real-world gait with digital technology? 
   Validation, insights and recommendations from the Mobilise-D consortium. J NeuroEngineering Rehabil 20, 78 (2023). 
   https://doi.org/10.1186/s12984-023-01198-5
   ```

2. Usage of individual algorithms:

   Besides the pipelines, we also provide individual algorithms to be used independently or in custom pipelines.
   This can be helpful to build highly customised pipelines in a research context.
   But be aware that for most algorithms, we did not perform a specific validation outside the context of the official
   pipelines.
   Hence, we urge you to perform thorough validation of the algorithms in your specific use case.

   If you are using individual algorithms in this way, we recommend to cite the original papers the algorithms were
   proposed in and gaitlink as a software library.
   You can find the best references for each algorithm in the documentation of the respective algorithm.

   > Gait parameters were obtained using the {name of algorithm} algorithm [algo-citation] as implemented in the
   > gaitlink Python library version {insert version you used}.

   When appropriate, include the link to the gaitlink library as a footnote or as a "online resource" in the reference
   list.

## License and Usage of Names

The library was developed as part of the Mobilise-D project under the lead of the Friedrich-Alexander-Universität 
Erlangen-Nürnberg (FAU).
The original copyright lies with the Machine Learning and Data Analytics Lab
([MAD Lab](https://www.mad.tf.fau.de/)) at the FAU.
For any legal inquiries regarding copyright, contact 
[Björn Eskofier](https://www.mad.tf.fau.de/person/bjoern-eskofier/).
Copyright of any community contributions remains with the respective code authors.

The gaitlink library is licensed under an Apache 2.0 license.
This means it is free to use for any purpose (including commercial use), but you have to include the license text
in any distribution of the code.
See the [LICENSE](LICENSE) file for the full license text.

Please note that this software comes with no warranty, all code is provided as is.
In particular, we do not guarantee any correctness of the results, algorithmic performance or any other properties
of the software.
This software is not a medical product nor licensed for medical use.

Neither the name "Mobilise-D" nor "gaitlink" are registered trademarks.
However, we ask you to use the names appropriately when working with this software.
Ideally, we recommend to use the names as described in the usage recommendation above and not use the name 
"Mobilise-D algorithm pipeline" for any custom pipelines or pipelines with modified parameters.
If in doubt, feel free ask using the [Github issue tracker](https://github.com/mobilise-d/gaitlink/issues) or 
the Github discussions.
