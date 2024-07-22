# 2024-03-22 - Alpha Release 🎉

*Arne Küderle*

After working with a Mobilise-D internal team to rebuild the Mobilise-D algorithm, we are excited to announce the first
public alpha release of the new Mobilise-D algorithms.
This alpha and all future versions are released under an Apache 2.0 license.
This means that you can use the library for any purpose, including commercial use, as long as you include the license 
text and properly attribute the library in your work.

Just because the library is public now, does not mean that the development is done by any stretch of the imagination, 
but it does mean that we think that some of the building blocks that we have already developed are already useful for 
some members of the scientific community.

## We might kill your hamster

Still we want to be very clear, that the algorithms are not ready for production use and should not be used for any
serious research purposes. Most of the algorithms are not in their final state and none of them are properly 
re-validated on a large dataset.
In all aspects, we might update and change code without notice, which might break your code build on top of it.

## What is already there?

With this warning out of the way, let's talk about what is already there.

### Dataset Tooling

The library contains various tools to work with IMU data and data published using the 
[Mobilise-D data format](https://doi.org/10.1038/s41597-023-01930-9).
These tools provide the most standardized and reliable way to work with the data standardized using the Mobilise-D
matlab format, and we would highly recommend the use for all members of the Mobilise-D consortium and external 
researchers that are working with data that has been published in this format.

We further provide convenience tooling to work with raw aggregated data (i.e. the uncleaned DMO data) that was published
as part of the Mobilise-D CVS study.
This data is currently only available to members of the Mobilise-D consortium, but for the people that have access to
this data, we recommend the use of our loaders {py:class}`~mobgap.data.MobilisedCvsDmoDataset` to load and preprocess
this data correctly.

### Algorithms

The plan of this project is to reimplement at least all algorithms that were selected as part of the final Mobilise-D
pipelines.
So far, we are a little over halfway there.
To check our progress, have a look at [this github issue](https://github.com/mobilise-d/mobgap/issues/8) that we use
to track our progress.

The algorithms that are already implemented are ready to be tested by the community.
Note, that in many cases we expose various parameters of the algorithms to the user.
The default parameters are the ones that were used in the final Mobilise-D pipelines, but feel free to play around with
them and see if you can improve results for your specific use case.

### End-to-End Pipelines

Our goal is to provide easy to use pipelines that allow you to apply the recommended Mobilise-D algorithms to your data
with a single function call.
However, as we have not yet implemented all algorithms, we cannot provide a full end-to-end pipeline yet.

You will have to wait a little longer for that...

### Further tooling

Alongside the core algorithms, we provide various tools for processing (imu) data in general 
(see `mobgap.data_transforms` and `mobgap.utils`).
These might be helpful for your project, however, unless you are planning to use other parts of the library as well,
we would not recommend installing the library just for these tools.

## We need you!

Our main goal with this alpha release is to make it easier for members of the Mobilise-D consortium and the wider 
gait analysis community to test and get familiar with this new library.

Specifically, we are heavily interested in any feedback regarding the documentation and usability of the library.
If you find any bugs, or if you have any feature requests, please open an issue on our 
[github repository](https://github.com/mobilise-d/mobgap/issues).
For more general, questions and discussions, we would recommend using the 
[github discussions](https://github.com/mobilise-d/mobgap/discussions) board.

Of course, if you have your data available, we are also interested in any feedback regarding the correctness and 
performance the algorithms.
We might not address all issues immediately, but collect them to improve the algorithms, once we have a proper 
evaluation set up.

If you are planning to use the library/the final pipelines in a non-research context, we would also be interested in 
the unique requirements of your production environments.
What code would you need to interface with? Are you planning to use Docker? What constraints are important to you? ...

The more you tell us, the higher the chance that we can these aspects into account in the future development of the
library.

## Roadmap

This first alpha release does not mark any specific milestone in the development, but rather the decision that we want
to continue our development in the open together with the community.

Our first real milestone is the planned beta release (that coincides with the end of the Mobilise-D project).
This is planned for end of June 2024 and will include all algorithms and the full end-to-end pipelines.

Afterward, the work on a comprehensive, open and reproducible evaluation of the algorithmic performance will start.
Once this is completed, and we fixed all bugs and performance issues that surfaced during this evaluation, we will 
publish the validation procedure, the evaluation results and the first stable release of the library.

## Getting Started

We don't want to give you anything too specific here.
That would be kind of cheating.
We want that our library can be used by anyone arriving at our Github page.

So head over to our [github repository](https://github.com/mobilise-d/mobgap/) and check the README on how to get
started.
If you get stuck, let us know in the issues and discussions, so that we can improve our documentation for you and all
future users.


With that... We are looking forward to your feedback!


