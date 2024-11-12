.. _evaluation_guide:

Evaluation - Work in Progress
=============================

This guide gives an overview on how we approach evaluation in Mobgap.
With this you should get a better understanding on what tools to use and also on how to structure your evaluation
process.

This guide is meant for developers looking to work on the evaluation part of Mobgap or updating our public evaluation
results and for users looking to run their own validation on their own dataset or algorithms using the mobgap tooling.

In most cases, people are used to doing evaluation a two-step process:
First, we ran our algorithm against all of the data we have and obtain some results.
These results are saved in a file and we can look at them later.
Second, we take the results in our favorite statistical tool (e.g. R, Python, Excel) and analyze them.
This usually entails obtaining reference information through some other means and then calculating some metrics.

This approach works great (actually has some clear advantages, which we will get into later), but breaks down as soon
as we are dealing with any form of optimizations or parameter tuning as part of your validation process.
Then suddenly, we need train and test splits (or ideally cross-validation) and we need to keep track of a bunch of
metadata (e.g. which parameters did we use, which data did we use for training, etc.) alongside our results.

Most importantly, when doing any form of parameter tuning, we need to do validation on the fly, as we need to decide
how good a parameter setting is while we are doing our grid/random search.
This means we need to structure the way on how to obtain our reference information, and create at least a function
that can calculate the metrics we are interested in given results from an arbitrary subset of our dataset.

I have seen this done differently, as in running all parameter combinations on the full dataset, saving all results,
then "simulating" a train-test split/cross-validation during the manual analysis, and then finally creating a new script
that runs the best parameter setting on the full dataset.
This is how you get train-test leakage... and just PAIN. Trust me!

So for optimization algorithms, we need a structure, where we can calculate at least our core metrics on the fly.
This makes it possible to easily run gridsearches within cross-validations and cross-validations within your
gridsearches and whatever else you might come up with.

But it comes with the cost of having to build the code components that allow you to do this.
Yes `tpcp` and `mobgap` already handle a lot of this, but I wont pretend, that this is a trivial task.
That said, for when optimization is a core part of your evaluation process, I don't see a way around it.

This raises the question, should I attempt to do something if I am not planning to do any optimization?
In many cases I would say yes!
The reality is, that just because your current algorithm does not require any optimization (and are you sure a little
GridSearch wouldn't help?), it does not mean that you never want to compare to a machine learning algorithm that does.
And then we need to be ready to have a structure that allows us to run our "normal" and the machine learning algorithm
through the same train test splits and cross-validations to make sure we are actually comparing them on the same data.

Yes, we could do this on results that we calculated once and then simulated a train-test split on.
But again, this is pain and you WILL make mistakes.

With that, we decided to follow to core design principles for our evaluation tooling:

- Assume the most complicated case: We assume that we will need to do some form of optimization or parameter tuning
  as part of our evaluation process. This means we structure error calculation to be happening on the fly.
- Avoid pre-calculated results as much as possible: Instead, we make sure that re-running algorithms is easy and fast
  enough, that you can rerun your entire evaluation, if you decide to change the process.
  We avoid using pre-calculated results as much as possible.
  So no manual, doctoring and duplicating logic between your evaluation and your optimization.
  If you need to change the datasets, the optimization parameters or evaluation metrics, don't be afraid to scrap
  everything and rerun it.
  As a nice side-effect, you can always be confident, that your entire evaluation process is reproducible.


High level Tooling
------------------

With the above motivation in the back of our heads, let's have a look at the tools we have available to us.
At a high level, we provide two classes for evaluation:

- :class:`mobgap.utils.evaluation.Evaluation`: This is an algorithm that takes a dataset, a pipeline, and a scoring
  function and runs the pipeline on the dataset and calculates the scores on a per datapoint (1 datapoint = 1 row in
  the dataset) or across the entire dataset (this depends on the scoring function).
  It outputs aggregated scores, scores per datapoint, and even the raw results of the pipeline.
  This is the high-level wrapper around any standardized evaluation or benchmarking process.
  If you use the same instance of this class for multiple different algorithms, you can be sure that the results are
  calculated in exactly the same way.
- :class:`mobgap.utils.evaluation.EvaluationCV`: Just like `Evaluation`, but it runs a cross-validation.
  So all the complicated stuff of splitting the data, running the pipeline on the different splits, is abstracted away.
  It takes a dataset, a Optimizable wrapping a pipeline (more about this below), a scoring function, and
  cross-validation config.
  Like `Evaluation`, it outputs aggregated scores, scores per datapoint, and even the raw results of the pipeline, just
  additionally grouped by CV fold.

From the first look, you might agree, that the ``Evaluation`` class sounds nice, but does not seem to abstract away
much.
However, the ``EvaluationCV`` sounds much more useful, as all cross-validation stuff is complicated.
The magic is, that if you design your evaluation process around the ``Evaluation`` class, switching to
``EvaluationCV``, when you want to start optimizing parameters or include a machine learning algorithm, is trivial.


Low level Tooling
-----------------
Independent of how you structure your evaluation process, you will likely need to calculate some metrics.
This metrics can become quite complex for some of the algorithmic blocks.
For example, for the initial contact detection, you need to find the closest contact in the ground truth to a predicted
contact, while ensuring that you only get one prediction per ground truth contact.
This is not easily implemented.
Therefore, we provide metric functions for most algorithmic blocks.
They can be found in the `{blockname}.evaluation` module (e.g. ``mobgap.initial_contacts.evaluation``) and their are
explained in the accompanied example that can be found in the documentation (e.g. :ref:`icd_evaluation`).

These low level building blocks can be used to build a custom evaluation either on existing results, or results you
calculate with mobgap algorithms.
Further, these functions are also the building blocks we need to build our own scoring functions for the `Evaluation`
and `EvaluationCV` classes.

Building a custom evaluation
----------------------------

This explains the required building blocks to build a evaluating script that uses either ``Evaluation`` or
``EvaluationCV``.
For many of the building blocks, we already have implementations for the standard use-cases.
However, we want to go a little deeper here to explain how you would customize those.

The core building blocks for building a custom evaluation are:

- A dataset: Likely a subclass of :class:`mobgap.data.base.BaseGaitDatasetWithReference`.
  This is the data you want to run your algorithm on.
  We will not go into detail here, but have a look at this :ref:`guide <custom_datasets>` for more information.
- A pipeline: A pipeline wraps one or multiple algorithms to form the bridge between a single datapoint and the
  algorithmic blocks.
  It takes care of routing the correct information to the correct algorithmic block and structuring the results, so
  that the scoring function can calculate the metrics.
  You can learn more about the fundamental idea of pipelines in the :ref:`pipeline guide
  <https://tpcp.readthedocs.io/en/v2.0.0/guides/algorithms_pipelines_datasets.html#pipelines>`.
- A scorer: In its simplest form this is a function, that can run your pipeline on a datapoint (one row in your dataset)
  parse the results and calculate all metrics that you are interested in.
  You will see that this can get quite complex when you consider different ways you might want to aggregate the results.
  Hence, we will spend the majority of this guide on this topic.

Dataset
+++++++
As said not much detail here.
Head over to the :ref:`custom_datasets` guide for more information.
However, a couple of important points to be mentioned here.

TODO: FINISH THIS

We likely still want to do a comprehensive comparison at the end, recaluclating some metrics across certain subgroups
and generate some final plots and tables.
This presents us with a challenge.
You will likely need to duplicate the logic of calculating your metrics in two places.
And you better damn make sure that you are using the same logic in both places.
Otherwise, your optimized algorithm might not optimizing for what you are evaluating later on.

