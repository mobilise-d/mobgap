.. _own_algorithm:

===========================
Creating your own algorithm
===========================

While mobgap's main goal is to provide a set of algorithms and pipelines that work out of the box,
we want to give you the possibility to create your own algorithms and pipelines.
This can be done not just by selecting which combinations of algorithms to use and modifying their parameters, but also
by creating your own algorithm or extend existing once.

In this guide we will give a rough overview of how this process works.
You will find additional information in the :ref:`developer documentation for mobgap<development>`  (though some information there might not
apply to your project) and the general documentation of the `tpcp` package mobgap is based on
(`in particular this example <https://tpcp.readthedocs.io/en/latest/auto_examples/algorithms/_01_algorithms_qrs_detection.html>`_
and `this guide <https://tpcp.readthedocs.io/en/latest/guides/general_concepts.html>`_).

Getting started
---------------
To create new algorithms, you **don't** need to fork or modify the mobgap repository!
Just create a new Python project and install mobgap as a dependency.

Then you can create your own algorithms by either following the mobgap API spec or by inheriting from existing classes.

In case you implemented something that you think might be useful for others, please consider contributing it to mobgap
directly.
For this checkout the :ref:`contributing guide <contributing>`.

Adapting an existing algorithm
------------------------------

Before deciding to create your own algorithm, you should check if you can adapt an existing one.
As each algorithm is a class, you can simply inherit from it and overwrite the methods you want to change.

We use this pattern all the time in mobgap.
Typically, each group of algorithm has a base class (which can be found in the ``base.py`` file in the respective
subfolder) that describes the basic interface.
When all algorithm in a group have the same interface, it is easy to switch between them.

Let's say
If you would like to implement your own version of the `BarthDtw` algorithm, you could either subclass it directly
or inherit from the same baseclasses and mixins to create a fully custom algorithm.
This depends on how much of the existing functionality you want to keep.

As an example below you can see how to create a custom version of the `BarthDtw` algorithm with a different
postprocessing-check implemented.
For this we only need to overwrite the `_post_postprocess_check` method of the `BarthDtw` class.

.. code-block:: python

    from mobgap.stride_segmentation import BarthDtw, BaseStrideSegmentation, BaseDtw

    class MyBarthDtw(BarthDtw):
        def _post_postprocess_check(self, matches_start_end):
            super()._post_postprocess_check(matches_start_end)
            # do something else here

    # use the new algorithm
    algorithm = MyBarthDtw()

You can see that this is relatively easy to do.
In general, we tried to split the functionality of algorithms into small methods, so that you can easily overwrite
individual parts.

In case you come across a situation where you need to change the functionality of an algorithm, but you cannot find a
method that you can easily overwrite, please open an issue on github.
We are very open to split up larger methods even further, if it helps to make the algorithms more extendable.

Creating a new algorithm for an existing algorithm type
-------------------------------------------------------

Sometimes you might want to create a completely new algorithm.
This is also possible, but requires a bit more work.

In this case, start with the base class of the algorithm type you want to implement.
This will give you a set of result attributes (denoted by the trailing underscore) and methods that you need to
implement.

For example, if you want to create a new stride segmentation algorithm, you should start with the
:class:`~mobgap.base.BaseStrideSegmentation` class.
In its implementation you will see that all stride segmentation algorithms should implement a result attribute called
`stride_list_` and a method called `segment`.
After the call of the `segment` method, the `stride_list_` attribute should contain the result of the segmentation.

These are the minimal requirements your custom algorithm needs to fulfill.
To better understand what is expected of the `segment` method and the structure of the result, you can look at the
implementation of the existing algorithms.

Note, that you need to strictly adhere to the interface of the base class (e.g. don't add additional inputs to the
action method).
Otherwise, certain higher level functions might not work with your algorithm.

Further, you should make sure that your algorithm is a valid `tpcp.Algorithm` class.
This means, you should not modify any parameters of the class in the `__init__` method and your action method should
must not modify any of your parameters at should return the modified algorithm object.
The latter can be enforced by using the :func:`~tpcp.make_action_safe` decorator on your action method.

Note, that json serialization of your algorithm might not work, depending on the parameters you use.
In this case, please open an issue on github with your usecase.

Creating a new algorithm type
-----------------------------

If you want to extend mobgap with an entirely new algorithm type, we recommend creating a new base class inheriting
from :class:`tpcp.Algorithm`.
This baseclass should specify the minimal interface that all algorithms of this type should implement.

This includes the name of the action method provided by the `_action_method` attribute and a implementation of the
action method that simply raises a `NotImplementedError`.
You can further add the type definitions for some of the expected result objects.

However, keep in mind that Python does not explicitly enforce the interface of classes.
So your baseclass acts more as a guidance and to help your IDE to provide better autocompletion.

Algorithms can then inherit from this baseclass and implement the action method and other methods as needed.

==========================
Creating your own pipeline
==========================

If you understand how to create your own algorithm, creating your own pipeline should be easy.
There 