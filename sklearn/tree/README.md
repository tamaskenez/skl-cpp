directory copied from this commit: ea42c55b7be9f9c

The interesting original files contained the following classes:

tree.py
    class BaseDecisionTree(six.with_metaclass(ABCMeta, BaseEstimator,_LearntSelectorMixin)):
    class DecisionTreeClassifier(BaseDecisionTree, ClassifierMixin):
    class DecisionTreeRegressor(BaseDecisionTree, RegressorMixin):
    class ExtraTreeClassifier(DecisionTreeClassifier):
    class ExtraTreeRegressor(DecisionTreeRegressor):

_tree.pyx
    cdef class Criterion
    cdef class ClassificationCriterion(Criterion):

    cdef class Entropy(ClassificationCriterion):
    cdef class Gini(ClassificationCriterion):
    cdef class RegressionCriterion(Criterion):

    cdef class MSE(RegressionCriterion):
    cdef class FriedmanMSE(MSE):
    cdef class Splitter:
    cdef class BaseDenseSplitter(Splitter):
    cdef class BestSplitter(BaseDenseSplitter):

    cdef class RandomSplitter(BaseDenseSplitter):
    cdef class PresortBestSplitter(BaseDenseSplitter):

    cdef class BaseSparseSplitter(Splitter):
    cdef class BestSparseSplitter(BaseSparseSplitter):

    cdef class RandomSparseSplitter(BaseSparseSplitter):
    cdef class TreeBuilder:

    cdef class DepthFirstTreeBuilder(TreeBuilder):
    cdef class BestFirstTreeBuilder(TreeBuilder):
    cdef class Tree:

_tree.pxd
    cdef class Criterion:
    cdef struct SplitRecord:
    cdef class Splitter:
    cdef struct Node:
    cdef class Tree:
    cdef class TreeBuilder:
    cdef class Criterion:
    cdef struct SplitRecord:
    cdef class Splitter:
    cdef struct Node:
    cdef class Tree:
    cdef class TreeBuilder:

utils.pyx
    cdef class Stack:
    cdef class PriorityHeap:

utils.pxd
    cdef struct StackRecord:
    cdef class Stack:
    cdef struct PriorityHeapRecord:
    cdef class PriorityHeap:

