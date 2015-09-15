/*
This module gathers tree-based methods, including decision, regression and
randomized trees. Single and multi-output problems are both handled.
*/

// Authors: Gilles Louppe <g.louppe@gmail.com>
//          Peter Prettenhofer <peter.prettenhofer@gmail.com>
//          Brian Holt <bdholt1@gmail.com>
//          Noel Dawe <noel@dawe.me>
//          Satrajit Gosh <satrajit.ghosh@gmail.com>
//          Joly Arnaud <arnaud.v.joly@gmail.com>
//          Fares Hedayati <fares.hedayati@gmail.com>
//
// Licence: BSD 3 clause

#if 0
from __future__ import division


import numbers
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.sparse import issparse

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin
from ..externals import six
from ..feature_selection.from_model import _LearntSelectorMixin
from ..utils import check_array, check_random_state, compute_sample_weight
from ..utils.validation import NotFittedError


from ._tree import Criterion
from ._tree import Splitter
from ._tree import DepthFirstTreeBuilder, BestFirstTreeBuilder
from ._tree import Tree
from . import _tree

__all__ = ["DecisionTreeClassifier",
           "DecisionTreeRegressor",
           "ExtraTreeClassifier",
           "ExtraTreeRegressor"]

#endif

#include "sx/range.h"
#include "sx/abbrev.h"
#include "sx/algorithm.h"
#include "sx/string.h"
#include "sx/sort.h"

#include "sklearn/tree/_tree.pxd.h"
#include "sklearn/tree/aux_types.h"
#include "sklearn/utils/class_weight.py.h"

namespace sklearn {

using sx::matrix_view;
using sx::array_view;
using sx::multi_array;
using sx::matrix;
using sx::range;
// =============================================================================
// Types and constants
// =============================================================================

//CRITERIA_CLF = {"gini": _tree.Gini, "entropy": _tree.Entropy}
//CRITERIA_REG = {"mse": _tree.MSE, "friedman_mse": _tree.FriedmanMSE}
//
//DENSE_SPLITTERS = {"best": _tree.BestSplitter,
//                   "presort-best": _tree.PresortBestSplitter,
//                   "random": _tree.RandomSplitter}
//
//SPARSE_SPLITTERS = {"best": _tree.BestSparseSplitter,
//                    "random": _tree.RandomSparseSplitter}

// =============================================================================
// Base decision tree
// =============================================================================


class BaseDecisionTree
//(six.with_metaclass(ABCMeta, BaseEstimator,
//                                          _LearntSelectorMixin)):
{
    /*Base class for decision trees.

    Warning: This class should not be used directly.
    Use derived classes instead.
    */
private:
    const criterion_union criterion;
    const splitter_union splitter;
    const SIZE_t max_depth;
    const SIZE_t min_samples_split;
    const SIZE_t min_samples_leaf;
    const double min_weight_fraction_leaf;
    const max_features_union max_features;
    const SIZE_t max_leaf_nodes;
    std::default_random_engine random_state;
    const bool is_classification;
    const class_weight_union<DOUBLE_t> class_weight;
    SIZE_t n_features_ = 0;
    SIZE_t n_outputs_ = 0;
    std::vector<std::vector<DOUBLE_t>> classes_; //classes_[output_idx]
    NClasses n_classes_; //n_classes_[output_idx]
    SIZE_t max_features_ = 0;
    std::unique_ptr<Tree> tree_;
protected:
    BaseDecisionTree(
                 criterion_union&& criterion,
                 splitter_union&& splitter,
                 SIZE_t max_depth,
                 SIZE_t min_samples_split,
                 SIZE_t min_samples_leaf,
                 double min_weight_fraction_leaf,
                 max_features_union max_features,
                 SIZE_t max_leaf_nodes,
                 std::default_random_engine random_state,
                 bool is_classification,
                 class_weight_union<DOUBLE_t>&& class_weight = nullptr)
    : criterion(std::move(criterion))
    , splitter(std::move(splitter))
        , max_depth(max_depth)
        , min_samples_split(min_samples_split)
        , min_samples_leaf(min_samples_leaf)
        , min_weight_fraction_leaf(min_weight_fraction_leaf)
        , max_features(max_features)
    , max_leaf_nodes(max_leaf_nodes)
        , random_state(random_state)
        , is_classification(is_classification)
    , class_weight(std::move(class_weight))
    {}
    BaseDecisionTree& fit(
        matrix_view<const float> X, matrix_view<const double> y,
        array_view<const double> sample_weight=array_view<const double>()) {
        /*Build a decision tree from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression). In the regression case, use ``dtype=np.float64`` and
            ``order='C'`` for maximum efficiency.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        self : object
            Returns self.
        */

        auto& self = *this;

        // Determine output settings
        auto n_samples = X.extents(0);
        self.n_features_ = X.extents(1);

        std::vector<double> expanded_class_weight;

        self.n_outputs_ = y.extents(1);

        matrix<double> y_store_unique_indices;

        if(self.is_classification) {
            self.classes_.clear();
            self.n_classes_.clear();

            matrix_view<const double> y_original;
            if(!self.class_weight.is_none())
                y_original = y;

            y_store_unique_indices.assign(y.extents(), sx::array_layout::c_order, 0.0);
            std::vector<double> v;
            for(auto k: range(self.n_outputs_)) {
                v.assign(BEGINEND(y(sx::all, k)));
                sx::sort_unique_inplace(v);
                self.classes_.emplace_back(v);
                self.n_classes_.push_back(v.size());
                for(auto ix: range(y.extents(0))) {
                    auto it = std::lower_bound(BEGINEND(v), y(ix, k));
                    y_store_unique_indices(ix, k) = it - v.begin();
                }
            }
            y = y_store_unique_indices;

            if(!self.class_weight.is_none()) {
                expanded_class_weight = compute_sample_weight(
                    self.class_weight, y_original);
            }
        } else {
            self.classes_.assign(self.n_outputs_, std::vector<DOUBLE_t>());
            self.n_classes_.assign(self.n_outputs_, 1);
        }

        // Check parameters
        auto max_depth = self.max_depth == 0 ? ((1 << 31) - 1) : self.max_depth;
        auto max_leaf_nodes = self.max_leaf_nodes == 0 ? -1 : self.max_leaf_nodes;

        int max_features;
        if(self.max_features.is_string()) {
            if(self.max_features.is_string("auto")) {
                if(self.is_classification)
                    max_features = std::max(1, int(sqrt(self.n_features_)));
                else
                    max_features = self.n_features_;
            } else if(self.max_features.is_string("sqrt"))
                max_features = std::max(1, int(sqrt(self.n_features_)));
            else if(self.max_features.is_string("log2"))
                max_features = std::max(1, int(log2(self.n_features_)));
            else
                throw ValueError("Invalid value for max_features. Allowed string "
                                 "values are \"auto\", \"sqrt\" or \"log2\".");
        } else if(self.max_features.is_none())
            max_features = self.n_features_;
        else if(self.max_features.is_int())
            max_features = self.max_features.i;
        else {
            assert(self.max_features.is_float());
            if(self.max_features.f > 0.0)
                max_features = std::max(1, int(self.max_features.f * self.n_features_));
            else
                max_features = 0;
        }

        self.max_features_ = max_features;

        using sx::stringf;

        if(y.extents(0) != n_samples)
            throw ValueError(stringf("Number of labels=%d does not match "
                             "number of samples=%d", (int)y.extents(0), (int)n_samples));
        if(self.min_samples_split <= 0)
            throw ValueError("min_samples_split must be greater than zero.");
        if(self.min_samples_leaf <= 0)
            throw ValueError("min_samples_leaf must be greater than zero.");
        if(!sx::leq_and_leq(0, self.min_weight_fraction_leaf, 0.5))
            throw ValueError("min_weight_fraction_leaf must in [0, 0.5]");
        if(max_depth <= 0)
            throw ValueError("max_depth must be greater than zero. ");
        if(!sx::less_and_leq(0, max_features, self.n_features_))
            throw ValueError("max_features must be in (0, n_features]");
        if(sx::less_and_less(-1, max_leaf_nodes, 2))
            throw ValueError(stringf("max_leaf_nodes %d must be either smaller than "
                              "0 or larger than 1", (int)max_leaf_nodes));

        if(!isempty(sample_weight)) {
            if(length(sample_weight) != n_samples) {
                throw ValueError(stringf("Number of weights=%d does not match "
                                 "number of samples=%d",
                                 (int)length(sample_weight), (int)n_samples));
            }
        }
        if(!sx::isempty(expanded_class_weight)) {
            if(!isempty(sample_weight)) {
                std::vector<double> updated_sample_weight(BEGINEND(sample_weight));
                assert(length(sample_weight) == sx::length(expanded_class_weight));
                for(auto i: range(length(sample_weight)))
                    updated_sample_weight[i] *= expanded_class_weight[i];
                sample_weight = updated_sample_weight;
            } else {
                sample_weight = expanded_class_weight;
            }
        }

        double min_weight_leaf;
        // Set min_weight_leaf from min_weight_fraction_leaf
        if(self.min_weight_fraction_leaf != 0. && !isempty(sample_weight)) {
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               sum(sample_weight));
        } else {
            min_weight_leaf = 0.;
        }

        // Set min_samples_split sensibly
        int min_samples_split = std::max(self.min_samples_split,
                                2 * self.min_samples_leaf);

        // Build tree
        Criterion* criterion = self.criterion.is_object()
            ? self.criterion.object.get()
            : nullptr;
        std::unique_ptr<Criterion> local_criterion_object;
        if(!criterion) {
            if(self.is_classification) {
                if(self.criterion.is_string("gini"))
                    local_criterion_object = std::make_unique<Gini>(self.n_outputs_, self.n_classes_);
                else if(self.criterion.is_string("entropy"))
                    local_criterion_object = std::make_unique<Entropy>(self.n_outputs_, self.n_classes_);
                else
                    throw std::runtime_error("Classiciation criterion preset can be only gini, entropy");
            } else {
                if(self.criterion.is_string("mse"))
                    local_criterion_object = std::make_unique<MSE>(self.n_outputs_);
                else if(self.criterion.is_string("friedman_mse"))
                    local_criterion_object = std::make_unique<FriedmanMSE>(self.n_outputs_);
                else
                    throw std::runtime_error("Regression criterion preset can be only mse, friedman_mse");
            }
            criterion = local_criterion_object.get();
        }

        //SPLITTERS = SPARSE_SPLITTERS if issparse(X) else DENSE_SPLITTERS

        Splitter* splitter = self.splitter.is_object()
            ? self.splitter.object.get()
            : nullptr;
        std::unique_ptr<Splitter> local_splitter_object;
        if(!splitter) {
            if(self.splitter.is_string("best"))
                local_splitter_object = std::make_unique<BestSplitter>(
                    criterion, self.max_features_, self.min_samples_leaf,
                    min_weight_leaf, random_state);
            else if(self.splitter.is_string("presort_best"))
                local_splitter_object = std::make_unique<PresortBestSplitter>(
                    criterion, self.max_features_, self.min_samples_leaf,
                    min_weight_leaf, random_state);
            else if(self.splitter.is_string("random"))
                local_splitter_object = std::make_unique<RandomSplitter>(
                    criterion, self.max_features_, self.min_samples_leaf,
                    min_weight_leaf, random_state);
            else
                throw std::runtime_error("Splitter preset can be only best, presort-best, random");
            splitter = local_splitter_object.get();
        }

        self.tree_ = std::make_unique<Tree>(self.n_features_, self.n_classes_, self.n_outputs_);

        // Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        std::unique_ptr<TreeBuilder> builder;
        if(max_leaf_nodes < 0) {
            builder = std::make_unique<DepthFirstTreeBuilder>(splitter, min_samples_split,
                                            self.min_samples_leaf,
                                            min_weight_leaf,
                                            max_depth);
        } else {
            builder = std::make_unique<BestFirstTreeBuilder>(splitter, min_samples_split,
                                           self.min_samples_leaf,
                                           min_weight_leaf,
                                           max_depth,
                                           max_leaf_nodes);
        }
        builder->build(*self.tree_, X, y, sample_weight);

        return self;
    }

    void _validate_X_predict(matrix_view<const DTYPE_t> X, bool check_input) const {
        /*Validate X whenever one tries to predict, apply, predict_proba*/
        auto& self = *this;
        if(!self.tree_)
            throw NotFittedError("Estimator not fitted, "
                                 "call `fit` before exploiting the model.");

        auto n_features = X.extents(1);
        if(self.n_features_ != n_features)
            throw ValueError(sx::stringf("Number of features of the model must "
                             " match the input. Model n_features is %d and "
                             " input n_features is %d "
                             , (int)self.n_features_, (int)n_features));
    }

    matrix<DOUBLE_t> predict(
        matrix_view<const DTYPE_t> X, bool check_input = true
    ) const {
        /*Predict class or regression value for X.

        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes, or the predict values.
        */

        auto& self = *this;

        self._validate_X_predict(X, check_input);
        auto proba = self.tree_->predict(X);
        auto n_samples = X.extents(0);

        // Classification
        if(is_classification) {
            matrix<DOUBLE_t> predictions({n_samples, self.n_outputs_}, sx::array_layout::c_order);

            for(auto k: range(self.n_outputs_)) {
                predictions(sx::all, k) <<=
                take_at(self.classes_[k],
                    sx::indmax_along(
                        proba(sx::all, {n_classes_.offset(k), sx::length = n_classes_.count(k)})
                        , 1
                    )
                );
            }
            return predictions;
        // Regression
        } else {
            return proba; //[:, :, 0]
        }
    }
    std::vector<SIZE_t> apply(matrix_view<const DTYPE_t> X, bool check_input = true) const {
        /*
        Returns the index of the leaf that each sample is predicted as.

        Parameters
        ----------
        X : array_like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        X_leaves : array_like, shape = [n_samples,]
            For each datapoint x in X, return the index of the leaf x
            ends up in. Leaves are numbered within
            ``[0; self.tree_.node_count)``, possibly with gaps in the
            numbering.
        */

        auto& self = *this;

        self._validate_X_predict(X, check_input);
        return self.tree_->apply(X);
    }

    std::vector<double> feature_importances_() const {
        /*Return the feature importances.

        The importance of a feature is computed as the (normalized) total
        reduction of the criterion brought by that feature.
        It is also known as the Gini importance.

        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        */
        if(!this->tree_)
            throw NotFittedError("Estimator not fitted, call `fit` before"
                                 " `feature_importances_`.");

        return this->tree_->compute_feature_importances();
    }
};
#if 0
// =============================================================================
// Public estimators
// =============================================================================

class DecisionTreeClassifier(BaseDecisionTree, ClassifierMixin):
    /*A decision tree classifier.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.

    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:
          - If int, then consider `max_features` features at each split.
          - If float, then `max_features` is a percentage and
            `int(max_features * n_features)` features are considered at each
            split.
          - If "auto", then `max_features=sqrt(n_features)`.
          - If "sqrt", then `max_features=sqrt(n_features)`.
          - If "log2", then `max_features=log2(n_features)`.
          - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        Ignored if ``max_leaf_nodes`` is not None.

    min_samples_split : int, optional (default=2)
        The minimum number of samples required to split an internal node.

    min_samples_leaf : int, optional (default=1)
        The minimum number of samples required to be at a leaf node.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the input samples required to be at a
        leaf node.

    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
        If not None then ``max_depth`` will be ignored.

    class_weight : dict, list of dicts, "balanced" or None, optional
                   (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    classes_ : array of shape = [n_classes] or a list of such arrays
        The classes labels (single output problem),
        or a list of arrays of class labels (multi-output problem).

    feature_importances_ : array of shape = [n_features]
        The feature importances. The higher, the more important the
        feature. The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance [4]_.

    max_features_ : int,
        The inferred value of max_features.

    n_classes_ : int or list
        The number of classes (for single output problems),
        or a list containing the number of classes for each
        output (for multi-output problems).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree object
        The underlying Tree object.

    See also
    --------
    DecisionTreeRegressor

    References
    ----------

    .. [1] http://en.wikipedia.org/wiki/Decision_tree_learning

    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.

    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.

    .. [4] L. Breiman, and A. Cutler, "Random Forests",
           http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.cross_validation import cross_val_score
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> clf = DecisionTreeClassifier(random_state=0)
    >>> iris = load_iris()
    >>> cross_val_score(clf, iris.data, iris.target, cv=10)
    ...                             // doctest: +SKIP
    ...
    array([ 1.     ,  0.93...,  0.86...,  0.93...,  0.93...,
            0.93...,  0.93...,  1.     ,  0.93...,  1.      ])
    */
    def __init__(self,
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 class_weight=None):
        super(DecisionTreeClassifier, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state)

    def predict_proba(self, X, check_input=True):
        /*Predict class probabilities of the input samples X.

        The predicted class probability is the fraction of samples of the same
        class in a leaf.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        */
        X = self._validate_X_predict(X, check_input)
        proba = self.tree_.predict(X)

        if self.n_outputs_ == 1:
            proba = proba[:, :self.n_classes_]
            normalizer = proba.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            proba /= normalizer

            return proba

        else:
            all_proba = []

            for k in range(self.n_outputs_):
                proba_k = proba[:, k, :self.n_classes_[k]]
                normalizer = proba_k.sum(axis=1)[:, np.newaxis]
                normalizer[normalizer == 0.0] = 1.0
                proba_k /= normalizer
                all_proba.append(proba_k)

            return all_proba

    def predict_log_proba(self, X):
        /*Predict class log-probabilities of the input samples X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        */
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return np.log(proba)

        else:
            for k in range(self.n_outputs_):
                proba[k] = np.log(proba[k])

            return proba


class DecisionTreeRegressor(BaseDecisionTree, RegressorMixin):
    /*A decision tree regressor.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : string, optional (default="mse")
        The function to measure the quality of a split. The only supported
        criterion is "mse" for the mean squared error, which is equal to
        variance reduction as feature selection criterion.

    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:
          - If int, then consider `max_features` features at each split.
          - If float, then `max_features` is a percentage and
            `int(max_features * n_features)` features are considered at each
            split.
          - If "auto", then `max_features=n_features`.
          - If "sqrt", then `max_features=sqrt(n_features)`.
          - If "log2", then `max_features=log2(n_features)`.
          - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        Ignored if ``max_leaf_nodes`` is not None.

    min_samples_split : int, optional (default=2)
        The minimum number of samples required to split an internal node.

    min_samples_leaf : int, optional (default=1)
        The minimum number of samples required to be at a leaf node.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the input samples required to be at a
        leaf node.

    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
        If not None then ``max_depth`` will be ignored.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    feature_importances_ : array of shape = [n_features]
        The feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the
        (normalized) total reduction of the criterion brought
        by that feature. It is also known as the Gini importance [4]_.

    max_features_ : int,
        The inferred value of max_features.

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree object
        The underlying Tree object.

    See also
    --------
    DecisionTreeClassifier

    References
    ----------

    .. [1] http://en.wikipedia.org/wiki/Decision_tree_learning

    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.

    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.

    .. [4] L. Breiman, and A. Cutler, "Random Forests",
           http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

    Examples
    --------
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.cross_validation import cross_val_score
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> boston = load_boston()
    >>> regressor = DecisionTreeRegressor(random_state=0)
    >>> cross_val_score(regressor, boston.data, boston.target, cv=10)
    ...                    // doctest: +SKIP
    ...
    array([ 0.61..., 0.57..., -0.34..., 0.41..., 0.75...,
            0.07..., 0.29..., 0.33..., -1.42..., -1.77...])
    */
    def __init__(self,
                 criterion="mse",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None):
        super(DecisionTreeRegressor, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state)


class ExtraTreeClassifier(DecisionTreeClassifier):
    /*An extremely randomized tree classifier.

    Extra-trees differ from classic decision trees in the way they are built.
    When looking for the best split to separate the samples of a node into two
    groups, random splits are drawn for each of the `max_features` randomly
    selected features and the best split among those is chosen. When
    `max_features` is set 1, this amounts to building a totally random
    decision tree.

    Warning: Extra-trees should only be used within ensemble methods.

    Read more in the :ref:`User Guide <tree>`.

    See also
    --------
    ExtraTreeRegressor, ExtraTreesClassifier, ExtraTreesRegressor

    References
    ----------

    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.
    */
    def __init__(self,
                 criterion="gini",
                 splitter="random",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 random_state=None,
                 max_leaf_nodes=None,
                 class_weight=None):
        super(ExtraTreeClassifier, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state)


class ExtraTreeRegressor(DecisionTreeRegressor):
    /*An extremely randomized tree regressor.

    Extra-trees differ from classic decision trees in the way they are built.
    When looking for the best split to separate the samples of a node into two
    groups, random splits are drawn for each of the `max_features` randomly
    selected features and the best split among those is chosen. When
    `max_features` is set 1, this amounts to building a totally random
    decision tree.

    Warning: Extra-trees should only be used within ensemble methods.

    Read more in the :ref:`User Guide <tree>`.

    See also
    --------
    ExtraTreeClassifier, ExtraTreesClassifier, ExtraTreesRegressor

    References
    ----------

    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.
    */
    def __init__(self,
                 criterion="mse",
                 splitter="random",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 random_state=None,
                 max_leaf_nodes=None):
        super(ExtraTreeRegressor, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state)
#endif

} //namespace sklearn
