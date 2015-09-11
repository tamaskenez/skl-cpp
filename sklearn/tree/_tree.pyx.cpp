#include "_tree.pxd.h"

#include <vector>
#include <algorithm>
#include <cmath>

#include "sx/abbrev.h"
#include "sx/range.h"
#include "sx/random_access_iterator_pair.h"
#include "sx/multi_array.h"
#include "sx/sort.h"

#include "_utils.pxd.h"

namespace sklcpp {

using sx::range;

// cython: cdivision=True
// cython: boundscheck=False
// cython: wraparound=False

// Authors: Gilles Louppe <g.louppe@gmail.com>
//          Peter Prettenhofer <peter.prettenhofer@gmail.com>
//          Brian Holt <bdholt1@gmail.com>
//          Noel Dawe <noel@dawe.me>
//          Satrajit Gosh <satrajit.ghosh@gmail.com>
//          Lars Buitinck <L.J.Buitinck@uva.nl>
//          Arnaud Joly <arnaud.v.joly@gmail.com>
//          Joel Nothman <joel.nothman@gmail.com>
//          Fares Hedayati <fares.hedayati@gmail.com>
//          Jacob Schreiber <jmschreiber91@gmail.com>
//
// Licence: BSD 3 clause

/*
from libc.stdlib cimport calloc, free, realloc, qsort

from libc.string cimport memcpy, memset
from libc.math cimport log as ln
from cpython cimport Py_INCREF, PyObject

import numpy as np
cimport numpy as np
np.import_array()

from scipy.sparse import issparse, csc_matrix, csr_matrix

from sklearn.tree._utils cimport Stack, StackRecord
from sklearn.tree._utils cimport PriorityHeap, PriorityHeapRecord



cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(object subtype, np.dtype descr,
                                int nd, np.npy_intp* dims,
                                np.npy_intp* strides,
                                void* data, int flags, object obj)

// =============================================================================
// Types and constants
// =============================================================================

cdef double INFINITY = np.inf
TREE_LEAF = -1
TREE_UNDEFINED = -2
*/
    const SIZE_t _TREE_LEAF = -1;
    const SIZE_t _TREE_UNDEFINED = -2;
const SIZE_t INITIAL_STACK_SIZE = 10;

const DTYPE_t MIN_IMPURITY_SPLIT = 1e-7;

// Mitigate precision differences between 32 bit and 64 bit
const DTYPE_t FEATURE_THRESHOLD = 1e-7;
/*
// Constant to switch between algorithm non zero value extract algorithm
// in SparseSplitter
cdef DTYPE_t EXTRACT_NNZ_SWITCH = 0.1
*/
// Some handy constants (BestFirstTreeBuilder)
    const int IS_FIRST = 1;
    const int IS_NOT_FIRST = 0;
    const int IS_LEFT = 1;
    const int IS_NOT_LEFT = 0;
/*
cdef enum:
    // Max value for our rand_r replacement (near the bottom).
    // We don't use RAND_MAX because it's different across platforms and
    // particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF

// Repeat struct definition for numpy
NODE_DTYPE = np.dtype({
    'names': ['left_child', 'right_child', 'feature', 'threshold', 'impurity',
              'n_node_samples', 'weighted_n_node_samples'],
    'formats': [np.intp, np.intp, np.intp, np.float64, np.float64, np.intp,
                np.float64],
    'offsets': [
        <Py_ssize_t> &(<Node*> NULL).left_child,
        <Py_ssize_t> &(<Node*> NULL).right_child,
        <Py_ssize_t> &(<Node*> NULL).feature,
        <Py_ssize_t> &(<Node*> NULL).threshold,
        <Py_ssize_t> &(<Node*> NULL).impurity,
        <Py_ssize_t> &(<Node*> NULL).n_node_samples,
        <Py_ssize_t> &(<Node*> NULL).weighted_n_node_samples
    ]
})

*/

template<typename Generator>
inline SIZE_t rand_int(SIZE_t low, SIZE_t high,
                       Generator& random_state) {
    /* Generate a random integer in [0; end).*/
    return std::uniform_int_distribution<SIZE_t>(low, high-1)(random_state);
}

template<typename Generator>
inline double rand_uniform(double low, double high,
                           Generator& random_state) {
    /* Generate a random double in [low; high).*/
    return std::uniform_real_distribution<double>(low, high)(random_state);
}

// Sort n-element arrays pointed to by Xf and samples, simultaneously,
// by the values in Xf. Algorithm: Introsort (Musser, SP&E, 1997).
template<typename It1, typename It2>
void sort(It1 Xf, It2 samples, SIZE_t n) {
    auto it = sx::make_random_access_iterator_pair(Xf, samples);
    std::sort(it, it + n, sx::less_by_first());
}


// =============================================================================
// Criterion
// =============================================================================
Criterion::
Criterion(SIZE_t n_outputs)
: start(0)
, pos(0)
, end(0)
, n_outputs(n_outputs)
, n_node_samples(0)
, weighted_n_samples(0)
, weighted_n_node_samples(0)
, weighted_n_left(0)
, weighted_n_right(0)
{}

double Criterion::
impurity_improvement(double impurity) const {
    double impurity_left;
    double impurity_right;

    children_impurity(&impurity_left, &impurity_right);

    return ((weighted_n_node_samples / weighted_n_samples) *
        (impurity - weighted_n_right / weighted_n_node_samples * impurity_right
            - weighted_n_left / weighted_n_node_samples * impurity_left));
}

class ClassificationCriterion
    : public Criterion {
    /* Abstract criterion for classification.*/

protected:
    NClassesRef n_classes;

    // These arrays can be indexed by n_classes.idx(output_idx, class_idx)
    // and have n_classes.n_elements() elements.
    std::vector<double> label_count_left;
    std::vector<double> label_count_right;
    std::vector<double> label_count_total;

protected:
    ClassificationCriterion(NClassesRef n_classes)
        /* Initialize attributes for this criterion.

        Parameters
        ----------
        n_outputs: SIZE_t
            The number of targets, the dimensionality of the prediction
        n_classes: numpy.ndarray, dtype=SIZE_t
            The number of unique classes in each target
        */
        : Criterion(n_classes.n_outputs())
        , n_classes(n_classes)
    {
        // Count labels for each output

        // For each target, set the number of unique classes in that target,
        // and also compute the maximal stride of all targets
        auto n_elements = n_classes.n_elements();
        label_count_left.assign(n_elements, 0.0);
        label_count_right.assign(n_elements, 0.0);
        label_count_total.assign(n_elements, 0.0);
    }

public:
    virtual void init(sx::array_view<const DOUBLE_t, 2> y, sx::array_view<const DOUBLE_t> sample_weight,
        double weighted_n_samples, sx::array_view<const SIZE_t> samples, SIZE_t start,
        SIZE_t end) override {
    /* Initialize the criterion at node samples[start:end] and
    children samples[start:start] and samples[start:end].

    Parameters
    ----------
    y: array-like, dtype=DOUBLE_t
        The target stored as a buffer for memory efficiency
    y_stride: SIZE_t
        The stride between elements in the buffer, important if there
        are multiple targets (multi-output)
    sample_weight: array-like, dtype=DTYPE_t
        The weight of each sample
    weighted_n_samples: SIZE_t
        The total weight of all samples
    samples: array-like, dtype=SIZE_t
        A mask on the samples, showing which ones we want to use
    start: SIZE_t
        The first sample to use in the mask
    end: SIZE_t
        The last sample to use in the mask
    */

        this->y = y;
        this->sample_weight = sample_weight;
        this->samples = samples;
        this->start = start;
        this->end = end;
        this->n_node_samples = end - start;
        this->weighted_n_samples = weighted_n_samples;
        weighted_n_node_samples = 0.0;

        std::fill(BEGINEND(label_count_total), 0.0);

        //todo this double nested loop should be turned
        //inside out for better cache performance
        //if the y array is really column major (values
        //for same output are contiguous)
        DOUBLE_t w = 1.0;
        for (auto p: range(start, end)) {
            auto i = samples[p];

            // w is originally set to be 1.0, meaning that if no sample weights
            // are given, the default weight of each sample is 1.0
            if (!sample_weight.empty())
                w = sample_weight[i];

            // Count weighted class frequency for each target
            for (auto k: range(n_outputs)) {
                auto yik = y[{i, k}];
                auto c = (SIZE_t)yik;
                assert(c == yik);
                label_count_total[n_classes.idx(k, c)] += w;
            }
            weighted_n_node_samples += w;
        }

        // Reset to pos=start
        reset();
    }

    virtual void reset() override {
    /* Reset the criterion at pos=start.*/

        pos = start;

        weighted_n_left = 0.0;
        weighted_n_right = weighted_n_node_samples;

        std::fill(BEGINEND(label_count_left), 0.0);
        label_count_right = label_count_total;
    }

    virtual void update(SIZE_t new_pos) override {
        /* Updated statistics by moving samples[pos:new_pos] to the left child.

        Parameters
        ----------
        new_pos: SIZE_t
            The new ending position for which to move samples from the right
            child to the left child.
        */

        DOUBLE_t w = 1.0;
        DOUBLE_t diff_w = 0.0;

        // Note: We assume start <= pos < new_pos <= end
        // Go through each sample in the new indexing
        for (auto p : range(pos, new_pos)) {
            auto i = samples[p];

            if (!sample_weight.empty())
                w = sample_weight[i];

            for (auto k: range(n_outputs)) {
                auto yik = y[{i, k}];
                auto c = (SIZE_t)yik;
                assert(c == yik);
                auto idx = n_classes.idx(k, c);
                label_count_left[idx] += w;
                label_count_right[idx] -= w;
            }
            diff_w += w;
        }
        weighted_n_left += diff_w;
        weighted_n_right -= diff_w;

        pos = new_pos;
    }

    virtual void node_value(sx::array_view<double> dest) const override {
        /* Compute the node value of samples[start:end] and save it into dest.

        Parameters
        ----------
        dest: double pointer
            The memory address which we will save the node value into.
        */

        assert(dest.extents(0) == label_count_total.size());
        std::copy_n(label_count_total.begin(), label_count_total.size(),
                    dest.begin());
    }
};

class Entropy
    : public ClassificationCriterion {

public:
    /* Cross Entropy impurity criterion.

    This handles cases where the target is a classification taking values
    0, 1, ... K-2, K-1. If node m represents a region Rm with Nm observations,
    then let

        count_k = 1 / Nm \sum_{x_i in Rm} I(yi = k)

    be the proportion of class k observations in node m.

    The cross-entropy is then defined as

        cross-entropy = -\sum_{k=0}^{K-1} count_k log(count_k)
    */

    virtual double node_impurity() const override {
        /* Evaluate the impurity of the current node, i.e. the impurity of
        samples[start:end], using the cross-entropy criterion.*/

        double total_entropy = 0.0;
        for(auto k: range(n_outputs)) {
            double entropy = 0.0;

            for(auto c: range(n_classes.count(k))) {
                double count_k = label_count_total[n_classes.idx(k, c)];
                if(count_k > 0.0) {
                    count_k /= weighted_n_node_samples;
                    entropy -= count_k * log(count_k);
                }
            }
            total_entropy += entropy;
        }
        return total_entropy / n_outputs;
    }

    virtual void children_impurity(double* impurity_left,
                                double* impurity_right) const override {
        /* Evaluate the impurity in children nodes

        i.e. the impurity of the left child (samples[start:pos]) and the
        impurity the right child (samples[pos:end]).

        Parameters
        ----------
        impurity_left: double pointer
            The memory address to save the impurity of the left node
        impurity_right: double pointer
            The memory address to save the impurity of the right node
        */

        double total_left = 0.0;
        double total_right = 0.0;

        for(auto k: range(n_outputs)) {
            double entropy_left = 0.0;
            double entropy_right = 0.0;

            for(auto c: range(n_classes.count(k))) {
                auto idx = n_classes.idx(k, c);
                auto count_k = label_count_left[idx];
                if(count_k > 0.0) {
                    count_k /= weighted_n_left;
                    entropy_left -= count_k * log(count_k);
                }
                count_k = label_count_right[idx];
                if(count_k > 0.0) {
                    count_k /= weighted_n_right;
                    entropy_right -= count_k * log(count_k);
                }
            }
            total_left += entropy_left;
            total_right += entropy_right;
        }
        *impurity_left = total_left / n_outputs;
        *impurity_right = total_right / n_outputs;
    }
};

class Gini
    : public ClassificationCriterion {
public:
    /* Gini Index impurity criterion.

    This handles cases where the target is a classification taking values
    0, 1, ... K-2, K-1. If node m represents a region Rm with Nm observations,
    then let

        count_k = 1/ Nm \sum_{x_i in Rm} I(yi = k)

    be the proportion of class k observations in node m.

    The Gini Index is then defined as:

        index = \sum_{k=0}^{K-1} count_k (1 - count_k)
              = 1 - \sum_{k=0}^{K-1} count_k ** 2
    */

    virtual double node_impurity() const override {
        /* Evaluate the impurity of the current node, i.e. the impurity of
        samples[start:end] using the Gini criterion.*/

        double total = 0.0;

        for(auto k: range(n_outputs)) {
            auto gini = 0.0;

            for(auto c: range(n_classes.count(k))) {
                auto count_k = label_count_total[n_classes.idx(k, c)];
                gini += count_k * count_k;
            }
            gini = 1.0 - gini / (weighted_n_node_samples *
                                 weighted_n_node_samples);

            total += gini;
        }
        return total / n_outputs;
    }

    virtual void children_impurity(double* impurity_left,
                                   double* impurity_right) const override {
        /* Evaluate the impurity in children nodes

        i.e. the impurity of the left child (samples[start:pos]) and the
        impurity the right child (samples[pos:end]) using the Gini index.

        Parameters
        ----------
        impurity_left: DTYPE_t
            The memory address to save the impurity of the left node to
        impurity_right: DTYPE_t
            The memory address to save the impurity of the right node to
        */

        double total_left = 0.0;
        double total_right = 0.0;

        for(auto k: range(n_outputs)) {
            auto gini_left = 0.0;
            auto gini_right = 0.0;

            for(auto c: range(n_classes.count(k))) {
                auto idx = n_classes.idx(k, c);
                auto count_k = label_count_left[idx];
                gini_left += count_k * count_k;

                count_k = label_count_right[idx];
                gini_right += count_k * count_k;
            }

            gini_left = 1.0 - gini_left / (weighted_n_left *
                                           weighted_n_left);

            gini_right = 1.0 - gini_right / (weighted_n_right *
                                             weighted_n_right);

            total_left += gini_left;
            total_right += gini_right;
        }

        *impurity_left = total_left / n_outputs;
        *impurity_right = total_right / n_outputs;
    }
};

class RegressionCriterion
    : public Criterion {
    /* Abstract regression criterion.

    This handles cases where the target is a continuous value, and is
    evaluated by computing the variance of the target values left and right
    of the split point. The computation takes linear time with `n_samples`
    by using ::

        var = \sum_i^n (y_i - y_bar) ** 2
            = (\sum_i^n y_i ** 2) - n_samples * y_bar ** 2
    */
protected:
    struct accumulators_t {
        double mean_left;
        double mean_right;
        double mean_total;
        double sq_sum_left;
        double sq_sum_right;
        double sq_sum_total;
        double var_left;
        double var_right;
        double sum_left;
        double sum_right;
        double sum_total;
        void set_zero() {
            mean_left = 0.0;
            mean_right = 0.0;
            mean_total = 0.0;
            sq_sum_left = 0.0;
            sq_sum_right = 0.0;
            sq_sum_total = 0.0;
            var_left = 0.0;
            var_right = 0.0;
            sum_left = 0.0;
            sum_right = 0.0;
            sum_total = 0.0;
        }
    };

    accumulators_t first_output_accumulators;
    std::vector<accumulators_t> remaining_outputs_accumulators;

    inline accumulators_t& accu(SIZE_t idx) {
        return idx == 0
            ? first_output_accumulators
            : remaining_outputs_accumulators[idx-1];
    }
    inline const accumulators_t& accu(SIZE_t idx) const {
        return idx == 0
        ? first_output_accumulators
        : remaining_outputs_accumulators[idx-1];
    }

    RegressionCriterion(SIZE_t n_outputs)
    : Criterion(n_outputs) {
        /* Initialize parameters for this criterion.

        Parameters
        ----------
        n_outputs: SIZE_t
            The number of targets to be predicted
        */

        // Allocate memory for the accumulators
        if(n_outputs > 0)
            remaining_outputs_accumulators.resize(n_outputs - 1);
    }

    virtual void init(
        sx::array_view<const DOUBLE_t, 2> y,
        sx::array_view<const DOUBLE_t> sample_weight,
        double weighted_n_samples,
        sx::array_view<const SIZE_t> samples, SIZE_t start,
        SIZE_t end) override {
        /* Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end].*/
        // Initialize fields
        this->y = y;
        this->sample_weight = sample_weight;
        this->samples = samples;
        this->start = start;
        this->end = end;
        n_node_samples = end - start;
        this->weighted_n_samples = weighted_n_samples;
        weighted_n_node_samples = 0.0;

        // Initialize accumulators
        DOUBLE_t w = 1.0;

        first_output_accumulators.set_zero();
        for(auto&a: remaining_outputs_accumulators)
            a.set_zero();

        for(auto p: range(start, end)) {
            auto i = samples[p];

            if(!sample_weight.empty())
                w = sample_weight[i];

            for(auto k: range(n_outputs)) {
                auto y_ik = y[{i, k}];
                auto w_y_ik = w * y_ik;
                auto&a = accu(k);
                a.sum_total += w_y_ik;
                a.sq_sum_total += w_y_ik * y_ik;
            }
            weighted_n_node_samples += w;
        }

        for(auto k: range(n_outputs)) {
            auto&a = accu(k);
            a.mean_total = a.sum_total / weighted_n_node_samples;
        }

        // Reset to pos=start
        reset();
    }

    virtual void reset() override {
        /* Reset the criterion at pos=start.*/
        pos = start;

        weighted_n_left = 0.0;
        weighted_n_right = weighted_n_node_samples;

        for(auto k: range(n_outputs)) {
            auto& a = accu(k);
            a.mean_right = a.mean_total;
            a.mean_left = 0.0;
            a.sq_sum_right = a.sq_sum_total;
            a.sq_sum_left = 0.0;
            a.var_right = a.sq_sum_right / weighted_n_node_samples -
                            a.mean_right * a.mean_right;
            a.var_left = 0.0;
            a.sum_right = a.sum_total;
            a.sum_left = 0.0;
        }
    }

    virtual void update(SIZE_t new_pos) override {
        /* Updated statistics by moving samples[pos:new_pos] to the left.*/

        DOUBLE_t w = 1.0;
        DOUBLE_t diff_w = 0.0;

        // Note: We assume start <= pos < new_pos <= end
        assert(start <= pos && pos < new_pos && new_pos <= end);
        for(auto p: range(pos, new_pos)) {
            auto i = samples[p];

            if(!sample_weight.empty())
                w = sample_weight[i];

            for(auto k: range(n_outputs)) {
                auto y_ik = y[{i, k}];
                auto w_y_ik = w * y_ik;

                auto& a = accu(k);
                a.sum_left += w_y_ik;
                a.sum_right -= w_y_ik;

                a.sq_sum_left += w_y_ik * y_ik;
                a.sq_sum_right -= w_y_ik * y_ik;
            }
            diff_w += w;
        }

        weighted_n_left += diff_w;
        weighted_n_right -= diff_w;

        for(auto k: range(n_outputs)) {
            auto&a = accu(k);
            a.mean_left = a.sum_left / weighted_n_left;
            a.mean_right = a.sum_right / weighted_n_right;
            a.var_left = (a.sq_sum_left / weighted_n_left -
                           a.mean_left * a.mean_left);
            a.var_right = (a.sq_sum_right / weighted_n_right -
                            a.mean_right * a.mean_right);
        }
    }

    virtual void node_value(sx::array_view<double> dest) const override {
        /* Compute the node value of samples[start:end] into dest.*/
        assert(dest.extents(0) == n_outputs);
        for(auto k: range(n_outputs))
            dest[k] = accu(k).mean_total;
    }
};

class MSE
: public RegressionCriterion {
    /* Mean squared error impurity criterion.

        MSE = var_left + var_right
    */
    virtual double node_impurity() const override {
        /* Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end].*/
        double total = 0.0;

        for(auto k: range(n_outputs)) {
            auto&a = accu(k);
            total += (a.sq_sum_total / weighted_n_node_samples -
                      a.mean_total * a.mean_total);
        }

        return total / n_outputs;
    }

    virtual void children_impurity(double* impurity_left,
                                double* impurity_right) const override {
        /* Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end]).*/
        double total_left = 0.0;
        double total_right = 0.0;

        for(auto k: range(n_outputs)) {
            auto& a = accu(k);
            total_left += a.var_left;
            total_right += a.var_right;
        }

        *impurity_left = total_left / n_outputs;
        *impurity_right = total_right / n_outputs;
    }
};

class FriedmanMSE
: public MSE {
    /* Mean squared error impurity criterion with improvement score by Friedman

    Uses the formula (35) in Friedmans original Gradient Boosting paper:

        diff = mean_left - mean_right
        improvement = n_left * n_right * diff^2 / (n_left + n_right)
    */

    virtual double impurity_improvement(double impurity) const override {
        double total_sum_left = 0.0;
        double total_sum_right = 0.0;

        for(auto k: range(n_outputs)) {
            auto&a = accu(k);
            total_sum_left += a.sum_left;
            total_sum_right += a.sum_right;
        }

        total_sum_left = total_sum_left / n_outputs;
        total_sum_right = total_sum_right / n_outputs;
        double diff = ((total_sum_left / weighted_n_left) -
                (total_sum_right / weighted_n_right));

        return (weighted_n_left * weighted_n_right * diff * diff /
                (weighted_n_left + weighted_n_right));
    }
};

// =============================================================================
// Splitter
// =============================================================================

inline void _init_split(SplitRecord* self, SIZE_t start_pos) {
    self->impurity_left = INFINITY;
    self->impurity_right = INFINITY;
    self->pos = start_pos;
    self->feature = 0;
    self->threshold = 0.0;
    self->improvement = -INFINITY;
}


Splitter::
Splitter(Criterion* criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  std::default_random_engine* random_state)
: criterion(criterion)
, max_features(max_features)
, min_samples_leaf(min_samples_leaf)
, min_weight_leaf(min_weight_leaf)
, random_state(random_state)
, n_samples(0)
, weighted_n_samples(0.0)
, n_features(0)
, start(0)
, end(0)
{}

void Splitter::
init(sx::array_view<const DTYPE_t, 2> X,
          sx::array_view<const DOUBLE_t, 2> y,
          sx::array_view<const DOUBLE_t> sample_weight) {

    // Create a new array which will be used to store nonzero
    // samples from the feature of interest
    samples.clear();
    samples.reserve(X.extents()[0]);

    weighted_n_samples = 0.0;

    for(auto i: range(n_samples)) {
        // Only work with positively weighted samples
        if(sample_weight.empty() || sample_weight[i] != 0.0)
            samples.push_back(i);

        if(!sample_weight.empty())
            weighted_n_samples += sample_weight[i];
        else
            weighted_n_samples += 1.0;
    }
    n_samples = samples.size();

    n_features = X.extents()[1];
    features.resize(n_features);

    std::iota(BEGINEND(features), 0);

    feature_values.resize(n_samples);
    constant_features.resize(n_features);

    this->y = y;
    this->sample_weight = sample_weight;
}

void Splitter::
node_reset(SIZE_t start, SIZE_t end,
             double* weighted_n_node_samples) {

    this->start = start;
    this->end = end;

    criterion->init(y,
                    sample_weight,
                    weighted_n_samples,
                    samples,
                    start,
                    end);

    *weighted_n_node_samples = criterion->get_weighted_n_node_samples();
}

void Splitter::
node_value(sx::array_view<double> dest) const {
    criterion->node_value(dest);
}

double Splitter::
node_impurity() const {
    /* Return the impurity of the current node.*/
    return criterion->node_impurity();
}

class BaseDenseSplitter
: public Splitter {
protected:
    sx::array_view<const DTYPE_t, 2> X;
public:
    BaseDenseSplitter(Criterion* criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  std::default_random_engine* random_state)
    : Splitter(criterion, max_features, min_samples_leaf, min_weight_leaf,
               random_state)
    {}

    virtual void init(sx::array_view<const DTYPE_t, 2> X,
              sx::array_view<const DOUBLE_t, 2> y,
              sx::array_view<const DOUBLE_t> sample_weight) override {
        /* Initialize the splitter.*/

        // Call parent init
        Splitter::init(X, y, sample_weight);

        // Initialize X
        this->X = X;
    }
};

class BestSplitter
: public BaseDenseSplitter {
    /* Splitter for finding the best split.*/
    virtual void node_split(double impurity, SplitRecord* split,
                         SIZE_t* n_constant_features) override {
        /* Find the best split on node samples[start:end].*/
        // Find the best split

        auto& Xf = feature_values;

        SplitRecord best, current;

        SIZE_t f_i = n_features;
        //SIZE_t f_j, p, tmp
        SIZE_t n_visited_features = 0;
        // Number of features discovered to be constant during the split search
        SIZE_t n_found_constants = 0;
        // Number of features known to be constant and drawn without replacement
        SIZE_t n_drawn_constants = 0;
        SIZE_t n_known_constants = *n_constant_features;
        // n_total_constants = n_known_constants + n_found_constants
        SIZE_t n_total_constants = n_known_constants;
        //DTYPE_t current_feature_value
        //cdef SIZE_t partition_end

        _init_split(&best, end);

        // Sample up to max_features without replacement using a
        // Fisher-Yates-based algorithm (using the local variables `f_i` and
        // `f_j` to compute a permutation of the `features` array).
        //
        // Skip the CPU intensive evaluation of the impurity criterion for
        // features that were already detected as constant (hence not suitable
        // for good splitting) by ancestor nodes and save the information on
        // newly discovered constant features to spare computation on descendant
        // nodes.
        while (f_i > n_total_constants &&   // Stop early if remaining features
                                            // are constant
                (n_visited_features < max_features ||
                 // At least one drawn features must be non constant
                 n_visited_features <= n_found_constants + n_drawn_constants)) {

            ++n_visited_features;

            // Loop invariant: elements of features in
            // - [:n_drawn_constant[ holds drawn and known constant features;
            // - [n_drawn_constant:n_known_constant[ holds known constant
            //   features that haven't been drawn yet;
            // - [n_known_constant:n_total_constant[ holds newly found constant
            //   features;
            // - [n_total_constant:f_i[ holds features that haven't been drawn
            //   yet and aren't constant apriori.
            // - [f_i:n_features[ holds features that have been drawn
            //   and aren't constant.

            // Draw a feature at random
            SIZE_t f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           *random_state);

            if(f_j < n_known_constants) {
                // f_j in the interval [n_drawn_constants, n_known_constants[
                std::swap(features[f_j], features[n_drawn_constants]);
                ++n_drawn_constants;
            } else {
                // f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += n_found_constants;
                // f_j in the interval [n_total_constants, f_i[

                current.feature = features[f_j];

                // Sort samples along that feature; first copy the feature
                // values for the active samples into Xf, s.t.
                // Xf[i] == X[samples[i], j], so the sort uses the cache more
                // effectively.
                for(auto p: range(start, end))
                    Xf[p] = X[{samples[p], current.feature}];

                sort(Xf.begin() + start, samples.begin() + start, end - start);

                if(Xf[end - 1] <= Xf[start] + FEATURE_THRESHOLD) {
                    features[f_j] = features[n_total_constants];
                    features[n_total_constants] = current.feature;

                    ++n_found_constants;
                    ++n_total_constants;
                } else {
                    --f_i;
                    std::swap(features[f_i], features[f_j]);

                    // Evaluate all splits
                    criterion->reset();
                    SIZE_t p = start;

                    while(p < end) {
                        while(p + 1 < end &&
                               Xf[p + 1] <= Xf[p] + FEATURE_THRESHOLD) {
                            p += 1;
                        }
                        // (p + 1 >= end) or (X[samples[p + 1], current.feature] >
                        //                    X[samples[p], current.feature])
                        ++p;
                        // (p >= end) or (X[samples[p], current.feature] >
                        //                X[samples[p - 1], current.feature])

                        if(p < end) {
                            current.pos = p;

                            // Reject if min_samples_leaf is not guaranteed
                            if (((current.pos - start) < min_samples_leaf) ||
                                    ((end - current.pos) < min_samples_leaf)) {
                                continue;
                            }

                            criterion->update(current.pos);

                            // Reject if min_weight_leaf is not satisfied
                            if ((criterion->get_weighted_n_left() < min_weight_leaf) ||
                                    (criterion->get_weighted_n_right() < min_weight_leaf)) {
                                continue;
                            }

                            current.improvement = criterion->impurity_improvement(impurity);

                            if(current.improvement > best.improvement) {
                                criterion->children_impurity(&current.impurity_left,
                                                                 &current.impurity_right);
                                current.threshold = (Xf[p - 1] + Xf[p]) / 2.0;

                                if(current.threshold == Xf[p])
                                    current.threshold = Xf[p - 1];

                                best = current;  // copy
                            }
                        } //if (p<end)
                    } //while(p<end)
                } //if(Xf[end - 1] <= Xf[start] + FEATURE_THRESHOLD)
            } //else of if(f_j < n_known_constants)
        } //while
        // Reorganize into samples[start:best.pos] + samples[best.pos:end]
        if(best.pos < end) {
            SIZE_t partition_end = end;
            SIZE_t p = start;

            while(p < partition_end) {
                if(X[{samples[p], best.feature}] <= best.threshold)
                    ++p;
                else {
                    --partition_end;
                    std::swap(samples[partition_end], samples[p]);
                }
            }
        }

        // Respect invariant for constant features: the original order of
        // element in features[:n_known_constants] must be preserved for sibling
        // and child nodes
        std::copy_n(constant_features.begin(), n_known_constants, features.begin());

        // Copy newly found constant features
        std::copy_n(features.begin() + n_known_constants, n_found_constants,
            constant_features.begin() + n_known_constants);

        // Return values
        *split = best;
        *n_constant_features = n_total_constants;
    }
};


class RandomSplitter
: public BaseDenseSplitter {
    /* Splitter for finding the best random split.*/
    virtual void node_split(double impurity, SplitRecord* split,
                         SIZE_t* n_constant_features) override {
        /* Find the best random split on node samples[start:end].*/
        // Draw random splits and pick the best
        auto& Xf = feature_values;

        SplitRecord best, current;

        SIZE_t f_i = n_features;
        // Number of features discovered to be constant during the split search
        SIZE_t n_found_constants = 0;
        // Number of features known to be constant and drawn without replacement
        SIZE_t n_drawn_constants = 0;
        SIZE_t n_known_constants = *n_constant_features;
        // n_total_constants = n_known_constants + n_found_constants
        SIZE_t n_total_constants = n_known_constants;
        SIZE_t n_visited_features = 0;

        _init_split(&best, end);

        // Sample up to max_features without replacement using a
        // Fisher-Yates-based algorithm (using the local variables `f_i` and
        // `f_j` to compute a permutation of the `features` array).
        //
        // Skip the CPU intensive evaluation of the impurity criterion for
        // features that were already detected as constant (hence not suitable
        // for good splitting) by ancestor nodes and save the information on
        // newly discovered constant features to spare computation on descendant
        // nodes.
        while(f_i > n_total_constants &&   // Stop early if remaining features
                                           // are constant
                (n_visited_features < max_features ||
                 // At least one drawn features must be non constant
                 n_visited_features <= n_found_constants + n_drawn_constants)) {

            ++n_visited_features;

            // Loop invariant: elements of features in
            // - [:n_drawn_constant[ holds drawn and known constant features;
            // - [n_drawn_constant:n_known_constant[ holds known constant
            //   features that haven't been drawn yet;
            // - [n_known_constant:n_total_constant[ holds newly found constant
            //   features;
            // - [n_total_constant:f_i[ holds features that haven't been drawn
            //   yet and aren't constant apriori.
            // - [f_i:n_features[ holds features that have been drawn
            //   and aren't constant.

            // Draw a feature at random
            SIZE_t f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           *random_state);

            if(f_j < n_known_constants) {
                // f_j in the interval [n_drawn_constants, n_known_constants[
                std::swap(features[f_j], features[n_drawn_constants]);
                ++n_drawn_constants;
            } else {
                // f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += n_found_constants;
                // f_j in the interval [n_total_constants, f_i[

                current.feature = features[f_j];

                // Find min, max
                DTYPE_t min_feature_value = X[{samples[start], current.feature}];
                DTYPE_t max_feature_value = min_feature_value;
                Xf[start] = min_feature_value;

                for(auto p: range(start + 1, end)) {
                    DTYPE_t current_feature_value =
                        X[{samples[p], current.feature}];
                    Xf[p] = current_feature_value;

                    if(current_feature_value < min_feature_value)
                        min_feature_value = current_feature_value;
                    else if(current_feature_value > max_feature_value)
                        max_feature_value = current_feature_value;
                }
                if(max_feature_value <= min_feature_value + FEATURE_THRESHOLD) {
                    features[f_j] = features[n_total_constants];
                    features[n_total_constants] = current.feature;

                    ++n_found_constants;
                    ++n_total_constants;
                } else {
                    --f_i;
                    std::swap(features[f_i], features[f_j]);

                    // Draw a random threshold
                    current.threshold = rand_uniform(min_feature_value,
                                                     max_feature_value,
                                                     *random_state);

                    if(current.threshold == max_feature_value)
                        current.threshold = min_feature_value;

                    // Partition
                    SIZE_t partition_end = end;
                    SIZE_t p = start;
                    while(p < partition_end) {
                        DTYPE_t current_feature_value = Xf[p];
                        if(current_feature_value <= current.threshold)
                            ++p;
                        else {
                            --partition_end;

                            Xf[p] = Xf[partition_end];
                            Xf[partition_end] = current_feature_value;

                            std::swap(samples[partition_end], samples[p]);
                        }
                    }
                    current.pos = partition_end;

                    // Reject if min_samples_leaf is not guaranteed
                    if((((current.pos - start) < min_samples_leaf) ||
                            ((end - current.pos) < min_samples_leaf))) {
                        continue;
                    }

                    // Evaluate split
                    criterion->reset();
                    criterion->update(current.pos);

                    // Reject if min_weight_leaf is not satisfied
                    if(((criterion->get_weighted_n_left() < min_weight_leaf) ||
                            (criterion->get_weighted_n_right() < min_weight_leaf))) {
                        continue;
                    }

                    current.improvement = criterion->impurity_improvement(impurity);

                    if(current.improvement > best.improvement) {
                        criterion->children_impurity(&current.impurity_left,
                                                         &current.impurity_right);
                        best = current;  // copy
                    }
                } // else of if(max_feature_value <= min_feature_value + FEATURE_THRESHOLD)
            } // else of if(f_j < n_known_constants)
        } //big while
        // Reorganize into samples[start:best.pos] + samples[best.pos:end]
        if(best.pos < end && current.feature != best.feature) {
            SIZE_t partition_end = end;
            SIZE_t p = start;

            while(p < partition_end) {
                if(X[{samples[p], best.feature}] <= best.threshold)
                    ++p;
                else {
                    --partition_end;
                    std::swap(samples[partition_end], samples[p]);
                }
            }
        }
        // Respect invariant for constant features: the original order of
        // element in features[:n_known_constants] must be preserved for sibling
        // and child nodes
        std::copy_n(constant_features.begin(), n_known_constants,
            features.begin());

        // Copy newly found constant features
        std::copy_n(features.begin() + n_known_constants, n_found_constants,
            constant_features.begin() + n_known_constants);

        // Return values
        *split = best;
        *n_constant_features = n_total_constants;
    }
};


class PresortBestSplitter
: public BaseDenseSplitter {
    /* Splitter for finding the best split, using presorting.*/
    sx::array_view<const DTYPE_t, 2> X_old;
    sx::matrix<int32_t> X_argsorted;

    SIZE_t n_total_samples;
    std::vector<bool> sample_mask;

    PresortBestSplitter(Criterion* criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf,
                  double min_weight_leaf,
                  std::default_random_engine* random_state)
    : BaseDenseSplitter(criterion, max_features, min_samples_leaf,
        min_weight_leaf, random_state) {
    }

    virtual void init(sx::array_view<const DTYPE_t, 2> X,
              sx::array_view<const DOUBLE_t, 2> y,
              sx::array_view<const DOUBLE_t> sample_weight) override {

        // Call parent initializer
        BaseDenseSplitter::init(X, y, sample_weight);
        // Pre-sort X
        if(!sx::is_same_view(X_old, X)) {
            X_old = X; //shallow copy
            X_argsorted = sx::sortperm<int32_t>(X, 0);
            /*
            for(auto c: range(ncols(X_argsorted))) {
                auto c = sxv::column(X_argsorted, c);
                std::iota(BEGINEND(c), 0);
                auto b = sx::make_random_access_iterator_pair(sxv::column(X, c), c.begin());
                std::sort(b, b + nrows(X), less_by_first());
            )
            */
            n_total_samples = X.extents()[0];
            sample_mask.assign(n_total_samples, false);
        }
    }

    virtual void node_split(double impurity, SplitRecord* split,
                         SIZE_t* n_constant_features) override {
        /* Find the best split on node samples[start:end].*/
        // Find the best split
        auto& Xf = feature_values;

        SplitRecord best, current;

        SIZE_t f_i = n_features;
        // Number of features discovered to be constant during the split search
        SIZE_t n_found_constants = 0;
        // Number of features known to be constant and drawn without replacement
        SIZE_t n_drawn_constants = 0;
        SIZE_t n_known_constants = *n_constant_features;
        // n_total_constants = n_known_constants + n_found_constants
        SIZE_t n_total_constants = n_known_constants;
        SIZE_t n_visited_features = 0;

        _init_split(&best, end);

        // Set sample mask
        for(auto p: range(start, end))
            sample_mask[samples[p]] = true;

        // Sample up to max_features without replacement using a
        // Fisher-Yates-based algorithm (using the local variables `f_i` and
        // `f_j` to compute a permutation of the `features` array).
        //
        // Skip the CPU intensive evaluation of the impurity criterion for
        // features that were already detected as constant (hence not suitable
        // for good splitting) by ancestor nodes and save the information on
        // newly discovered constant features to spare computation on descendant
        // nodes.
        while(f_i > n_total_constants &&  // Stop early if remaining features
                                            // are constant
                (n_visited_features < max_features or
                 // At least one drawn features must be non constant
                 n_visited_features <= n_found_constants + n_drawn_constants)) {
            ++n_visited_features;

            // Loop invariant: elements of features in
            // - [:n_drawn_constant[ holds drawn and known constant features;
            // - [n_drawn_constant:n_known_constant[ holds known constant
            //   features that haven't been drawn yet;
            // - [n_known_constant:n_total_constant[ holds newly found constant
            //   features;
            // - [n_total_constant:f_i[ holds features that haven't been drawn
            //   yet and aren't constant apriori.
            // - [f_i:n_features[ holds features that have been drawn
            //   and aren't constant.

            // Draw a feature at random
            SIZE_t f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           *random_state);

            if(f_j < n_known_constants) {
                // f_j is in [n_drawn_constants, n_known_constants[
                std::swap(features[f_j], features[n_drawn_constants]);
                ++n_drawn_constants;
            } else {
                // f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += n_found_constants;
                // f_j in the interval [n_total_constants, f_i[

                current.feature = features[f_j];

                // Extract ordering from X_argsorted
                SIZE_t p = start;

                sx::array_view<int, 2> a;
                a[{2, 3}];
                for(auto i: range(n_total_samples)) {
                    SIZE_t j = X_argsorted[{i, current.feature}];
                    if(sample_mask[j]) {
                        samples[p] = j;
                        Xf[p] = X[{j, current.feature}];
                        ++p;
                    }
                }

                // Evaluate all splits
                if(Xf[end - 1] <= Xf[start] + FEATURE_THRESHOLD) {
                    features[f_j] = features[n_total_constants];
                    features[n_total_constants] = current.feature;

                    ++n_found_constants;
                    ++n_total_constants;
                } else {
                    --f_i;
                    std::swap(features[f_i], features[f_j]);

                    criterion->reset();
                    SIZE_t p = start;

                    while(p < end) {
                        while(p + 1 < end &&
                               Xf[p + 1] <= Xf[p] + FEATURE_THRESHOLD) {
                            ++p;
                        }

                        // (p + 1 >= end) or (X[samples[p + 1], current.feature] >
                        //                    X[samples[p], current.feature])
                        ++p;
                        // (p >= end) or (X[samples[p], current.feature] >
                        //                X[samples[p - 1], current.feature])

                        if(p < end) {
                            current.pos = p;

                            // Reject if min_samples_leaf is not guaranteed
                            if ((((current.pos - start) < min_samples_leaf) ||
                                    ((end - current.pos) < min_samples_leaf))) {
                                continue;
                            }

                            criterion->update(current.pos);

                            // Reject if min_weight_leaf is not satisfied
                            if (((criterion->get_weighted_n_left() < min_weight_leaf) ||
                                    (criterion->get_weighted_n_right() < min_weight_leaf))) {
                                continue;
                            }

                            current.improvement = criterion->impurity_improvement(impurity);

                            if(current.improvement > best.improvement) {
                                criterion->children_impurity(&current.impurity_left,
                                                                 &current.impurity_right);

                                current.threshold = (Xf[p - 1] + Xf[p]) / 2.0;
                                if(current.threshold == Xf[p])
                                    current.threshold = Xf[p - 1];

                                best = current;  // copy
                            }
                        } // if(p < end)
                    } // while(p < end)
                } // else of if(Xf[end - 1] <= Xf[start] + FEATURE_THRESHOLD)
            } // else of if(f_j < n_known_constants)
        } // big while

        // Reorganize into samples[start:best.pos] + samples[best.pos:end]
        if(best.pos < end) {
            SIZE_t partition_end = end;
            SIZE_t p = start;

            while(p < partition_end) {
                if(X[{samples[p], best.feature}] <= best.threshold) {
                    ++p;
                } else {
                    --partition_end;
                    std::swap(samples[partition_end], samples[p]);
                }
            }
        }

        // Reset sample mask
        for(auto p: range(start, end))
            sample_mask[samples[p]] = false;

        // Respect invariant for constant features: the original order of
        // element in features[:n_known_constants] must be preserved for sibling
        // and child nodes
        std::copy_n(constant_features.begin(), n_known_constants, features.begin());

        // Copy newly found constant features
        std::copy_n(features.begin() + n_known_constants, n_found_constants,
            constant_features.begin() + n_known_constants);

        // Return values
        *split = best;
        *n_constant_features = n_total_constants;
    }
};

#if 0
cdef class BaseSparseSplitter(Splitter):
    // The sparse splitter works only with csc sparse matrix format
    cdef DTYPE_t* X_data
    cdef INT32_t* X_indices
    cdef INT32_t* X_indptr

    cdef SIZE_t n_total_samples

    cdef SIZE_t* index_to_samples
    cdef SIZE_t* sorted_samples

    def __cinit__(self, Criterion criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  object random_state):
        // Parent __cinit__ is automatically called

        self.X_data = NULL
        self.X_indices = NULL
        self.X_indptr = NULL

        self.n_total_samples = 0

        self.index_to_samples = NULL
        self.sorted_samples = NULL

    def __dealloc__(self):
        /* Deallocate memory.*/
        free(self.index_to_samples)
        free(self.sorted_samples)

    cdef void init(self,
                   object X,
                   np.ndarray[DOUBLE_t, ndim=2, mode="c"] y,
                   DOUBLE_t* sample_weight) except *:
        /* Initialize the splitter.*/

        // Call parent init
        Splitter.init(self, X, y, sample_weight)

        if not isinstance(X, csc_matrix):
            raise ValueError("X should be in csc format")

        cdef SIZE_t* samples = self.samples
        cdef SIZE_t n_samples = self.n_samples

        // Initialize X
        cdef np.ndarray[dtype=DTYPE_t, ndim=1] data = X.data
        cdef np.ndarray[dtype=INT32_t, ndim=1] indices = X.indices
        cdef np.ndarray[dtype=INT32_t, ndim=1] indptr = X.indptr
        cdef SIZE_t n_total_samples = X.shape[0]

        self.X_data = <DTYPE_t*> data.data
        self.X_indices = <INT32_t*> indices.data
        self.X_indptr = <INT32_t*> indptr.data
        self.n_total_samples = n_total_samples

        // Initialize auxiliary array used to perform split
        safe_realloc(&self.index_to_samples, n_total_samples * sizeof(SIZE_t))
        safe_realloc(&self.sorted_samples, n_samples * sizeof(SIZE_t))

        cdef SIZE_t* index_to_samples = self.index_to_samples
        cdef SIZE_t p
        for p in range(n_total_samples):
            index_to_samples[p] = -1

        for p in range(n_samples):
            index_to_samples[samples[p]] = p

    cdef inline SIZE_t _partition(self, double threshold,
                                  SIZE_t end_negative, SIZE_t start_positive,
                                  SIZE_t zero_pos) nogil:
        /* Partition samples[start:end] based on threshold.*/

        cdef double value
        cdef SIZE_t partition_end
        cdef SIZE_t p

        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t* index_to_samples = self.index_to_samples

        if threshold < 0.:
            p = self.start
            partition_end = end_negative
        elif threshold > 0.:
            p = start_positive
            partition_end = self.end
        else:
            // Data are already split
            return zero_pos

        while p < partition_end:
            value = Xf[p]

            if value <= threshold:
                p += 1

            else:
                partition_end -= 1

                Xf[p] = Xf[partition_end]
                Xf[partition_end] = value
                sparse_swap(index_to_samples, samples, p, partition_end)

        return partition_end

    cdef inline void extract_nnz(self, SIZE_t feature,
                                 SIZE_t* end_negative, SIZE_t* start_positive,
                                 bint* is_samples_sorted) nogil:
        /* Extract and partition values for a given feature.

        The extracted values are partitioned between negative values
        Xf[start:end_negative[0]] and positive values Xf[start_positive[0]:end].
        The samples and index_to_samples are modified according to this
        partition.

        The extraction corresponds to the intersection between the arrays
        X_indices[indptr_start:indptr_end] and samples[start:end].
        This is done efficiently using either an index_to_samples based approach
        or binary search based approach.

        Parameters
        ----------
        feature : SIZE_t,
            Index of the feature we want to extract non zero value.


        end_negative, start_positive : SIZE_t*, SIZE_t*,
            Return extracted non zero values in self.samples[start:end] where
            negative values are in self.feature_values[start:end_negative[0]]
            and positive values are in
            self.feature_values[start_positive[0]:end].

        is_samples_sorted : bint*,
            If is_samples_sorted, then self.sorted_samples[start:end] will be
            the sorted version of self.samples[start:end].

        */
        cdef SIZE_t indptr_start = self.X_indptr[feature],
        cdef SIZE_t indptr_end = self.X_indptr[feature + 1]
        cdef SIZE_t n_indices = <SIZE_t>(indptr_end - indptr_start)
        cdef SIZE_t n_samples = self.end - self.start

        // Use binary search if n_samples * log(n_indices) <
        // n_indices and index_to_samples approach otherwise.
        // O(n_samples * log(n_indices)) is the running time of binary
        // search and O(n_indices) is the running time of index_to_samples
        // approach.
        if ((1 - is_samples_sorted[0]) * n_samples * log(n_samples) +
                n_samples * log(n_indices) < EXTRACT_NNZ_SWITCH * n_indices):
            extract_nnz_binary_search(self.X_indices, self.X_data,
                                      indptr_start, indptr_end,
                                      self.samples, self.start, self.end,
                                      self.index_to_samples,
                                      self.feature_values,
                                      end_negative, start_positive,
                                      self.sorted_samples, is_samples_sorted)

        // Using an index to samples  technique to extract non zero values
        // index_to_samples is a mapping from X_indices to samples
        else:
            extract_nnz_index_to_samples(self.X_indices, self.X_data,
                                         indptr_start, indptr_end,
                                         self.samples, self.start, self.end,
                                         self.index_to_samples,
                                         self.feature_values,
                                         end_negative, start_positive)


cdef int compare_SIZE_t(const void* a, const void* b) nogil:
    /* Comparison function for sort.*/
    return <int>((<SIZE_t*>a)[0] - (<SIZE_t*>b)[0])


cdef inline void binary_search(INT32_t* sorted_array,
                               INT32_t start, INT32_t end,
                               SIZE_t value, SIZE_t* index,
                               INT32_t* new_start) nogil:
    /* Return the index of value in the sorted array.

    If not found, return -1. new_start is the last pivot + 1
    */
    cdef INT32_t pivot
    index[0] = -1
    while start < end:
        pivot = start + (end - start) / 2

        if sorted_array[pivot] == value:
            index[0] = pivot
            start = pivot + 1
            break

        if sorted_array[pivot] < value:
            start = pivot + 1
        else:
            end = pivot
    new_start[0] = start


cdef inline void extract_nnz_index_to_samples(INT32_t* X_indices,
                                              DTYPE_t* X_data,
                                              INT32_t indptr_start,
                                              INT32_t indptr_end,
                                              SIZE_t* samples,
                                              SIZE_t start,
                                              SIZE_t end,
                                              SIZE_t* index_to_samples,
                                              DTYPE_t* Xf,
                                              SIZE_t* end_negative,
                                              SIZE_t* start_positive) nogil:
    /* Extract and partition values for a feature using index_to_samples.

    Complexity is O(indptr_end - indptr_start).
    */
    cdef INT32_t k
    cdef SIZE_t index
    cdef SIZE_t end_negative_ = start
    cdef SIZE_t start_positive_ = end

    for k in range(indptr_start, indptr_end):
        if start <= index_to_samples[X_indices[k]] < end:
            if X_data[k] > 0:
                start_positive_ -= 1
                Xf[start_positive_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, start_positive_)


            elif X_data[k] < 0:
                Xf[end_negative_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, end_negative_)
                end_negative_ += 1

    // Returned values
    end_negative[0] = end_negative_
    start_positive[0] = start_positive_


cdef inline void extract_nnz_binary_search(INT32_t* X_indices,
                                           DTYPE_t* X_data,
                                           INT32_t indptr_start,
                                           INT32_t indptr_end,
                                           SIZE_t* samples,
                                           SIZE_t start,
                                           SIZE_t end,
                                           SIZE_t* index_to_samples,
                                           DTYPE_t* Xf,
                                           SIZE_t* end_negative,
                                           SIZE_t* start_positive,
                                           SIZE_t* sorted_samples,
                                           bint* is_samples_sorted) nogil:
    /* Extract and partition values for a given feature using binary search.

    If n_samples = end - start and n_indices = indptr_end - indptr_start,
    the complexity is

        O((1 - is_samples_sorted[0]) * n_samples * log(n_samples) +
          n_samples * log(n_indices)).
    */
    cdef SIZE_t n_samples

    if not is_samples_sorted[0]:
        n_samples = end - start
        memcpy(sorted_samples + start, samples + start,
               n_samples * sizeof(SIZE_t))
        qsort(sorted_samples + start, n_samples, sizeof(SIZE_t),
              compare_SIZE_t)
        is_samples_sorted[0] = 1

    while (indptr_start < indptr_end and
           sorted_samples[start] > X_indices[indptr_start]):
        indptr_start += 1

    while (indptr_start < indptr_end and
           sorted_samples[end - 1] < X_indices[indptr_end - 1]):
        indptr_end -= 1

    cdef SIZE_t p = start
    cdef SIZE_t index
    cdef SIZE_t k
    cdef SIZE_t end_negative_ = start
    cdef SIZE_t start_positive_ = end

    while (p < end and indptr_start < indptr_end):
        // Find index of sorted_samples[p] in X_indices
        binary_search(X_indices, indptr_start, indptr_end,
                      sorted_samples[p], &k, &indptr_start)

        if k != -1:
             // If k != -1, we have found a non zero value

            if X_data[k] > 0:
                start_positive_ -= 1
                Xf[start_positive_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, start_positive_)


            elif X_data[k] < 0:
                Xf[end_negative_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, end_negative_)
                end_negative_ += 1
        p += 1

    // Returned values
    end_negative[0] = end_negative_
    start_positive[0] = start_positive_


cdef inline void sparse_swap(SIZE_t* index_to_samples, SIZE_t* samples,
                             SIZE_t pos_1, SIZE_t pos_2) nogil  :
    /* Swap sample pos_1 and pos_2 preserving sparse invariant.*/
    samples[pos_1], samples[pos_2] =  samples[pos_2], samples[pos_1]
    index_to_samples[samples[pos_1]] = pos_1
    index_to_samples[samples[pos_2]] = pos_2


cdef class BestSparseSplitter(BaseSparseSplitter):
    /* Splitter for finding the best split, using the sparse data.*/

    def __reduce__(self):
        return (BestSparseSplitter, (self.criterion,
                                     self.max_features,
                                     self.min_samples_leaf,
                                     self.min_weight_leaf,
                                     self.random_state), self.__getstate__())

    cdef void node_split(self, double impurity, SplitRecord* split,
                         SIZE_t* n_constant_features) nogil:
        /* Find the best split on node samples[start:end], using sparse
           features.
        */
        // Find the best split
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef INT32_t* X_indices = self.X_indices
        cdef INT32_t* X_indptr = self.X_indptr
        cdef DTYPE_t* X_data = self.X_data

        cdef SIZE_t* features = self.features
        cdef SIZE_t* constant_features = self.constant_features
        cdef SIZE_t n_features = self.n_features

        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t* sorted_samples = self.sorted_samples
        cdef SIZE_t* index_to_samples = self.index_to_samples
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef SplitRecord best, current
        _init_split(&best, end)

        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j, p, tmp
        cdef SIZE_t n_visited_features = 0
        // Number of features discovered to be constant during the split search
        cdef SIZE_t n_found_constants = 0
        // Number of features known to be constant and drawn without replacement
        cdef SIZE_t n_drawn_constants = 0
        cdef SIZE_t n_known_constants = n_constant_features[0]
        // n_total_constants = n_known_constants + n_found_constants
        cdef SIZE_t n_total_constants = n_known_constants
        cdef DTYPE_t current_feature_value

        cdef SIZE_t p_next
        cdef SIZE_t p_prev
        cdef bint is_samples_sorted = 0  // indicate is sorted_samples is
                                         // inititialized

        // We assume implicitely that end_positive = end and
        // start_negative = start
        cdef SIZE_t start_positive
        cdef SIZE_t end_negative

        // Sample up to max_features without replacement using a
        // Fisher-Yates-based algorithm (using the local variables `f_i` and
        // `f_j` to compute a permutation of the `features` array).
        //
        // Skip the CPU intensive evaluation of the impurity criterion for
        // features that were already detected as constant (hence not suitable
        // for good splitting) by ancestor nodes and save the information on
        // newly discovered constant features to spare computation on descendant
        // nodes.
        while (f_i > n_total_constants and  // Stop early if remaining features
                                            // are constant
                (n_visited_features < max_features or
                 // At least one drawn features must be non constant
                 n_visited_features <= n_found_constants + n_drawn_constants)):

            n_visited_features += 1

            // Loop invariant: elements of features in
            // - [:n_drawn_constant[ holds drawn and known constant features;
            // - [n_drawn_constant:n_known_constant[ holds known constant
            //   features that haven't been drawn yet;
            // - [n_known_constant:n_total_constant[ holds newly found constant
            //   features;
            // - [n_total_constant:f_i[ holds features that haven't been drawn
            //   yet and aren't constant apriori.
            // - [f_i:n_features[ holds features that have been drawn
            //   and aren't constant.

            // Draw a feature at random
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           random_state)

            if f_j < n_known_constants:
                // f_j in the interval [n_drawn_constants, n_known_constants[
                tmp = features[f_j]
                features[f_j] = features[n_drawn_constants]
                features[n_drawn_constants] = tmp

                n_drawn_constants += 1

            else:
                // f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += n_found_constants
                // f_j in the interval [n_total_constants, f_i[

                current.feature = features[f_j]
                self.extract_nnz(current.feature,
                                 &end_negative, &start_positive,
                                 &is_samples_sorted)

                // Sort the positive and negative parts of `Xf`
                sort(Xf + start, samples + start, end_negative - start)
                sort(Xf + start_positive, samples + start_positive,
                     end - start_positive)

                // Update index_to_samples to take into account the sort
                for p in range(start, end_negative):
                    index_to_samples[samples[p]] = p
                for p in range(start_positive, end):
                    index_to_samples[samples[p]] = p

                // Add one or two zeros in Xf, if there is any
                if end_negative < start_positive:
                    start_positive -= 1
                    Xf[start_positive] = 0.

                    if end_negative != start_positive:
                        Xf[end_negative] = 0.
                        end_negative += 1

                if Xf[end - 1] <= Xf[start] + FEATURE_THRESHOLD:
                    features[f_j] = features[n_total_constants]
                    features[n_total_constants] = current.feature

                    n_found_constants += 1
                    n_total_constants += 1

                else:
                    f_i -= 1
                    features[f_i], features[f_j] = features[f_j], features[f_i]

                    // Evaluate all splits
                    self.criterion.reset()
                    p = start

                    while p < end:
                        if p + 1 != end_negative:
                            p_next = p + 1
                        else:
                            p_next = start_positive

                        while (p_next < end and
                               Xf[p_next] <= Xf[p] + FEATURE_THRESHOLD):
                            p = p_next
                            if p + 1 != end_negative:
                                p_next = p + 1
                            else:
                                p_next = start_positive


                        // (p_next >= end) or (X[samples[p_next], current.feature] >
                        //                     X[samples[p], current.feature])
                        p_prev = p
                        p = p_next
                        // (p >= end) or (X[samples[p], current.feature] >
                        //                X[samples[p_prev], current.feature])


                        if p < end:
                            current.pos = p

                            // Reject if min_samples_leaf is not guaranteed
                            if (((current.pos - start) < min_samples_leaf) or
                                    ((end - current.pos) < min_samples_leaf)):
                                continue

                            self.criterion.update(current.pos)

                            // Reject if min_weight_leaf is not satisfied
                            if ((self.criterion.weighted_n_left < min_weight_leaf) or
                                    (self.criterion.weighted_n_right < min_weight_leaf)):
                                continue

                            current.improvement = self.criterion.impurity_improvement(impurity)
                            if current.improvement > best.improvement:
                                self.criterion.children_impurity(&current.impurity_left,
                                                                 &current.impurity_right)

                                current.threshold = (Xf[p_prev] + Xf[p]) / 2.0
                                if current.threshold == Xf[p]:
                                    current.threshold = Xf[p_prev]

                                best = current

        // Reorganize into samples[start:best.pos] + samples[best.pos:end]
        if best.pos < end:
            self.extract_nnz(best.feature, &end_negative, &start_positive,
                             &is_samples_sorted)

            self._partition(best.threshold, end_negative, start_positive,
                            best.pos)

        // Respect invariant for constant features: the original order of
        // element in features[:n_known_constants] must be preserved for sibling
        // and child nodes
        memcpy(features, constant_features, sizeof(SIZE_t) * n_known_constants)

        // Copy newly found constant features
        memcpy(constant_features + n_known_constants,
               features + n_known_constants,
               sizeof(SIZE_t) * n_found_constants)

        // Return values
        split[0] = best
        n_constant_features[0] = n_total_constants


cdef class RandomSparseSplitter(BaseSparseSplitter):
    /* Splitter for finding a random split, using the sparse data.*/

    def __reduce__(self):
        return (RandomSparseSplitter, (self.criterion,
                                       self.max_features,
                                       self.min_samples_leaf,
                                       self.min_weight_leaf,
                                       self.random_state), self.__getstate__())

    cdef void node_split(self, double impurity, SplitRecord* split,
                         SIZE_t* n_constant_features) nogil:
        /* Find a random split on node samples[start:end], using sparse
           features.
        */
        // Find the best split
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef INT32_t* X_indices = self.X_indices
        cdef INT32_t* X_indptr = self.X_indptr
        cdef DTYPE_t* X_data = self.X_data

        cdef SIZE_t* features = self.features
        cdef SIZE_t* constant_features = self.constant_features
        cdef SIZE_t n_features = self.n_features

        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t* sorted_samples = self.sorted_samples
        cdef SIZE_t* index_to_samples = self.index_to_samples
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef SplitRecord best, current
        _init_split(&best, end)

        cdef DTYPE_t current_feature_value

        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j, p, tmp
        cdef SIZE_t n_visited_features = 0
        // Number of features discovered to be constant during the split search
        cdef SIZE_t n_found_constants = 0
        // Number of features known to be constant and drawn without replacement
        cdef SIZE_t n_drawn_constants = 0
        cdef SIZE_t n_known_constants = n_constant_features[0]
        // n_total_constants = n_known_constants + n_found_constants
        cdef SIZE_t n_total_constants = n_known_constants
        cdef SIZE_t partition_end

        cdef DTYPE_t min_feature_value
        cdef DTYPE_t max_feature_value

        cdef bint is_samples_sorted = 0  // indicate that sorted_samples is
                                         // inititialized

        // We assume implicitely that end_positive = end and
        // start_negative = start
        cdef SIZE_t start_positive
        cdef SIZE_t end_negative

        // Sample up to max_features without replacement using a
        // Fisher-Yates-based algorithm (using the local variables `f_i` and
        // `f_j` to compute a permutation of the `features` array).
        //
        // Skip the CPU intensive evaluation of the impurity criterion for
        // features that were already detected as constant (hence not suitable
        // for good splitting) by ancestor nodes and save the information on
        // newly discovered constant features to spare computation on descendant
        // nodes.
        while (f_i > n_total_constants and  // Stop early if remaining features
                                            // are constant
                (n_visited_features < max_features or
                 // At least one drawn features must be non constant
                 n_visited_features <= n_found_constants + n_drawn_constants)):

            n_visited_features += 1

            // Loop invariant: elements of features in
            // - [:n_drawn_constant[ holds drawn and known constant features;
            // - [n_drawn_constant:n_known_constant[ holds known constant
            //   features that haven't been drawn yet;
            // - [n_known_constant:n_total_constant[ holds newly found constant
            //   features;
            // - [n_total_constant:f_i[ holds features that haven't been drawn
            //   yet and aren't constant apriori.
            // - [f_i:n_features[ holds features that have been drawn
            //   and aren't constant.

            // Draw a feature at random
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           random_state)

            if f_j < n_known_constants:
                // f_j in the interval [n_drawn_constants, n_known_constants[
                tmp = features[f_j]
                features[f_j] = features[n_drawn_constants]
                features[n_drawn_constants] = tmp

                n_drawn_constants += 1

            else:
                // f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += n_found_constants
                // f_j in the interval [n_total_constants, f_i[

                current.feature = features[f_j]

                self.extract_nnz(current.feature,
                                 &end_negative, &start_positive,
                                 &is_samples_sorted)

                // Add one or two zeros in Xf, if there is any
                if end_negative < start_positive:
                    start_positive -= 1
                    Xf[start_positive] = 0.

                    if end_negative != start_positive:
                        Xf[end_negative] = 0.
                        end_negative += 1

                // Find min, max in Xf[start:end_negative]
                min_feature_value = Xf[start]
                max_feature_value = min_feature_value

                for p in range(start, end_negative):
                    current_feature_value = Xf[p]

                    if current_feature_value < min_feature_value:
                        min_feature_value = current_feature_value
                    elif current_feature_value > max_feature_value:
                        max_feature_value = current_feature_value

                // Update min, max given Xf[start_positive:end]
                for p in range(start_positive, end):
                    current_feature_value = Xf[p]

                    if current_feature_value < min_feature_value:
                        min_feature_value = current_feature_value
                    elif current_feature_value > max_feature_value:
                        max_feature_value = current_feature_value

                if max_feature_value <= min_feature_value + FEATURE_THRESHOLD:
                    features[f_j] = features[n_total_constants]
                    features[n_total_constants] = current.feature

                    n_found_constants += 1
                    n_total_constants += 1

                else:
                    f_i -= 1
                    features[f_i], features[f_j] = features[f_j], features[f_i]

                    // Draw a random threshold
                    current.threshold = rand_uniform(min_feature_value,
                                                     max_feature_value,
                                                     random_state)

                    if current.threshold == max_feature_value:
                        current.threshold = min_feature_value

                    // Partition
                    current.pos = self._partition(current.threshold,
                                                  end_negative,
                                                  start_positive,
                                                  start_positive +
                                                  (Xf[start_positive] == 0.))

                    // Reject if min_samples_leaf is not guaranteed
                    if (((current.pos - start) < min_samples_leaf) or
                            ((end - current.pos) < min_samples_leaf)):
                        continue

                    // Evaluate split
                    self.criterion.reset()
                    self.criterion.update(current.pos)

                    // Reject if min_weight_leaf is not satisfied
                    if ((self.criterion.weighted_n_left < min_weight_leaf) or
                            (self.criterion.weighted_n_right < min_weight_leaf)):
                        continue

                    current.improvement = self.criterion.impurity_improvement(impurity)

                    if current.improvement > best.improvement:
                        self.criterion.children_impurity(&current.impurity_left,
                                                         &current.impurity_right)
                        best = current

        // Reorganize into samples[start:best.pos] + samples[best.pos:end]
        if best.pos < end and current.feature != best.feature:
            self.extract_nnz(best.feature, &end_negative, &start_positive,
                             &is_samples_sorted)

            self._partition(best.threshold, end_negative, start_positive,
                            best.pos)

        // Respect invariant for constant features: the original order of
        // element in features[:n_known_constants] must be preserved for sibling
        // and child nodes
        memcpy(features, constant_features, sizeof(SIZE_t) * n_known_constants)

        // Copy newly found constant features
        memcpy(constant_features + n_known_constants,
               features + n_known_constants,
               sizeof(SIZE_t) * n_found_constants)

        // Return values
        split[0] = best
        n_constant_features[0] = n_total_constants
#endif

// =============================================================================
// Tree builders
// =============================================================================

TreeBuilder::TreeBuilder(Splitter* splitter, SIZE_t min_samples_split,
              SIZE_t min_samples_leaf, double min_weight_leaf,
              SIZE_t max_depth)
              : splitter(splitter)
              , min_samples_split(min_samples_split)
              , min_samples_leaf(min_samples_leaf)
              , min_weight_leaf(min_weight_leaf)
              , max_depth(max_depth)
              {}

void TreeBuilder::
build(Tree& tree, sx::matrix_view<const DTYPE_t> X, sx::matrix_view<const DOUBLE> y) {
    build(tree, X, y, sx::array_view<const DOUBLE>()); //pass empty sample_weight
}

    void TreeBuilder::_check_input(sx::matrix_view<const DTYPE_t> X,
        sx::matrix_view<const DOUBLE> y,
                             sx::array_view<const DOUBLE> sample_weight) {
        assert(X.extents(0) == y.extents(0));
        if(!sample_weight.empty())
            assert(X.extents(0) == sample_weight.extents(0));
        // todo:
        // the original code did this:
        // if issparse(X): for this check the original code
        // else if X's type is not DTYPE (= float32)
        //     copy X to a fortran-order DTYPE X
        // if y's type is not DOUBLE (float64) or not contiguous (C-order)
        //     copy y to a contiguous DOUBLE C-order array
        // if sample_weight is no null and not DOUBLE or not contiguous
        //     copy sample_weight to a DOUBLE C-order array
    }

    // Depth first builder ---------------------------------------------------------
class DepthFirstTreeBuilder : public TreeBuilder {
    /* Build a decision tree in depth-first fashion.*/

    DepthFirstTreeBuilder(Splitter* splitter, SIZE_t min_samples_split,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  SIZE_t max_depth)
        : TreeBuilder(splitter,
            min_samples_split, min_samples_leaf,
        min_weight_leaf, max_depth)
        {}

    virtual void build(Tree& tree, sx::matrix_view<const DTYPE_t> X,
        sx::matrix_view<const DOUBLE> y,
                             sx::array_view<const DOUBLE> sample_weight) {
        /* Build a decision tree from the training set (X, y).*/

        // check input
        _check_input(X, y, sample_weight);

        // Initial capacity
        int init_capacity;

        if(tree.max_depth <= 10)
            init_capacity = (1 << (tree.max_depth + 1)) - 1;
        else
            init_capacity = 2047;

        tree.reserve(init_capacity);

        // Recursive partition (without actual recursion)
        splitter->init(X, y, sample_weight);

        SplitRecord split;

        bool first = true;
        SIZE_t max_depth_seen = -1;

        Stack stack(INITIAL_STACK_SIZE);

        // push root node onto stack
        stack.push(0, splitter->get_n_samples(), 0, _TREE_UNDEFINED, 0, INFINITY, 0);

            while(!stack.is_empty()) {
                StackRecord stack_record;
                stack.pop(&stack_record);
                const auto& sr = stack_record;

                auto impurity = sr.impurity;
                auto n_constant_features = sr.n_constant_features;

                const SIZE_t n_node_samples = sr.end - sr.start;
                double weighted_n_node_samples;
                splitter->node_reset(sr.start, sr.end, &weighted_n_node_samples);

                bool is_leaf = ((sr.depth >= max_depth) ||
                           (n_node_samples < min_samples_split) ||
                           (n_node_samples < 2 * min_samples_leaf) ||
                           (weighted_n_node_samples < min_weight_leaf));

                if(first) {
                    impurity = splitter->node_impurity();
                    first = false;
                }
                is_leaf = is_leaf || (impurity <= MIN_IMPURITY_SPLIT);

                if(!is_leaf) {
                    splitter->node_split(impurity, &split, &n_constant_features);
                    is_leaf = is_leaf || (split.pos >= sr.end);
                }
                assert(false); //split uninitialized?
                SIZE_t node_id = tree._add_node(sr.parent, sr.is_left, is_leaf, split.feature,
                                         split.threshold, impurity, n_node_samples,
                                         weighted_n_node_samples);

                assert(node_id >= 0); //because out of memory will not be signalled this way

                // Store value for all nodes, to facilitate tree/model
                // inspection and interpretation
                splitter->node_value(tree.value(node_id, sx::all));

                if(!is_leaf) {
                    // Push right child on stack
                    stack.push(split.pos, sr.end, sr.depth + 1, node_id, 0,
                                    split.impurity_right, n_constant_features);

                    // Push left child on stack
                    stack.push(sr.start, split.pos, sr.depth + 1, node_id, 1,
                                    split.impurity_left, n_constant_features);
                }
                if(sr.depth > max_depth_seen)
                    max_depth_seen = sr.depth;
            } // while
            tree.max_depth = max_depth_seen;
}
};
// Best first builder ----------------------------------------------------------

class BestFirstTreeBuilder
: public TreeBuilder {
    /* Build a decision tree in best-first fashion.

    The best node to expand is given by the node at the frontier that has the
    highest impurity improvement.

    NOTE: this TreeBuilder will ignore ``tree.max_depth`` .
    */
    SIZE_t max_leaf_nodes;

    BestFirstTreeBuilder(Splitter* splitter, SIZE_t min_samples_split,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  SIZE_t max_depth, SIZE_t max_leaf_nodes)
        : TreeBuilder(splitter, min_samples_split,
            min_samples_leaf, min_weight_leaf,
            max_depth)
        , max_leaf_nodes(max_leaf_nodes)
        {}

    virtual void build(Tree& tree, sx::matrix_view<const DTYPE_t> X,
                       sx::matrix_view<const DOUBLE> y,
                       sx::array_view<const DOUBLE> sample_weight) override {
        /* Build a decision tree from the training set (X, y).*/

        // check input
        _check_input(X, y, sample_weight);

        // Recursive partition (without actual recursion)
        splitter->init(X, y, sample_weight);

        PriorityHeap frontier(INITIAL_STACK_SIZE);
        PriorityHeapRecord split_node_left;

        //SIZE_t n_node_samples = splitter.n_samples
        SIZE_t max_split_nodes = max_leaf_nodes - 1;
        SIZE_t max_depth_seen = -1;
        //int rc = 0

        // Initial capacity
        SIZE_t init_capacity = max_split_nodes + max_leaf_nodes;
        tree.reserve(init_capacity);

        // add root to frontier
        _add_split_node(splitter, tree, 0, splitter->get_n_samples(),
                                  INFINITY, IS_FIRST, IS_LEFT, NULL, 0,
                        &split_node_left);
        frontier.push(split_node_left);

        while(!frontier.is_empty()) {
            PriorityHeapRecord record;
            frontier.pop(&record);

            Node* node = &tree.nodes[record.node_id];
            bool is_leaf = (record.is_leaf || max_split_nodes <= 0);

            if(is_leaf) {
                // Node is not expandable; set node as leaf
                node->left_child = _TREE_LEAF;
                node->right_child = _TREE_LEAF;
                node->feature = _TREE_UNDEFINED;
                node->threshold = _TREE_UNDEFINED;
            } else {
                // Node is expandable

                // Decrement number of split nodes available
                --max_split_nodes;

                // Compute left split node
                _add_split_node(splitter, tree,
                                          record.start, record.pos,
                                          record.impurity_left,
                                          IS_NOT_FIRST, IS_LEFT, node,
                                          record.depth + 1,
                                          &split_node_left);

                // tree.nodes may have changed
                node = &tree.nodes[record.node_id];

                // Compute right split node
                PriorityHeapRecord split_node_right;
                _add_split_node(splitter, tree, record.pos,
                                          record.end,
                                          record.impurity_right,
                                          IS_NOT_FIRST, IS_NOT_LEFT, node,
                                          record.depth + 1,
                                &split_node_right);

                // Add nodes to queue
                frontier.push(split_node_left);
                frontier.push(split_node_right);
            }
            if(record.depth > max_depth_seen)
                max_depth_seen = record.depth;
        } // while
        tree.max_depth = max_depth_seen;
    }


    void _add_split_node(Splitter* splitter, Tree& tree,
                                    SIZE_t start, SIZE_t end, double impurity,
                                    bool is_first, bool is_left, Node* parent,
                                    SIZE_t depth,
                                    PriorityHeapRecord* res) {
        /* Adds node w/ partition ``[start, end)`` to the frontier. */
        double weighted_n_node_samples;
        splitter->node_reset(start, end, &weighted_n_node_samples);

        if(is_first)
            impurity = splitter->node_impurity();

        SIZE_t n_node_samples = end - start;
        bool is_leaf = ((depth > max_depth) ||
                   (n_node_samples < min_samples_split) ||
                   (n_node_samples < 2 * min_samples_leaf) ||
                   (weighted_n_node_samples < min_weight_leaf) ||
                   (impurity <= MIN_IMPURITY_SPLIT));

        SplitRecord split;
        if(!is_leaf) {
            SIZE_t n_constant_features;
            splitter->node_split(impurity, &split, &n_constant_features);
            is_leaf = is_leaf || (split.pos >= end);
        }
        assert(false); //split uninitialized?
        SIZE_t node_id = tree._add_node(
            parent != NULL ? parent - tree.nodes.data() : _TREE_UNDEFINED,
            is_left, is_leaf,
            split.feature, split.threshold, impurity, n_node_samples,
                                        weighted_n_node_samples);
        assert(node_id >=0 ); //memory errors not signalled by negative node_id

        // compute values also for split nodes (might become leafs later).
        splitter->node_value(tree.value(node_id, sx::all));

        res->node_id = node_id;
        res->start = start;
        res->end = end;
        res->depth = depth;
        res->impurity = impurity;

        if(!is_leaf) {
            // is split node
            res->pos = split.pos;
            res->is_leaf = false;
            res->improvement = split.improvement;
            res->impurity_left = split.impurity_left;
            res->impurity_right = split.impurity_right;
        } else {
            // is leaf => 0 improvement
            res->pos = end;
            res->is_leaf = true;
            res->improvement = 0.0;
            res->impurity_left = impurity;
            res->impurity_right = impurity;
        }
    }
};
// =============================================================================
// Tree
// =============================================================================

    Tree::Tree(int n_features, NClassesRef n_classes)
    : n_features(n_features)
    , n_classes(n_classes)
    , max_depth(0) {
    }

    void Tree::reserve(SIZE_t capacity) {
        nodes.reserve(capacity);
        value.reserve({capacity, n_classes.n_elements()});
    }

    SIZE_t Tree::_add_node(SIZE_t parent, bool is_left, bool is_leaf,
                          SIZE_t feature, double threshold, double impurity,
                          SIZE_t n_node_samples, double weighted_n_node_samples) {
        SIZE_t node_id = nodes.size();

        nodes.emplace_back();
        value.resize({value.extents(0) + 1, n_classes.n_elements()}, sx::array_layout::c_order, 0.0);
        Node& node = nodes.back();
        node.impurity = impurity;
        node.n_node_samples = n_node_samples;
        node.weighted_n_node_samples = weighted_n_node_samples;

        if(parent != _TREE_UNDEFINED) {
            if(is_left)
                nodes[parent].left_child = node_id;
            else
                nodes[parent].right_child = node_id;
        }

        if(is_leaf) {
            node.left_child = _TREE_LEAF;
            node.right_child = _TREE_LEAF;
            node.feature = _TREE_UNDEFINED;
            node.threshold = _TREE_UNDEFINED;
        } else {
            // left_child and right_child will be set later
            node.feature = feature;
            node.threshold = threshold;
        }

        return node_id;
    }

    sx::matrix<double> Tree::predict(sx::array_view<const DTYPE_t, 2> X) const {

        sx::matrix<double> out(sx::matrix<double>::extents_type{X.extents(0), value.extents(1)});
        for(auto i: range(X.extents(0)))
        {
            auto node_id = apply_single(X(i, sx::all));
            if(value.extents(1) == 1)
                out(i, 0) = value(node_id, 0);
            else {
                out(i, sx::all) <<= value(node_id, sx::all);
            }
        }
        return out;
    }

    SIZE_t Tree::_apply_dense_single(sx::array_view<const DTYPE_t> x) const {
        auto node = nodes.data();
        // While node not a leaf
        while(node->left_child != _TREE_LEAF) {
            // ... and node.right_child != _TREE_LEAF:
            if(x[node->feature] <= node->threshold)
                node = &nodes[node->left_child];
            else
                node = &nodes[node->right_child];
        }
        return (SIZE_t)(node - nodes.data());  // node offset
    }

    std::vector<SIZE_t> Tree::apply(sx::array_view<const DTYPE_t, 2> X) const {
#if 0
        if issparse(X):
            return self._apply_sparse_csr(X)
        else:
#endif
            return _apply_dense(X);
    }

    std::vector<SIZE_t> Tree::_apply_dense(sx::array_view<const DTYPE_t, 2> X) const {
        // Extract input
        SIZE_t n_samples = X.extents(0);

        // Initialize output
        std::vector<SIZE_t> out(n_samples);

        for(auto i: range(n_samples))
            out[i] = _apply_dense_single(X(i, sx::all));
        return out;
    }

#if 0
    cdef inline np.ndarray _apply_sparse_csr(self, object X):
        /* Finds the terminal region (=leaf node) for each sample in sparse X.

        */
        // Check input
        if not isinstance(X, csr_matrix):
            raise ValueError("X should be in csr_matrix format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        // Extract input
        cdef np.ndarray[ndim=1, dtype=DTYPE_t] X_data_ndarray = X.data
        cdef np.ndarray[ndim=1, dtype=INT32_t] X_indices_ndarray  = X.indices
        cdef np.ndarray[ndim=1, dtype=INT32_t] X_indptr_ndarray  = X.indptr

        cdef DTYPE_t* X_data = <DTYPE_t*>X_data_ndarray.data
        cdef INT32_t* X_indices = <INT32_t*>X_indices_ndarray.data
        cdef INT32_t* X_indptr = <INT32_t*>X_indptr_ndarray.data

        cdef SIZE_t n_samples = X.shape[0]
        cdef SIZE_t n_features = X.shape[1]

        // Initialize output
        cdef np.ndarray[SIZE_t, ndim=1] out = np.zeros((n_samples,),
                                                       dtype=np.intp)
        cdef SIZE_t* out_ptr = <SIZE_t*> out.data

        // Initialize auxiliary data-structure
        cdef DTYPE_t feature_value = 0.
        cdef Node* node = NULL
        cdef DTYPE_t* X_sample = NULL
        cdef SIZE_t i = 0
        cdef INT32_t k = 0

        // feature_to_sample as a data structure records the last seen sample
        // for each feature; functionally, it is an efficient way to identify
        // which features are nonzero in the present sample.
        cdef SIZE_t* feature_to_sample = NULL

        safe_realloc(&X_sample, n_features * sizeof(DTYPE_t))
        safe_realloc(&feature_to_sample, n_features * sizeof(SIZE_t))

        with nogil:
            memset(feature_to_sample, -1, n_features * sizeof(SIZE_t))

            for i in range(n_samples):
                node = self.nodes

                for k in range(X_indptr[i], X_indptr[i + 1]):
                    feature_to_sample[X_indices[k]] = i
                    X_sample[X_indices[k]] = X_data[k]

                // While node not a leaf
                while node.left_child != _TREE_LEAF:
                    // ... and node.right_child != _TREE_LEAF:
                    if feature_to_sample[node.feature] == i:
                        feature_value = X_sample[node.feature]

                    else:
                        feature_value = 0.

                    if feature_value <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                out_ptr[i] = <SIZE_t>(node - self.nodes)  // node offset

            // Free auxiliary arrays
            free(X_sample)
            free(feature_to_sample)

        return out
#endif
    std::vector<double> Tree::compute_feature_importances(bool normalize) {
        Node* node = &nodes.front();
        Node* end_node = node + nodes.size();

        std::vector<double> importances(n_features, 0.0);

        while(node != end_node) {
            if(node->left_child != _TREE_LEAF) {
                // ... and node.right_child != _TREE_LEAF:
                Node* left = &nodes[node->left_child];
                Node* right = &nodes[node->right_child];

                importances[node->feature] += (
                    node->weighted_n_node_samples * node->impurity -
                    left->weighted_n_node_samples * left->impurity -
                    right->weighted_n_node_samples * right->impurity);
            }
            ++node;
        }

        for(auto&d:importances)
            d /= nodes[0].weighted_n_node_samples;

        if(normalize) {
            auto normalizer = std::accumulate(BEGINEND(importances), 0.0);

            if(normalizer > 0.0) {
                // Avoid dividing by zero (e.g., when root is pure)
                for(auto&d: importances) d /= normalizer;
            }
        }
        return importances;
    }
#if 0
    cdef np.ndarray _get_node_ndarray(self):
        /* Wraps nodes as a NumPy struct array.

        The array keeps a reference to this Tree, which manages the underlying
        memory. Individual fields are publicly accessible as properties of the
        Tree.
        */
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.node_count
        cdef np.npy_intp strides[1]
        strides[0] = sizeof(Node)
        cdef np.ndarray arr
        Py_INCREF(NODE_DTYPE)
        arr = PyArray_NewFromDescr(np.ndarray, <np.dtype> NODE_DTYPE, 1, shape,
                                   strides, <void*> self.nodes,
                                   np.NPY_DEFAULT, None)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr

// =============================================================================
// Utils
// =============================================================================

// safe_realloc(&p, n) resizes the allocation of p to n * sizeof(*p) bytes or
// raises a MemoryError. It never calls free, since that's __dealloc__'s job.
//   cdef DTYPE_t *p = NULL
//   safe_realloc(&p, n)
// is equivalent to p = malloc(n * sizeof(*p)) with error checking.
ctypedef fused realloc_ptr:
    // Add pointer types here as needed.
    (DTYPE_t*)
    (SIZE_t*)
    (unsigned char*)

cdef realloc_ptr safe_realloc(realloc_ptr* p, size_t nelems) except *:
    // sizeof(realloc_ptr[0]) would be more like idiomatic C, but causes Cython
    // 0.20.1 to crash.
    cdef size_t nbytes = nelems * sizeof(p[0][0])
    if nbytes / sizeof(p[0][0]) != nelems:
        // Overflow in the multiplication
        raise MemoryError("could not allocate (%d * %d) bytes"
                          % (nelems, sizeof(p[0][0])))
    cdef realloc_ptr tmp = <realloc_ptr>realloc(p[0], nbytes)
    if tmp == NULL:
        raise MemoryError("could not allocate %d bytes" % nbytes)

    p[0] = tmp
    return tmp  // for convenience


def _realloc_test():
    // Helper for tests. Tries to allocate <size_t>(-1) / 2 * sizeof(size_t)
    // bytes, which will always overflow.
    cdef SIZE_t* p = NULL
    safe_realloc(&p, <size_t>(-1) / 2)
    if p != NULL:
        free(p)
        assert False


// rand_r replacement using a 32bit XorShift generator
// See http://www.jstatsoft.org/v08/i14/paper for details
cdef inline UINT32_t our_rand_r(UINT32_t* seed) nogil:
    seed[0] ^= <UINT32_t>(seed[0] << 13)
    seed[0] ^= <UINT32_t>(seed[0] >> 17)
    seed[0] ^= <UINT32_t>(seed[0] << 5)

    return seed[0] % (<UINT32_t>RAND_R_MAX + 1)

cdef inline np.ndarray sizet_ptr_to_ndarray(SIZE_t* data, SIZE_t size):
    /* Encapsulate data into a 1D numpy array of intp's.*/
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp> size
    return np.PyArray_SimpleNewFromData(1, shape, np.NPY_INTP, data)


cdef inline double log(double x) nogil:
    return ln(x) / ln(2.0)
#endif
        }
