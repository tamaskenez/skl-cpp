#ifndef _TREE_INCLUDED_23648023948234
#define _TREE_INCLUDED_23648023948234

#include <cstdint>
#include <random>

#include "sx/array_view.h"
#include "nclasses.h"
#include "sx/multi_array.h"

namespace sklearn {
// Authors: Gilles Louppe <g.louppe@gmail.com>
//          Peter Prettenhofer <peter.prettenhofer@gmail.com>
//          Brian Holt <bdholt1@gmail.com>
//          Joel Nothman <joel.nothman@gmail.com>
//          Arnaud Joly <arnaud.v.joly@gmail.com>
//
// Licence: BSD 3 clause

// See _tree.pyx for details.

//import numpy as np
//cimport numpy as np
//
typedef float    DTYPE_t;       // Type of X
typedef double   DOUBLE_t;      // Type of y, sample_weight
typedef intptr_t SIZE_t;        // Type for indices and counters
typedef int32_t  INT32_t;       // Signed 32 bit integer
typedef uint32_t UINT32_t;      // Unsigned 32 bit integer

typedef float DTYPE;
typedef double DOUBLE;

// =============================================================================
// Criterion
// =============================================================================

class Criterion {
	/* Interface for impurity criteria.

	This object stores methods on how to calculate how good a split is using
	different metrics.
	*/

	// The criterion computes the impurity of a node and the reduction of
	// impurity of a split on that node. It also computes the output statistics
	// such as the mean in regression and class probabilities in classification.

protected:
	// Internal structures

	sx::array_view<const DOUBLE_t, 2> y;  // Values of y, y[{sample_idx, output_idx}]
	sx::array_view<const DOUBLE_t> sample_weight; // Sample weights

	sx::array_view<const SIZE_t> samples;         // Sample indices in X, y
	SIZE_t start;                           // samples[start:pos] are the samples in the left node
	SIZE_t pos;                             // samples[pos:end] are the samples in the right node
	SIZE_t end;

	SIZE_t n_outputs;                       // Number of outputs
	SIZE_t n_node_samples;                  // Number of samples in the node (end-start)
	double weighted_n_samples;              // Weighted number of samples (in total)
	double weighted_n_node_samples;         // Weighted number of samples in the node
	double weighted_n_left;                 // Weighted number of samples in the left node
	double weighted_n_right;                // Weighted number of samples in the right node

	// The criterion object is maintained such that left and right collected
	// statistics correspond to samples[start:pos] and samples[pos:end].

    // Methods
    Criterion(SIZE_t n_outputs);

public:

    double get_weighted_n_node_samples() const { return weighted_n_node_samples; }
    double get_weighted_n_left() const { return weighted_n_left; }
    double get_weighted_n_right() const { return weighted_n_right; }

    virtual void init(sx::array_view<const DOUBLE_t, 2> y, sx::array_view<const DOUBLE_t> sample_weight,
		double weighted_n_samples, sx::array_view<const SIZE_t> samples, SIZE_t start,
		SIZE_t end) = 0;
    /* Placeholder for a method which will initialize the criterion.

    Parameters
    ----------
    y: array-like, dtype=DOUBLE_t
        y is a buffer that can store values for n_outputs target variables
        index the ith sample and the kth output value as follows: y[i, k]
    sample_weight: array-like, dtype=DOUBLE_t
        The weight of each sample
    weighted_n_samples: DOUBLE_t
        The total weight of the samples being considered
    samples: array-like, dtype=DOUBLE_t
        Indices of the samples in X and y, where samples[start:end]
        correspond to the samples in this node
    start: SIZE_t
        The first sample to be used on this node
    end: SIZE_t
        The last sample used on this node
	*/

    virtual void reset() = 0;
	// Reset the criterion at pos=start.

    virtual void update(SIZE_t new_pos) = 0;
    /* Updated statistics by moving samples[pos:new_pos] to the left child.

    This updates the collected statistics by moving samples[pos:new_pos]
    from the right child to the left child. It must be implemented by
    the subclass.

    Parameters
    ----------
    new_pos: SIZE_t
        New starting index position of the samples in the right child
    */
    virtual double node_impurity() const = 0;
	/* Placeholder for calculating the impurity of the node.

	Placeholder for a method which will evaluate the impurity of
	the current node, i.e. the impurity of samples[start:end]. This is the
	primary function of the criterion class.
	*/

    virtual void children_impurity(double* impurity_left, double* impurity_right) const = 0;
    /* Placeholder for calculating the impurity of children.

    Placeholder for a method which evaluates the impurity in
    children nodes, i.e. the impurity of samples[start:pos] + the impurity
    of samples[pos:end].

    Parameters
    ----------
    impurity_left: double pointer
        The memory address where the impurity of the left child should be
        stored.
    impurity_right: double pointer
        The memory address where the impurity of the right child should be
        stored
    */

    virtual void node_value(sx::array_view<double> dest) const = 0;
	/* Placeholder for storing the node value.

	Placeholder for a method which will compute the node value
	of samples[start:end] and save the value into dest.

	Parameters
	----------
	dest: array of size n_outputs
	The memory address where the node value should be stored.
	*/

	virtual double impurity_improvement(double impurity) const;
	/* Placeholder for improvement in impurity after a split.

	Placeholder for a method which computes the improvement
	in impurity when a split occurs. The weighted impurity improvement
	equation is the following:

	N_t / N * (impurity - N_t_R / N_t * right_impurity
	- N_t_L / N_t * left_impurity)

	where N is the total number of samples, N_t is the number of samples
	at the current node, N_t_L is the number of samples in the left child,
	and N_t_R is the number of samples in the right child,

	Parameters
	----------
	impurity: double
	The initial impurity of the node before the split

	Return
	------
	double: improvement in impurity after the split occurs
	*/
};


// =============================================================================
// Criterion
// =============================================================================

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
    ClassificationCriterion(NClassesRef n_classes);
        /* Initialize attributes for this criterion.

        Parameters
        ----------
        n_outputs: SIZE_t
            The number of targets, the dimensionality of the prediction
        n_classes: numpy.ndarray, dtype=SIZE_t
            The number of unique classes in each target
        */
public:
    virtual void init(sx::array_view<const DOUBLE_t, 2> y, sx::array_view<const DOUBLE_t> sample_weight,
        double weighted_n_samples, sx::array_view<const SIZE_t> samples, SIZE_t start,
        SIZE_t end) override;
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

    virtual void reset() override;
    /* Reset the criterion at pos=start.*/

    virtual void update(SIZE_t new_pos) override;
        /* Updated statistics by moving samples[pos:new_pos] to the left child.

        Parameters
        ----------
        new_pos: SIZE_t
            The new ending position for which to move samples from the right
            child to the left child.
        */

    virtual void node_value(sx::array_view<double> dest) const override;
        /* Compute the node value of samples[start:end] and save it into dest.

        Parameters
        ----------
        dest: double pointer
            The memory address which we will save the node value into.
        */
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

    virtual double node_impurity() const override;
        /* Evaluate the impurity of the current node, i.e. the impurity of
        samples[start:end], using the cross-entropy criterion.*/

    virtual void children_impurity(double* impurity_left,
                                double* impurity_right) const override;
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

    virtual double node_impurity() const override;
        /* Evaluate the impurity of the current node, i.e. the impurity of
        samples[start:end] using the Gini criterion.*/

    virtual void children_impurity(double* impurity_left,
                                   double* impurity_right) const override;
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

    inline accumulators_t& accu(SIZE_t idx);
    inline const accumulators_t& accu(SIZE_t idx) const;

    RegressionCriterion(SIZE_t n_outputs);
        /* Initialize parameters for this criterion.

        Parameters
        ----------
        n_outputs: SIZE_t
            The number of targets to be predicted
        */

        // Allocate memory for the accumulators

    virtual void init(
        sx::array_view<const DOUBLE_t, 2> y,
        sx::array_view<const DOUBLE_t> sample_weight,
        double weighted_n_samples,
        sx::array_view<const SIZE_t> samples, SIZE_t start,
        SIZE_t end) override;
        /* Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end].*/

    virtual void reset() override;
        /* Reset the criterion at pos=start.*/

    virtual void update(SIZE_t new_pos) override;
        /* Updated statistics by moving samples[pos:new_pos] to the left.*/

    virtual void node_value(sx::array_view<double> dest) const override;
        /* Compute the node value of samples[start:end] into dest.*/
};

class MSE
: public RegressionCriterion {
    /* Mean squared error impurity criterion.

        MSE = var_left + var_right
    */
    virtual double node_impurity() const override;
        /* Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end].*/

    virtual void children_impurity(double* impurity_left,
                                double* impurity_right) const override;
        /* Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end]).*/
};

class FriedmanMSE
: public MSE {
    /* Mean squared error impurity criterion with improvement score by Friedman

    Uses the formula (35) in Friedmans original Gradient Boosting paper:

        diff = mean_left - mean_right
        improvement = n_left * n_right * diff^2 / (n_left + n_right)
    */

    virtual double impurity_improvement(double impurity) const override;
};


// =============================================================================
// Splitter
// =============================================================================

struct SplitRecord {
    // Data to track sample split
    SIZE_t feature;        // Which feature to split on.
    SIZE_t pos;            // Split samples array at the given position,
                           // i.e. count of samples below threshold for feature.
                           // pos is >= end if the node is a leaf.
    double threshold;      // Threshold to split at.
    double improvement;    // Impurity improvement given parent node.
    double impurity_left;  // Impurity of the left split.
    double impurity_right; // Impurity of the right split.
};


class Splitter {
    /* Abstract splitter class.

    Splitters are called by tree builders to find the best splits on both
    sparse and dense data, one split at a time.
    */

    // The splitter searches in the input space for a feature and a threshold
    // to split the samples samples[start:end].
    //
    // The impurity computations are delegated to a criterion object.

    // Internal structures
protected:
    Criterion* criterion;      // Impurity criterion
    SIZE_t max_features;       // Number of features to test
    SIZE_t min_samples_leaf;   // Min samples in a leaf
    double min_weight_leaf;    // Minimum weight in a leaf
    std::default_random_engine* random_state;
    std::vector<SIZE_t> samples;              // Sample indices in X, y
private:
    SIZE_t n_samples;                         // X.shape[0]
public:
    SIZE_t get_n_samples() const { return n_samples; }
private:
    double weighted_n_samples;                // Weighted number of samples
protected:
    std::vector<SIZE_t> features;             // Feature indices in X
    std::vector<SIZE_t> constant_features;    // Constant features indices
    SIZE_t n_features;                        // X.shape[1]
    std::vector<DTYPE_t> feature_values;      // temp. array holding feature values
    SIZE_t start;                             // Start position for the current node
    SIZE_t end;                               // End position for the current node

private:
    sx::array_view<const DOUBLE_t, 2> y;
    sx::array_view<const DOUBLE_t> sample_weight;

    // The samples vector `samples` is maintained by the Splitter object such
    // that the samples contained in a node are contiguous. With this setting,
    // `node_split` reorganizes the node samples `samples[start:end]` in two
    // subsets `samples[start:pos]` and `samples[pos:end]`.

    // The 1-d  `features` array of size n_features contains the features
    // indices and allows fast sampling without replacement of features.

    // The 1-d `constant_features` array of size n_features holds in
    // `constant_features[:n_constant_features]` the feature ids with
    // constant values for all the samples that reached a specific node.
    // The value `n_constant_features` is given by the the parent node to its
    // child nodes.  The content of the range `[n_constant_features:]` is left
    // undefined, but preallocated for performance reasons
    // This allows optimization with depth-based tree building.

    // Methods
protected:
    Splitter(Criterion* criterion, SIZE_t max_features,
                      SIZE_t min_samples_leaf, double min_weight_leaf,
                      std::default_random_engine* random_state);
    /*
    Parameters
    ----------
    criterion: Criterion
      The criterion to measure the quality of a split.

    max_features: SIZE_t
      The maximal number of randomly selected features which can be
      considered for a split.

    min_samples_leaf: SIZE_t
      The minimal number of samples each leaf can have, where splits
      which would result in having less samples in a leaf are not
      considered.

    min_weight_leaf: double
      The minimal weight each leaf can have, where the weight is the sum
      of the weights of each sample in it.

    random_state: object
      The user inputted random state to be used for pseudo-randomness
    */

public:
    virtual void init(sx::array_view<const DTYPE_t, 2> X,
              sx::array_view<const DOUBLE_t, 2> y,
              sx::array_view<const DOUBLE_t> sample_weight);
    /* Initialize the splitter.

    Take in the input data X, the target Y, and optional sample weights.

    Parameters
    ----------
    X: object
       This contains the inputs. Usually it is a 2d numpy array.

    y: numpy.ndarray, dtype=DOUBLE_t
       This is the vector of targets, or true labels, for the samples

    sample_weight: numpy.ndarray, dtype=DOUBLE_t (optional)
       The weights of the samples, where higher weighted samples are fit
       closer than lower weight samples. If not provided, all samples
       are assumed to have uniform weight.
    */

    void node_reset(SIZE_t start, SIZE_t end,
                    double* weighted_n_node_samples);
     /* Reset splitter on node samples[start:end].

     Parameters
     ----------
     start: SIZE_t
         The index of the first sample to consider
     end: SIZE_t
         The index of the last sample to consider
     weighted_n_node_samples: numpy.ndarray, dtype=double pointer
         The total weight of those samples
     */

    virtual void node_split(double impurity,   // Impurity of the node
                         SplitRecord* split,
                         SIZE_t* n_constant_features) = 0;
    /* Find the best split on node samples[start:end]. */

    void node_value(sx::array_view<double> dest) const;
    /* Copy the value of node samples[start:end] into dest.*/

    double node_impurity() const;
};

class BaseDenseSplitter
: public Splitter {
protected:
    sx::array_view<const DTYPE_t, 2> X;
public:
    BaseDenseSplitter(Criterion* criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  std::default_random_engine* random_state);

    virtual void init(sx::array_view<const DTYPE_t, 2> X,
              sx::array_view<const DOUBLE_t, 2> y,
              sx::array_view<const DOUBLE_t> sample_weight) override;
        /* Initialize the splitter.*/
};

class BestSplitter
: public BaseDenseSplitter {
    /* Splitter for finding the best split.*/
    virtual void node_split(double impurity, SplitRecord* split,
                         SIZE_t* n_constant_features) override;
        /* Find the best split on node samples[start:end].*/
};


class RandomSplitter
: public BaseDenseSplitter {
    /* Splitter for finding the best random split.*/
    virtual void node_split(double impurity, SplitRecord* split,
                         SIZE_t* n_constant_features) override;
        /* Find the best random split on node samples[start:end].*/
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
                  std::default_random_engine* random_state);

    virtual void init(sx::array_view<const DTYPE_t, 2> X,
              sx::array_view<const DOUBLE_t, 2> y,
              sx::array_view<const DOUBLE_t> sample_weight) override;

    virtual void node_split(double impurity, SplitRecord* split,
                         SIZE_t* n_constant_features) override;
        /* Find the best split on node samples[start:end].*/
};


// =============================================================================
// Tree
// =============================================================================

struct Node {
    // Base storage structure for the nodes in a Tree object

    SIZE_t left_child;                   // id of the left child of the node
    SIZE_t right_child;                  // id of the right child of the node
    SIZE_t feature;                      // Feature used for splitting the node
    DOUBLE_t threshold;                  // Threshold value at the node
    DOUBLE_t impurity;                   // Impurity of the node (i.e., the value of the criterion)
    SIZE_t n_node_samples;               // Number of samples at the node
    DOUBLE_t weighted_n_node_samples;    // Weighted number of samples at the node
};

class Tree {
	/* Array-based representation of a binary decision tree.

    The binary tree is represented as a number of parallel arrays. The i-th
    element of each array holds information about the node `i`. Node 0 is the
    tree's root. You can find a detailed description of all arrays in
    `_tree.pxd`. NOTE: Some of the arrays only apply to either leaves or split
    nodes, resp. In this case the values of nodes of the other type are
    arbitrary!

    Attributes
    ----------
    node_count : int
        The number of nodes (internal nodes + leaves) in the tree.

    capacity : int
        The current capacity (i.e., size) of the arrays, which is at least as
        great as `node_count`.

    max_depth : int
        The maximal depth of the tree.

    children_left : array of int, shape [node_count]
        children_left[i] holds the node id of the left child of node i.
        For leaves, children_left[i] == TREE_LEAF. Otherwise,
        children_left[i] > i. This child handles the case where
        X[:, feature[i]] <= threshold[i].

    children_right : array of int, shape [node_count]
        children_right[i] holds the node id of the right child of node i.
        For leaves, children_right[i] == TREE_LEAF. Otherwise,
        children_right[i] > i. This child handles the case where
        X[:, feature[i]] > threshold[i].

    feature : array of int, shape [node_count]
        feature[i] holds the feature to split on, for the internal node i.

    threshold : array of double, shape [node_count]
        threshold[i] holds the threshold for the internal node i.

    value : array of double, shape [node_count, n_outputs, max_n_classes]
        Contains the constant prediction value of each node.

    impurity : array of double, shape [node_count]
        impurity[i] holds the impurity (i.e., the value of the splitting
        criterion) at node i.

    n_node_samples : array of int, shape [node_count]
        n_node_samples[i] holds the number of training samples reaching node i.

    weighted_n_node_samples : array of int, shape [node_count]
        weighted_n_node_samples[i] holds the weighted number of training samples
        reaching node i.
    */

    // The Tree object is a binary tree structure constructed by the
    // TreeBuilder. The tree structure is used for predictions and
    // feature importances.
public:
    // Input/Output layout
    SIZE_t n_features;           // Number of features in X
    NClassesRef n_classes;       // Number of classes in y[:, k]

    // Inner structures: values are stored separately from node structure,
    // since size is determined at runtime.
    SIZE_t max_depth;            // Max depth of the tree
    std::vector<Node> nodes;     // Array of nodes
	sx::matrix<double> value;    // [{node, c}] where c is to be calculated from NClassesRef

    // Methods
	Tree(int n_features, NClassesRef n_classes);
	void reserve(SIZE_t capacity);
    SIZE_t _add_node(SIZE_t parent, bool is_left, bool is_leaf,
                          SIZE_t feature, double threshold, double impurity,
                          SIZE_t n_node_samples,
                          double weighted_n_samples);
	/* Add a node to the tree.

	The new node registers itself as the child of its parent.

	Returns (size_t)(-1) on error.
	*/

    //np.ndarray _get_node_ndarray()

    sx::matrix<double> predict(sx::array_view<const DTYPE_t, 2> X) const;
	/* Predict target for X.*/

    std::vector<SIZE_t> apply(sx::array_view<const DTYPE_t, 2> X) const;
	/* Finds the terminal region (=leaf node) for each sample in X.*/

	std::vector<SIZE_t> _apply_dense(sx::array_view<const DTYPE_t, 2> X) const;
	/* Finds the terminal region (=leaf node) for each sample in X.*/

	SIZE_t apply_single(sx::array_view<const DTYPE_t> x) const;
	SIZE_t _apply_dense_single(sx::array_view<const DTYPE_t> x) const;

    //np.ndarray _apply_sparse_csr(object X)

    std::vector<double> compute_feature_importances(bool normalize = true);
	/* Computes the importance of each feature (aka variable).*/
};

// =============================================================================
// Tree builder
// =============================================================================

class TreeBuilder {
	/* Interface for different tree building strategies.*/

    // The TreeBuilder recursively builds a Tree object from training samples,
    // using a Splitter object for splitting internal nodes and assigning
    // values to leaves.
    //
    // This class controls the various stopping criteria and the node splitting
    // evaluation order, e.g. depth-first or best-first.
protected:
    Splitter* splitter;          // Splitting algorithm

    SIZE_t min_samples_split;   // Minimum number of samples in an internal node
    SIZE_t min_samples_leaf;    // Minimum number of samples in a leaf
    double min_weight_leaf;     // Minimum weight in a leaf
    SIZE_t max_depth;           // Maximal tree depth

protected:
	TreeBuilder(Splitter* splitter, SIZE_t min_samples_split,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  SIZE_t max_depth);
public:
	void build(Tree& tree, sx::matrix_view<const DTYPE_t> X, sx::matrix_view<const DOUBLE> y);
    virtual void build(Tree& tree, sx::matrix_view<const DTYPE_t> X,
							sx::matrix_view<const DOUBLE> y,
                             sx::array_view<const DOUBLE> sample_weight) = 0;
	/* Build a decision tree from the training set (X, y).*/
protected:
    void _check_input(sx::matrix_view<const DTYPE_t> X,
						sx::matrix_view<const DOUBLE> y,
                             sx::array_view<const DOUBLE> sample_weight);
};

// Depth first builder ---------------------------------------------------------
class DepthFirstTreeBuilder : public TreeBuilder {
/* Build a decision tree in depth-first fashion.*/
public:
DepthFirstTreeBuilder(Splitter* splitter, SIZE_t min_samples_split,
			  SIZE_t min_samples_leaf, double min_weight_leaf,
			  SIZE_t max_depth);

virtual void build(Tree& tree, sx::matrix_view<const DTYPE_t> X,
	sx::matrix_view<const DOUBLE> y,
						 sx::array_view<const DOUBLE> sample_weight) override;
	/* Build a decision tree from the training set (X, y).*/
};
// Best first builder ----------------------------------------------------------

    struct PriorityHeapRecord;

class BestFirstTreeBuilder
: public TreeBuilder {
/* Build a decision tree in best-first fashion.

The best node to expand is given by the node at the frontier that has the
highest impurity improvement.

NOTE: this TreeBuilder will ignore ``tree.max_depth`` .
*/
SIZE_t max_leaf_nodes;
public:
BestFirstTreeBuilder(Splitter* splitter, SIZE_t min_samples_split,
			  SIZE_t min_samples_leaf, double min_weight_leaf,
			  SIZE_t max_depth, SIZE_t max_leaf_nodes);

virtual void build(Tree& tree, sx::matrix_view<const DTYPE_t> X,
				   sx::matrix_view<const DOUBLE> y,
				   sx::array_view<const DOUBLE> sample_weight) override;
	/* Build a decision tree from the training set (X, y).*/
private:
void _add_split_node(Splitter* splitter, Tree& tree,
								SIZE_t start, SIZE_t end, double impurity,
								bool is_first, bool is_left, Node* parent,
								SIZE_t depth,
								PriorityHeapRecord* res);
	/* Adds node w/ partition ``[start, end)`` to the frontier. */
};

}
#endif //incl guard
