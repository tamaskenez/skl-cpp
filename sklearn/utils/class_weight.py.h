#ifndef CLASS_WEIGHT_INCLUDED_1643983423
#define CLASS_WEIGHT_INCLUDED_1643983423

#include "range/view/take_at.hpp"
#include "sx/utility.h"

#include "sklearn/pythonemu/class_weight_union.h"
#include "sklearn/preprocessing/label.py.h"
#include "sklearn/pythonemu/errors.h"

namespace sklearn {

// Authors: Andreas Mueller
//          Manoj Kumar
// License: BSD 3 clause

//import warnings
//import numpy as np
//from ..externals import six
//from ..utils.fixes import in1d
//
//from .fixes import bincount

    using sx::array_view;
    using sx::matrix_view;

template<typename OutputValue>
std::vector<double> compute_class_weight(
    const class_weight_union<OutputValue>& class_weight, int output_idx,
    array_view<const OutputValue> classes,
    array_view<const OutputValue> y) {
    /*Estimate class weights for unbalanced datasets.

    Parameters
    ----------
    class_weight : dict, 'balanced' or None
        If 'balanced', class weights will be given by
        ``n_samples / (n_classes * np.bincount(y))``.
        If a dictionary is given, keys are classes and values
        are corresponding class weights.
        If None is given, the class weights will be uniform.

    classes : ndarray
        Array of the classes occurring in the data, as given by
        ``np.unique(y_org)`` with ``y_org`` the original class labels.

    y : array-like, shape (n_samples,)
        Array of original class labels per sample;

    Returns
    -------
    class_weight_vect : ndarray, shape (n_classes,)
        Array with class_weight_vect[i] the weight for i-th class

    References
    ----------
    The "balanced" heuristic is inspired by
    Logistic Regression in Rare Events Data, King, Zen, 2001.
    */
    // Import error caused by circular imports.

    namespace view = ranges::view;

    std::vector<double> weight;
    if(class_weight.is_none()) {
        // uniform class weights
        weight.assign(classes.size(), 1.0);
    } else if(class_weight.is_string("auto") || class_weight.is_string("balanced")) {
        // Find the weight of each class as present in y.
        LabelEncoder<OutputValue, int> le;
        auto y_ind = le.fit_transform(y);
        for(auto&c: classes) {
            if(!std::binary_search(BEGINEND(le.classes_), c))
                throw ValueError("classes should have valid labels that are in y");
        }

        // inversely proportional to the number of samples in the class
        if(class_weight.is_string("auto")) {
            auto recip_freq = 1.0 / y_ind;
            weight = view::take_at(recip_freq, le.transform(classes)) / sx::mean(recip_freq);
        } else {
            auto recip_freq = length(y) / ((double)length(le.classes_) *
                                   bincount(y_ind));
            auto weight = view::take_at(recip_freq, le.transform(classes));
        }
    } else {
        // user-defined dictionary
        std::vector<double> weight(classes.size(), 1.0);
        if(!class_weight.is_dict()) {
            throw ValueError(stringf("class_weight must be dict, 'auto', or None,"
                             " got: %s", class_weight.string));
        }
        for(auto c: class_weight.dicts[output_idx]) {
            auto i = std::lower_bound(BEGINEND(classes), c.first);
            if(i == classes.end() || *i != c.first)
                throw ValueError(stringf("Class label %f not present.", c.first));
            else
                weight[i-classes.begin()] = c.second;
        }
    }
    return weight;
}

template<typename OutputValue, typename Index = std::size_t>
std::vector<double> compute_sample_weight(
    const class_weight_union<OutputValue>& class_weight,
    matrix_view<const OutputValue> y,
    array_view<const Index> indices = array_view<const Index>()) {
    /*Estimate sample weights by class for unbalanced datasets.

    Parameters
    ----------
    class_weight : dict, list of dicts, "balanced", or None, optional
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data:
        ``n_samples / (n_classes * np.bincount(y))``.

        For multi-output, the weights of each column of y will be multiplied.

    y : array-like, shape = [n_samples] or [n_samples, n_outputs]
        Array of original class labels per sample.

    indices : array-like, shape (n_subsample,), or None
        Array of indices to be used in a subsample. Can be of length less than
        n_samples in the case of a subsample, or equal to n_samples in the
        case of a bootstrap subsample with repeated indices. If None, the
        sample weight will be calculated over the full sample. Only "auto" is
        supported for class_weight if this is provided.

    Returns
    -------
    sample_weight_vect : ndarray, shape (n_samples,)
        Array with sample weights as applied to the original y
    */

    auto n_outputs = y.extents(1);

    if(class_weight.is_string()) {
        if(!class_weight.is_string("balanced")
            && !class_weight.is_string("auto"))
        {
            throw ValueError(sx::stringf("The only valid preset for class_weight is "
                             "\"balanced\". Given \"%s\".", class_weight.string.c_str()));
        }
    } else if(!indices.empty() &&
          !class_weight.is_string()) {
        throw ValueError(sx::stringf("The only valid class_weight for subsampling is "
                         "\"balanced\". Given \"%s\".", class_weight.string.c_str()));
    } else if(n_outputs > 1) {
        if (!class_weight.is_dicts()) {
            throw ValueError("For multi-output, class_weight should be a "
                             "list of dicts, or a valid string.");
        }
        if(class_weight.dicts.size() != n_outputs) {
            throw ValueError("For multi-output, number of elements in "
                             "class_weight should match number of outputs.");
        }
    }

    namespace view = ranges::view;
    using sx::range;

    std::vector<double> expanded_class_weight(y.extents(0), 1.0);
    for(auto k: range(n_outputs)) {

        auto y_full = y(sx::all, k);
        auto classes_full = sx::make_vector(y_full);
        sx::sort_unique_inplace(classes_full);
        std::vector<OutputValue> classes_missing;

        std::vector<double> weight_k;

        if(!indices.empty()) {
            // Get class weights for the subsample, covering all classes in
            // case some labels that were present in the original data are
            // missing from the sample.
            auto y_subsample = view::take_at(y(sx::all, k), indices);
            auto classes_subsample = sx::sort_unique_inplace(make_vector(y_subsample));

            auto indices_of_classes_full_in_classes_subsample = sx::searchsorted<Index>(
                classes_subsample,
                classes_full);

            auto computed_class_weight =
                compute_class_weight(class_weight, k,
                                     classes_subsample,
                                     y_subsample);

            weight_k = sx::make_vector(
                view::take_at(
                    computed_class_weight,
                    indices_of_classes_full_in_classes_subsample
                )
            );

            auto classes_missing = set_difference(
                                                  sx::make_vector(classes_full), classes_subsample);
        } else {
            weight_k = compute_class_weight(class_weight, k,
                                            classes_full,
                                            y_full);
        }

        auto indices_of_y_full_in_classes_full =
            sx::searchsorted<Index>(classes_full, y_full);

        weight_k = sx::make_vector(
            view::take_at(weight_k, indices_of_y_full_in_classes_full)
        );

        if(!classes_missing.empty()) {
            assert(y_full.size() == weight_k.size());
            assert(std::is_sorted(BEGINEND(classes_missing)));
            // Make missing classes' weight zero
            for(auto idx: range(y_full.size())) {
                if(std::binary_search(BEGINEND(classes_missing), y_full[idx]))
                    weight_k[idx] = 0;
            }
        }
        assert(y.extents[0] == expanded_class_weight.size());
        assert(y.extents[0] == weight_k.size());
        for(auto idx: range(y.extents[0]))
            expanded_class_weight[idx] *= weight_k[idx];
    } //for all outputs

    return expanded_class_weight;
}

} //namespace sklearn
#endif
