// cython: cdivision=True
// cython: boundscheck=False
// cython: wraparound=False

// Authors: Gilles Louppe <g.louppe@gmail.com>
//          Peter Prettenhofer <peter.prettenhofer@gmail.com>
//          Arnaud Joly <arnaud.v.joly@gmail.com>
//
// Licence: BSD 3 clause

#include "_utils.pxd.h"

namespace sklcpp {
// =============================================================================
// Stack data structure
// =============================================================================

Stack::Stack(SIZE_t initial_capacity) {
    stack_.reserve(initial_capacity);
}


    bool Stack::is_empty() const {
        return stack_.empty();
    }

    void Stack::push(SIZE_t start, SIZE_t end, SIZE_t depth, SIZE_t parent,
                  bool is_left, double impurity,
                  SIZE_t n_constant_features) {
        stack_.emplace_back();
        auto&t=stack_.back();
        t.start = start;
        t.end = end;
        t.depth = depth;
        t.parent = parent;
        t.is_left = is_left;
        t.impurity = impurity;
        t.n_constant_features = n_constant_features;
}

    int Stack::pop(StackRecord* res) {
        if(stack_.empty())
            return -1;

        *res = stack_.back();
        stack_.pop_back();
        return 0;
    }

// =============================================================================
// PriorityHeap data structure
// =============================================================================

    PriorityHeap::PriorityHeap(SIZE_t initial_capacity)
    : heap_(
         std::less<PriorityHeapRecord>(),
            (
             [initial_capacity]()
             {
                 std::vector<PriorityHeapRecord> v;
                 v.reserve(initial_capacity);
                 return v;
             }
             )()
    )
    {}

    bool PriorityHeap::is_empty() const {
        return heap_.empty();
    }

    void PriorityHeap::push(SIZE_t node_id, SIZE_t start, SIZE_t end, SIZE_t pos,
                  SIZE_t depth, bool is_leaf, double improvement,
                  double impurity, double impurity_left,
                  double impurity_right) {

        // Put element as last element of heap
        heap_.emplace(node_id, start, end, pos, depth, is_leaf,
            impurity, impurity_left, impurity_right, improvement);
    }

    void PriorityHeap::push(const PriorityHeapRecord& rec) {
        heap_.emplace(rec);
    }

    int PriorityHeap::pop(PriorityHeapRecord* res) {
        if(heap_.empty())
            return -1;

        // Take first element
        *res = heap_.top();
        heap_.pop();

        return 0;
    }
} //namespace sklcpp
