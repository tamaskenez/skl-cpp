#ifndef UTILS_PXD_INCLUDED_923469283
#define UTILS_PXD_INCLUDED_923469283

// Authors: Gilles Louppe <g.louppe@gmail.com>
//          Peter Prettenhofer <peter.prettenhofer@gmail.com>
//          Arnaud Joly <arnaud.v.joly@gmail.com>
//
// Licence: BSD 3 clause

// See _utils.pyx for details.

#include <cstdint>
#include <vector>

namespace sklcpp {

typedef intptr_t SIZE_t;              // Type for indices and counters


// =============================================================================
// Stack data structure
// =============================================================================

// A record on the stack for depth-first tree growing
struct StackRecord {
    SIZE_t start;
    SIZE_t end;
    SIZE_t depth;
    SIZE_t parent;
    bool is_left;
    double impurity;
    SIZE_t n_constant_features;
};

class Stack {
    /* A LIFO data structure.

    Attributes
    ----------
    capacity : SIZE_t
        The elements the stack can hold; if more added then ``self.stack_``
        needs to be resized.

    top : SIZE_t
        The number of elements currently on the stack.

    stack : StackRecord pointer
        The stack of records (upward in the stack corresponds to the right).
    */

    std::vector<StackRecord> stack_;
public:
    Stack(SIZE_t initial_capacity);
    bool is_empty() const;

    void push(SIZE_t start, SIZE_t end, SIZE_t depth, SIZE_t parent,
                  bool is_left, double impurity,
                  SIZE_t n_constant_features);
    /* Push a new element onto the stack.
    */

    int pop(StackRecord* res);
    /* Remove the top element from the stack and copy to ``res``.

    Returns 0 if pop was successful (and ``res`` is set); -1
    otherwise.
    */
};

#if 0
// =============================================================================
// PriorityHeap data structure
// =============================================================================

// A record on the frontier for best-first tree growing
cdef struct PriorityHeapRecord:
    SIZE_t node_id
    SIZE_t start
    SIZE_t end
    SIZE_t pos
    SIZE_t depth
    bint is_leaf
    double impurity
    double impurity_left
    double impurity_right
    double improvement

cdef class PriorityHeap:
    cdef SIZE_t capacity
    cdef SIZE_t heap_ptr
    cdef PriorityHeapRecord* heap_

    cdef bint is_empty(self) nogil
    cdef int push(self, SIZE_t node_id, SIZE_t start, SIZE_t end, SIZE_t pos,
                  SIZE_t depth, bint is_leaf, double improvement,
                  double impurity, double impurity_left,
                  double impurity_right) nogil
    cdef int pop(self, PriorityHeapRecord* res) nogil

#endif
} // namespace sklcpp
#endif // incl guard
