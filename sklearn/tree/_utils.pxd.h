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
#include <queue>

namespace sklearn {

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

// =============================================================================
// PriorityHeap data structure
// =============================================================================

// A record on the frontier for best-first tree growing
struct PriorityHeapRecord {
    PriorityHeapRecord() = default;
    PriorityHeapRecord(
        SIZE_t node_id,
        SIZE_t start,
        SIZE_t end,
        SIZE_t pos,
        SIZE_t depth,
        bool is_leaf,
        double impurity,
        double impurity_left,
        double impurity_right,
        double improvement
    ): node_id(node_id)
    , start(start)
    , end(end)
    , pos(pos)
    , depth(depth)
    , is_leaf(is_leaf)
    , impurity(impurity)
    , impurity_left(impurity_left)
    , impurity_right(impurity_right)
    , improvement(improvement)
    {}

    SIZE_t node_id;
    SIZE_t start;
    SIZE_t end;
    SIZE_t pos;
    SIZE_t depth;
    bool is_leaf;
    double impurity;
    double impurity_left;
    double impurity_right;
    double improvement;

    bool operator<(const PriorityHeapRecord& x) const {
        return improvement < x.improvement;
    }
};

class PriorityHeap {
    /* A priority queue implemented as a binary heap.

    The heap invariant is that the impurity improvement of the parent record
    is larger then the impurity improvement of the children.

    Attributes
    ----------
    capacity : SIZE_t
        The capacity of the heap

    heap_ptr : SIZE_t
        The water mark of the heap; the heap grows from left to right in the
        array ``heap_``. The following invariant holds ``heap_ptr < capacity``.

    heap_ : PriorityHeapRecord*
        The array of heap records. The maximum element is on the left;
        the heap grows from left to right
    */

    std::priority_queue<PriorityHeapRecord> heap_;
public:
    PriorityHeap(SIZE_t initial_capacity);
    bool is_empty() const;
    void push(SIZE_t node_id, SIZE_t start, SIZE_t end, SIZE_t pos,
                  SIZE_t depth, bool is_leaf, double improvement,
                  double impurity, double impurity_left,
                  double impurity_right);
    void push(const PriorityHeapRecord& rec);
    /* Push record on the priority heap.
    */

    int pop(PriorityHeapRecord* res);
};

} // namespace sklearn
#endif // incl guard
