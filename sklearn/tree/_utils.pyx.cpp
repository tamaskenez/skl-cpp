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

#if 0
// =============================================================================
// PriorityHeap data structure
// =============================================================================

cdef void heapify_up(PriorityHeapRecord* heap, SIZE_t pos) nogil:
    """Restore heap invariant parent.improvement > child.improvement from
       ``pos`` upwards. """
    if pos == 0:
        return

    cdef SIZE_t parent_pos = (pos - 1) / 2

    if heap[parent_pos].improvement < heap[pos].improvement:
        heap[parent_pos], heap[pos] = heap[pos], heap[parent_pos]
        heapify_up(heap, parent_pos)


cdef void heapify_down(PriorityHeapRecord* heap, SIZE_t pos,
                       SIZE_t heap_length) nogil:
    """Restore heap invariant parent.improvement > children.improvement from
       ``pos`` downwards. """
    cdef SIZE_t left_pos = 2 * (pos + 1) - 1
    cdef SIZE_t right_pos = 2 * (pos + 1)
    cdef SIZE_t largest = pos

    if (left_pos < heap_length and
            heap[left_pos].improvement > heap[largest].improvement):
        largest = left_pos

    if (right_pos < heap_length and
            heap[right_pos].improvement > heap[largest].improvement):
        largest = right_pos

    if largest != pos:
        heap[pos], heap[largest] = heap[largest], heap[pos]
        heapify_down(heap, largest, heap_length)


cdef class PriorityHeap:
    """A priority queue implemented as a binary heap.

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
    """

    def __cinit__(self, SIZE_t capacity):
        self.capacity = capacity
        self.heap_ptr = 0
        self.heap_ = <PriorityHeapRecord*> malloc(capacity * sizeof(PriorityHeapRecord))
        if self.heap_ == NULL:
            raise MemoryError()

    def __dealloc__(self):
        free(self.heap_)

    cdef bint is_empty(self) nogil:
        return self.heap_ptr <= 0

    cdef int push(self, SIZE_t node_id, SIZE_t start, SIZE_t end, SIZE_t pos,
                  SIZE_t depth, bint is_leaf, double improvement,
                  double impurity, double impurity_left,
                  double impurity_right) nogil:
        """Push record on the priority heap.

        Returns 0 if successful; -1 on out of memory error.
        """
        cdef SIZE_t heap_ptr = self.heap_ptr
        cdef PriorityHeapRecord* heap = NULL

        // Resize if capacity not sufficient
        if heap_ptr >= self.capacity:
            self.capacity *= 2
            heap = <PriorityHeapRecord*> realloc(self.heap_,
                                                 self.capacity *
                                                 sizeof(PriorityHeapRecord))
            if heap == NULL:
                // no free; __dealloc__ handles that
                return -1
            self.heap_ = heap

        // Put element as last element of heap
        heap = self.heap_
        heap[heap_ptr].node_id = node_id
        heap[heap_ptr].start = start
        heap[heap_ptr].end = end
        heap[heap_ptr].pos = pos
        heap[heap_ptr].depth = depth
        heap[heap_ptr].is_leaf = is_leaf
        heap[heap_ptr].impurity = impurity
        heap[heap_ptr].impurity_left = impurity_left
        heap[heap_ptr].impurity_right = impurity_right
        heap[heap_ptr].improvement = improvement

        // Heapify up
        heapify_up(heap, heap_ptr)

        // Increase element count
        self.heap_ptr = heap_ptr + 1
        return 0

    cdef int pop(self, PriorityHeapRecord* res) nogil:
        """Remove max element from the heap. """
        cdef SIZE_t heap_ptr = self.heap_ptr
        cdef PriorityHeapRecord* heap = self.heap_

        if heap_ptr <= 0:
            return -1

        // Take first element
        res[0] = heap[0]

        // Put last element to the front
        heap[0], heap[heap_ptr - 1] = heap[heap_ptr - 1], heap[0]

        // Restore heap invariant
        if heap_ptr > 1:
            heapify_down(heap, 0, heap_ptr - 1)

        self.heap_ptr = heap_ptr - 1

        return 0
#endif
} //namespace sklcpp
