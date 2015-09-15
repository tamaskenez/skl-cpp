#ifndef NCLASSES_INCLUDED_24703974234
#define NCLASSES_INCLUDED_24703974234

#include <vector>
#include "sx/range.h"

namespace sklearn {
	// Defines the NClasses and NClassesRef classes
	// NClasses the main class which owns the data, NClassesRef
	// is a weak pointer and also reader interface to NClasses

	// The original python code had the n_classes 1-D array to store
	// the number of classes for each output.
	// It stored a single output value as a 2-D array of size
	// n_outputs x max(n_classes)
	// Of course this 2-D array can be more optimally stored in vector
	// of length sum(n_classes). NClasses and NClassesRef help in this task.

	// NClasses contains the n_classes array which describes the number of
	// classes for each output.
	// Additionally it contains offsets = cumsum(n_classes) to help to store
	// output values in a contiguous storage.
	// It also contains an optimization where all n_classes values are 1
	class NClassesRef {
	protected:
	  struct Item {
          Item(int n_classes, int offset) : n_classes(n_classes), offset(offset) {}
	    int n_classes;
	    int offset;
	  };
	  int n_outputs_;
	  const Item* n_classes_ptr;
	public:
	  NClassesRef()
	  : n_outputs_(0)
	  , n_classes_ptr(nullptr)
	  {}

	  int n_outputs() const { return n_outputs_; }
	  inline int count(int output_idx) const { return n_classes_ptr ? n_classes_ptr[output_idx].n_classes : 1; }
	  inline int offset(int output_idx) const { return n_classes_ptr ? n_classes_ptr[output_idx].offset : output_idx; }
	  int n_elements() const { return n_classes_ptr ? n_classes_ptr[n_outputs()].offset : n_outputs(); }
	  inline int offset(int output_idx, int class_idx) const {
	    assert(0 <= output_idx && output_idx < n_outputs());
	    assert(0 <= class_idx && class_idx < count(output_idx));
	    return offset(output_idx) + class_idx;
	  }
	};

	class NClasses
	: public NClassesRef
	{
	  std::vector<Item> v;
	public:
	  NClasses()
	  : v(1, Item{0, 0})
	  {}

	  void push_back(int n_classes) {
	      v.back().n_classes = n_classes;
	      v.emplace_back(0, v.back().offset + n_classes);
	      if(n_classes_ptr || n_classes != 1)
	        n_classes_ptr = v.data();
	      n_outputs_ = v.size() - 1;
	  }
        void clear() {
            n_outputs_ = 0;
            n_classes_ptr = nullptr;
            v.assign(1, Item{0, 0});
        }
        void assign(int n_outputs, int n_classes) {
            clear();
            for(int i: sx::range(n_outputs)) {
                (void)i;
                push_back(n_classes);
            }
        }
	};
}

#endif
