#ifndef CLASS_WEIGHT_UNION_INCLUDED_273984234
#define CLASS_WEIGHT_UNION_INCLUDED_273984234

namespace sklearn {
	// ad-hoc discriminative union of
	// - list of dicts (flat maps), one for each output
	// - a string
	// - or none of the above
    template<typename OutputValue>
	struct class_weight_union {
        enum kind_t {K_NONE, K_DICTS, K_STRING};

        using dict_value_t = std::pair<OutputValue, double>; //map outputvalue -> weight
	    using dict_t = std::vector<dict_value_t>; //implements flat map (sorted)

        class_weight_union() = delete;
		class_weight_union(const class_weight_union&) = delete;
		class_weight_union(class_weight_union&& x)
        : dicts(std::move(x.dicts))
        , kind(x.kind)
		, string(std::move(x.string))
        {
		}

		class_weight_union(std::nullptr_t)
        : kind(K_NONE) {}
		class_weight_union(const char* s)
		: kind(K_STRING)
		, string(s)
		{}
        class_weight_union(std::vector<dict_t>&& dicts)
		: dicts(std::move(dicts))
        , kind(K_DICTS)
		{}

	    // dicts and b_balanced are mutually exclusive
	    std::vector<dict_t> dicts; //dicts[output_idx]
		const kind_t kind;
		const std::string string;

	    bool is_none() const { return kind == K_NONE; }
		bool is_string() const { return kind == K_STRING; }
		bool is_dicts() const { return kind == K_DICTS; }
	};
}

#endif
