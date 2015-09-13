#ifndef AUX_TYPES_INCLUDED_270349234
#define AUX_TYPES_INCLUDED_270349234

#include <vector>

namespace sklcpp {

	// ad-hoc discriminative union of
	// - list of dicts (flat maps), one for each output
	// - "balanced"
	// - or none of the above
	struct class_weight_union {
        enum kind_t {K_NONE, K_DICTS, K_BALANCED};

        using dict_value_t = std::pair<int, double>;
	    using dict_t = std::vector<dict_value_t>; //implements flat map (sorted)

        class_weight_union() = delete;
        class_weight_union(kind_t kind)
        : kind(kind) {}
		class_weight_union(const class_weight_union&) = delete;
		class_weight_union(class_weight_union&& x)
        : dicts(std::move(x.dicts))
        , kind(x.kind)
        {
		}

		class_weight_union(const char* s)
		: kind(K_BALANCED)
		{
			if(strcmp(s, "balanced"))
				throw std::runtime_error("class_weight_union: the only acceptable string argument is 'balanced'");
		}
        class_weight_union(std::vector<dict_t>&& dicts)
		: dicts(std::move(dicts))
        , kind(K_DICTS)
		{}

	    // dicts and b_balanced are mutually exclusive
	    std::vector<dict_t> dicts; //dicts[output_idx]
		const kind_t kind;

	    bool is_none() const { return kind == K_NONE; }
		bool is_balanced() const { return kind == K_BALANCED; }
		bool is_dicts() const { return kind == K_DICTS; }
	};

	// ad-hoc discriminative union of
	// - int
	// - float
	// - 'auto', 'sqrt', 'log2'
	// - or none of the above
	// default is 'auto'
	struct max_features_union {
	private:
		enum kind_t { K_NONE, K_INT, K_FLOAT, K_AUTO, K_SQRT, K_LOG2 };
		static kind_t kind_from_string(const char* s) {
			if(strcmp(s, "auto") == 0)      return K_AUTO;
			else if(strcmp(s, "sqrt") == 0) return K_SQRT;
			else if(strcmp(s, "log2") == 0) return K_LOG2;
			else if(strcmp(s, "none") == 0) return K_NONE;
			else
				throw std::runtime_error("max_features_union:: string label must be one of none, auto, sqrt, log2");
		}
	public:
		max_features_union() = delete;
		max_features_union(int i) : i(i), kind(K_INT) {}
		max_features_union(float f) : f(f), kind(K_FLOAT) {}
		max_features_union(const char *s)
		: kind(kind_from_string(s))
		{}

		const int i = INT_MIN;
		const float f = NAN;
		const kind_t kind;

		bool is_int() const { return kind == K_INT; }
		bool is_float() const { return kind == K_FLOAT; }
		bool is_auto() const { return kind == K_AUTO; }
		bool is_sqrt() const { return kind == K_SQRT; }
		bool is_log2() const { return kind == K_LOG2; }
		bool is_none() const { return kind == K_NONE; }
	};

	// discriminative union for Criterion, can be one of:
	// owned Criterion*
	// - gini, entropy, mse, friedman_mse
	struct criterion_union {
        enum kind_t { K_OBJECT, K_GINI, K_ENTROPY, K_MSE, K_FRIEDMAN_MSE };
	private:
		kind_t kind_from_string(const char* s) {
			if(strcmp(s, "gini") == 0) return K_GINI;
			else if(strcmp(s, "entropy") == 0) return K_ENTROPY;
			else if(strcmp(s, "mse") == 0) return K_MSE;
			else if(strcmp(s, "friedman_mse") == 0) return K_FRIEDMAN_MSE;
			else
				throw std::runtime_error("criterion_union: string must be gini, entropy, mse, friedman_mse");
		}
	public:
		criterion_union() = delete;
		criterion_union(const criterion_union&) = delete;
		criterion_union(criterion_union&& x)
		: object(std::move(x.object))
		, kind(x.kind)
		{}

		criterion_union(std::unique_ptr<Criterion>&& object) //takes ownership
		: object(std::move(object))
		, kind(K_OBJECT)
		{}
		criterion_union(const char* s)
		: kind(kind_from_string(s))
		{}
		std::unique_ptr<Criterion> object;
		const kind_t kind;

		bool is_object() const { return kind == K_OBJECT; }
		bool is_gini() const { return kind == K_GINI; }
		bool is_entropy() const { return kind == K_GINI; }
		bool is_mse() const { return kind == K_GINI; }
		bool is_friedman_mse() const { return kind == K_GINI; }
	};

	// discriminative union for Splitter, can be one of:
	// owned Splitter*
	// - gini, entropy, mse, friedman_mse
	struct splitter_union {
		enum kind_t { K_OBJECT, K_BEST, K_PRESORT_BEST, K_RANDOM };
	private:
		kind_t kind_from_string(const char* s) {
			if(strcmp(s, "best") == 0) return K_BEST;
			else if(strcmp(s, "presort-best") == 0) return K_PRESORT_BEST;
			else if(strcmp(s, "random") == 0) return K_RANDOM;
			else
				throw std::runtime_error("splitter_union: string must be best, presort-best, random");
		}
	public:
		splitter_union() = delete;
		splitter_union(const splitter_union&) = delete;
		splitter_union(splitter_union&& x)
		: object(std::move(x.object))
		, kind(x.kind)
		{}

		splitter_union(std::unique_ptr<Splitter>&& object) //takes ownership
		: object(std::move(object))
		, kind(K_OBJECT)
		{}
		splitter_union(const char* s)
		: kind(kind_from_string(s))
		{}
		std::unique_ptr<Splitter> object;
		const kind_t kind;

		bool is_object() const { return kind == K_OBJECT; }
		bool is_best() const { return kind == K_BEST; }
		bool is_presort_best() const { return kind == K_PRESORT_BEST; }
		bool is_random() const { return kind == K_RANDOM; }
    };
}

#endif
