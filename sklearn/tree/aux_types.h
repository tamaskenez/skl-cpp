#ifndef AUX_TYPES_INCLUDED_270349234
#define AUX_TYPES_INCLUDED_270349234

#include <vector>

namespace sklearn {

	// ad-hoc discriminative union of
	// - int
	// - float
	// - 'auto', 'sqrt', 'log2'
	// - or none of the above
	struct max_features_union {
	private:
		enum kind_t { K_NONE, K_STRING, K_INT, K_FLOAT };
	public:
        const int i = INT_MIN;
        const float f = NAN;
        const std::string string;
        const kind_t kind;
        
		max_features_union() = delete;
        max_features_union(std::nullptr_t) : kind(K_NONE) {}
        max_features_union(int i) : i(i), kind(K_INT) {}
		max_features_union(float f) : f(f), kind(K_FLOAT) {}
		max_features_union(const char *s)
		: string(s), kind(K_STRING)
		{}

		bool is_int() const { return kind == K_INT; }
		bool is_float() const { return kind == K_FLOAT; }
		bool is_none() const { return kind == K_NONE; }
        bool is_string() const { return kind == K_STRING; }
        bool is_string(const char* s) const { return is_string() && string == s; }
	};

	// discriminative union for Criterion, can be one of:
	// owned Criterion*
	// - gini, entropy, mse, friedman_mse
	struct criterion_union {
        enum kind_t { K_OBJECT, K_STRING };
	public:
        std::unique_ptr<Criterion> object;
        const kind_t kind;
        const std::string string;

        criterion_union() = delete;
		criterion_union(const criterion_union&) = delete;
		criterion_union(criterion_union&& x)
		: object(std::move(x.object))
		, kind(x.kind)
        , string(std::move(x.string))
		{}

		criterion_union(std::unique_ptr<Criterion>&& object) //takes ownership
		: object(std::move(object))
		, kind(K_OBJECT)
		{}
		criterion_union(const char* s)
		: kind(K_STRING)
        , string(s)
		{}

		bool is_object() const { return kind == K_OBJECT; }
		bool is_string() const { return kind == K_STRING; }
        bool is_string(const char* s) const { return is_string() && string == s; }
	};

	// discriminative union for Splitter, can be one of:
	// owned Splitter*
	// - gini, entropy, mse, friedman_mse
	struct splitter_union {
		enum kind_t { K_OBJECT, K_STRING };
	public:
        std::unique_ptr<Splitter> object;
        const kind_t kind;
        std::string string;

		splitter_union() = delete;
		splitter_union(const splitter_union&) = delete;
		splitter_union(splitter_union&& x)
		: object(std::move(x.object))
		, kind(x.kind)
        , string(x.string)
		{}

		splitter_union(std::unique_ptr<Splitter>&& object) //takes ownership
		: object(std::move(object))
		, kind(K_OBJECT)
		{}
		splitter_union(const char* s)
		: kind(K_STRING)
        , string(s)
		{}

        bool is_object() const { return kind == K_OBJECT; }
        bool is_string() const { return kind == K_STRING; }
        bool is_string(const char* s) const { return is_string() && string == s; }
    };
}

#endif
