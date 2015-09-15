#ifndef ERRORS_INCLUDED_20340234234
#define ERRORS_INCLUDED_20340234234

#include <stdexcept>
#include <string>

namespace sklearn {
    class ValueError: public std::runtime_error {
	public:
		explicit ValueError(const char* s)
		: std::runtime_error(s)
		{}
		explicit ValueError(const std::string& s)
		: std::runtime_error(s)
		{}
	};
    class NotFittedError: public std::runtime_error {
    public:
        explicit NotFittedError(const char* s)
        : std::runtime_error(s)
        {}
        explicit NotFittedError(const std::string& s)
        : std::runtime_error(s)
        {}
    };
}

#endif
