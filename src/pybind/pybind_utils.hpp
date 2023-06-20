/*
 * Copyright 2023 Blue Brain Project, EPFL.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <memory>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace pybind11 {
namespace detail {

template <typename StringType>
struct CopyFromPython {
    void operator()(char* start, size_t n, StringType data) {
        char* buffer;
        ssize_t length;
        if (PYBIND11_BYTES_AS_STRING_AND_SIZE(data.ptr(), &buffer, &length))
            pybind11_fail("Unable to extract string contents! (invalid type)");
        std::memcpy(start, buffer, n);
    }
};


template <>
struct CopyFromPython<str> {
    void operator()(char* start, size_t n, str data) {
        if (PyUnicode_Check(data.ptr())) {
            data = reinterpret_steal<object>(PyUnicode_AsUTF8String(data.ptr()));
            if (!data)
                pybind11_fail("Unable to extract string contents! (encoding issue)");
        }
        CopyFromPython<bytes>()(start, n, data);
    }
};


template <typename StringType>
class pythonibuf: public std::streambuf {
  private:
    using traits_type = std::streambuf::traits_type;

    const static std::size_t put_back_ = 1;
    const static std::size_t buf_sz = 1024 + put_back_;
    char d_buffer[buf_sz];

    object pyistream;
    object pyread;

    // copy ctor and assignment not implemented;
    // copying not allowed
    pythonibuf(const pythonibuf&);
    pythonibuf& operator=(const pythonibuf&);

    int_type underflow() {
        if (gptr() < egptr()) {  // buffer not exhausted
            return traits_type::to_int_type(*gptr());
        }

        char* base = d_buffer;
        char* start = base;
        if (eback() == base) {
            std::memmove(base, egptr() - put_back_, put_back_);
            start += put_back_;
        }
        StringType data = pyread(buf_sz - (start - base));
        size_t n = len(data);
        if (n == 0) {
            return traits_type::eof();
        }
        CopyFromPython<StringType>()(start, n, data);
        setg(base, start, start + n);
        return traits_type::to_int_type(*gptr());
    }


  public:
    pythonibuf(object pyistream)
        : pyistream(pyistream)
        , pyread(pyistream.attr("read")) {
        char* end = d_buffer + buf_sz;
        setg(end, end, end);
    }
};
}  // namespace detail
}  // namespace pybind11

class VisitorOStreamResources {
  protected:
    std::unique_ptr<pybind11::detail::pythonbuf> buf;
    std::unique_ptr<std::ostream> ostream;

  public:
    VisitorOStreamResources() = default;
    VisitorOStreamResources(pybind11::object object)
        : buf(new pybind11::detail::pythonbuf(object))
        , ostream(new std::ostream(buf.get())) {}
    void flush() {
        ostream->flush();
    }
};
