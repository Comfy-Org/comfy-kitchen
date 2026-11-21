// SPDX-License-Identifier: Apache-2.0
// Extension for GPUs that only run the geometry kernels (wave64 CDNA); see CMakeLists.txt.
#include <nanobind/nanobind.h>

namespace nb = nanobind;

void register_geometry(nb::module_& m);

NB_MODULE(_geom, m) {
    register_geometry(m);
}
