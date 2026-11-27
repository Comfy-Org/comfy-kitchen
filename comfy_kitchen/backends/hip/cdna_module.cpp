// SPDX-License-Identifier: Apache-2.0
// Extension for wave64 CDNA targets: the geometry kernels and the MFMA INT8 linear; see CMakeLists.txt.
#include <nanobind/nanobind.h>

namespace nb = nanobind;

void register_geometry(nb::module_& m);
void register_cdna_int8(nb::module_& m);

NB_MODULE(_cdna, m) {
    register_geometry(m);
    register_cdna_int8(m);
}
