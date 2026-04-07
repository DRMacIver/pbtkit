# This file is part of Pbtkit, which may be found at
# https://github.com/DRMacIver/pbtkit
#
# This work is copyright (C) 2020 David R. MacIver.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""
pbtkit2 is a minimal property-based testing library.

It's not really intended to be used as is, but is instead designed
to illustrate the core ideas of Hypothesis for educational and porting
purposes.

pbtkit2 is the second iteration of pbtkit. The first, to be
found at https://github.com/DRMacIver/pbtkit, implemented the
previous generation architecture of Hypothesis (based on a single
datatype). pbtkit2 started as an attempt to modernise pbtkit
to adopt the new Hypothesis architecture of having multiple types of
primitive value in its representation.

As part of that its complexity necessarily grew significantly, so it
has been significantly modularised. It is split up into a core and
a number of extension modules, each of which adds some piece of
functionality.

The current pbtkit modules of interest are:

* core: This implements a minimal hypothesis-style property-based
  testing library. It provides primitives for booleans and integers,
  and support for integrated shrinking and a test database. It also
  provides a number of unimplemented methods that represent the full
  feature set.
* generators: This is a complete, if simplistic, set of generators
  as might be found in the hypothesis.strategies module. Many of them
  depend on specific functionality that is implemented in other modules.
* floats, bytes, text, colletions: These provide tools that are needed
  to support generating and shrinking their respective data types.
* target: This provides an implementation of the `target` function,
  which can be used for guiding tests towards greater or lesser values
  of some score.

Understanding and porting `core` is the the main thing you need to get
the basic idea of pbtkit. Each of the individual datatypes are, of
course, quite important, but you can easily build simpler (but worse)
versions of them on top of the core without understanding that.

`target` is honestly mostly a gimmick and hasn't proven that useful in
practice. If you want to understand it, it's not very hard, but perhaps
wait until it's useful unless you're actively interested.

The test case database is *very* useful and I strongly encourage
you to support it, but if it's fiddly feel free to leave it out
of a first pass.

The caching layer you can skip. It's used more heavily in Hypothesis
proper, but in pbtkit you only really need it for shrinking
performance, so it's mostly a nice to have.
"""

from __future__ import annotations

# Disable modules before importing extensions.
import pbtkit.features

# Public API and internal names re-exported from the core.
from pbtkit.core import (
    Generator,
    TestCase,
    Unsatisfiable,
    run_test,
)
from pbtkit.database import (
    Database,
    DirectoryDB,
)

__all__ = [
    "Database",
    "DirectoryDB",
    "Generator",
    "TestCase",
    "Unsatisfiable",
    "run_test",
]

# Import type-specific modules for their side effects: each one
# registers its serializer, shrink pass, and draw method on TestCase.
import pbtkit.bytes
import pbtkit.caching
import pbtkit.collections
import pbtkit.database
import pbtkit.draw_names
import pbtkit.edge_case_boosting
import pbtkit.floats
import pbtkit.multi_example
import pbtkit.shrinking.advanced_integer_passes
import pbtkit.shrinking.bind_deletion
import pbtkit.shrinking.duplication_passes
import pbtkit.shrinking.sorting
import pbtkit.spans
import pbtkit.targeting
import pbtkit.text

# Advanced passes that depend on specific type modules or features.
# Hardcoded dependency list: each entry is (module, required_feature).
_FEATURE_DEPENDENT_MODULES = {
    "pbtkit.shrinking.advanced_bytes_passes": "bytes",
    "pbtkit.shrinking.advanced_string_passes": "text",
    "pbtkit.shrinking.index_passes": "indexing",
    "pbtkit.shrinking.mutation": "indexing",
    "pbtkit.span_mutation": "spans",
}

from pbtkit.features import DISABLED_MODULES as _disabled

for _mod in sorted(
    m for m, f in _FEATURE_DEPENDENT_MODULES.items() if f not in _disabled
):
    __import__(_mod)
