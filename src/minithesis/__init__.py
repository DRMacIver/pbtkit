# This file is part of Minithesis, which may be found at
# https://github.com/DRMacIver/minithesis
#
# This work is copyright (C) 2020 David R. MacIver.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""
minithesis2 is a minimal property-based testing library.

It's not really intended to be used as is, but is instead designed
to illustrate the core ideas of Hypothesis for educational and porting
purposes.

minithesis2 is the second iteration of minithesis. The first, to be
found at https://github.com/DRMacIver/minithesis, implemented the
previous generation architecture of Hypothesis (based on a single
datatype). minithesis2 started as an attempt to modernise minithesis
to adopt the new Hypothesis architecture of having multiple types of
primitive value in its representation.

As part of that its complexity necessarily grew significantly, so it
has been significantly modularised. It is split up into a core and
a number of extension modules, each of which adds some piece of
functionality.

The current minithesis modules of interest are:

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
the basic idea of minithesis. Each of the individual datatypes are, of
course, quite important, but you can easily build simpler (but worse)
versions of them on top of the core without understanding that.

`target` is honestly mostly a gimmick and hasn't proven that useful in
practice. If you want to understand it, it's not very hard, but perhaps
wait until it's useful unless you're actively interested.

The test case database is *very* useful and I strongly encourage
you to support it, but if it's fiddly feel free to leave it out
of a first pass.

The caching layer you can skip. It's used more heavily in Hypothesis
proper, but in minithesis you only really need it for shrinking
performance, so it's mostly a nice to have.
"""

from __future__ import annotations

# Disable modules before importing extensions.
import minithesis.features

# Public API and internal names re-exported from the core.
from minithesis.core import (
    Generator,
    TestCase,
    Unsatisfiable,
    run_test,
)
from minithesis.database import (
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
import minithesis.bytes
import minithesis.caching
import minithesis.database
import minithesis.floats
import minithesis.shrinking.advanced_integer_passes
import minithesis.shrinking.bind_deletion
import minithesis.shrinking.duplication_passes
import minithesis.shrinking.sorting
import minithesis.targeting
import minithesis.text
