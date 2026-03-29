# This file is part of Minithesis, which may be found at
# https://github.com/DRMacIver/minithesis
#
# This work is copyright (C) 2020 David R. MacIver.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""
minithesis — a minimal property-based testing library.

This module re-exports the core (integers + booleans) from
minithesis.minithesis and adds float, bytes, and string support
by importing the type-specific submodules which register their
choice types, draw methods, serializers, and shrink passes.
"""

from __future__ import annotations

# Public API and internal names re-exported from the core.
from minithesis.minithesis import (
    Database,
    DirectoryDB,
    Generator,
    TestCase,
    Unsatisfiable,
    run_test,
)

__all__ = [
    "BytesChoice",
    "Database",
    "DirectoryDB",
    "FloatChoice",
    "Generator",
    "StringChoice",
    "TestCase",
    "Unsatisfiable",
    "run_test",
]

# Import type-specific modules for their side effects: each one
# registers its serializer, shrink pass, and draw method on TestCase.
from minithesis.bytes import BytesChoice
from minithesis.floats import FloatChoice
from minithesis.text import StringChoice
