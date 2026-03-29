# This file is part of Minithesis, which may be found at
# https://github.com/DRMacIver/minithesis
#
# This work is copyright (C) 2020 David R. MacIver.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from random import Random

import pytest

from minithesis import run_test
from minithesis.generators import integers, lists

pytestmark = pytest.mark.requires("collections")


@pytest.mark.parametrize("seed", range(10))
def test_finds_small_list(capsys, seed):

    with pytest.raises(AssertionError):

        @run_test(database={}, random=Random(seed))
        def _(test_case):
            ls = test_case.any(lists(integers(0, 10000)))
            assert sum(ls) <= 1000

    captured = capsys.readouterr()

    assert (
        captured.out.strip()
        == "any(lists(integers(min_value=0, max_value=10000), min_size=0, max_size=inf)): [1001]"
    )
