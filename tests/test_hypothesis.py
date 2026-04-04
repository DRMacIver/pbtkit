# This file is part of Pbtkit, which may be found at
# https://github.com/DRMacIver/pbtkit
#
# This work is copyright (C) 2020 David R. MacIver.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from collections import defaultdict
from random import Random

import pytest
from hypothesis import HealthCheck, given, note, reject, settings
from hypothesis import strategies as st

from pbtkit import Generator, Unsatisfiable, run_test
from pbtkit.core import Status


@Generator
def list_of_integers(test_case):
    result = []
    while test_case.weighted(0.9):
        result.append(test_case.choice(10000))
    return result


class Failure(Exception):
    pass


@settings(
    suppress_health_check=list(HealthCheck),
    deadline=None,
    report_multiple_bugs=False,
    max_examples=50,
)
@given(st.data())
def test_give_pbtkit_a_workout(data):
    seed = data.draw(st.integers(0, 1000))
    rnd = Random(seed)
    max_examples = data.draw(st.integers(1, 100))

    method_call = st.one_of(
        st.tuples(
            st.just("mark_status"),
            st.sampled_from((Status.INVALID, Status.VALID, Status.INTERESTING)),
        ),
        st.tuples(st.just("target"), st.floats(0.0, 1.0)),
        st.tuples(st.just("choice"), st.integers(0, 1000)),
        st.tuples(st.just("weighted"), st.floats(0.0, 1.0)),
    )

    def new_node():
        return [None, defaultdict(new_node)]

    tree = new_node()

    database = {}
    failed = False
    call_count = 0
    valid_count = 0

    try:
        try:

            @run_test(
                max_examples=max_examples,
                random=rnd,
                database=database,
                quiet=True,
            )
            def test_function(test_case):
                node = tree
                depth = 0
                nonlocal call_count, valid_count, failed
                call_count += 1

                while True:
                    depth += 1
                    if node[0] is None:
                        node[0] = data.draw(method_call)
                    if node[0] == ("mark_status", Status.INTERESTING):
                        failed = True
                        raise Failure()
                    if node[0] == ("mark_status", Status.VALID):
                        valid_count += 1
                    name, *rest = node[0]

                    result = getattr(test_case, name)(*rest)
                    node = node[1][result]

        except Failure:
            failed = True
        except Unsatisfiable:
            reject()

        if not failed:
            assert valid_count <= max_examples
            assert call_count <= max_examples * 10
    except Exception as e:

        @note
        def tree_as_code():
            """If the test fails, print out a test that will trigger that
            failure rather than making me hand-edit it into something useful."""

            i = 1
            while True:
                test_name = f"test_failure_from_hypothesis_{i}"
                if test_name not in globals():
                    break
                i += 1

            lines = [
                f"def {test_name}():",
                "    with pytest.raises(Failure):",
                f"        @run_test(max_examples=1000, database={{}}, random=Random({seed}))",
                "        def _(tc):",
            ]

            varcount = 0

            def recur(indent, node):
                nonlocal varcount

                if node[0] is None:
                    lines.append(" " * indent + "tc.reject()")
                    return

                method, *args = node[0]
                if method == "mark_status":
                    if args[0] == Status.INTERESTING:
                        lines.append(" " * indent + "raise Failure()")
                    elif args[0] == Status.VALID:
                        lines.append(" " * indent + "return")
                    elif args[0] == Status.INVALID:
                        lines.append(" " * indent + "tc.reject()")
                    else:
                        lines.append(
                            " " * indent + f"tc.mark_status(Status.{args[0].name})"
                        )
                elif method == "target":
                    lines.append(" " * indent + f"tc.target({args[0]})")
                    recur(indent, *node[1].values())
                elif method == "weighted":
                    cond = f"tc.weighted({args[0]})"
                    assert len(node[1]) > 0
                    if len(node[1]) == 2:
                        lines.append(" " * indent + "if {cond}:")
                        recur(indent + 4, node[1][True])
                        lines.append(" " * indent + "else:")
                        recur(indent + 4, node[1][False])
                    else:
                        if True in node[1]:
                            lines.append(" " * indent + f"if {cond}:")
                            recur(indent + 4, node[1][True])
                        else:
                            assert False in node[1]
                            lines.append(" " * indent + f"if not {cond}:")
                            recur(indent + 4, node[1][False])
                else:
                    varcount += 1
                    varname = f"n{varcount}"
                    lines.append(
                        " " * indent
                        + f"{varname} = tc.{method}({', '.join(map(repr, args))})"
                    )
                    first = True
                    for k, v in node[1].items():
                        if v[0] == ("mark_status", Status.INVALID):
                            continue
                        lines.append(
                            " " * indent
                            + ("if" if first else "elif")
                            + f" {varname} == {k}:"
                        )
                        first = False
                        recur(indent + 4, v)
                    lines.append(" " * indent + "else:")
                    lines.append(" " * (indent + 4) + "tc.reject()")

            recur(12, tree)
            return "\n".join(lines)

        raise e


def test_failure_from_hypothesis_1():
    with pytest.raises(Failure):

        @run_test(max_examples=1000, database={}, random=Random(100))
        def _(tc):
            n1 = tc.weighted(0.0)
            if not n1:
                n2 = tc.choice(511)
                if n2 == 112:
                    n3 = tc.choice(511)
                    if n3 == 124:
                        raise Failure()
                    elif n3 == 93:
                        raise Failure()
                    else:
                        tc.mark_status(Status.INVALID)
                elif n2 == 93:
                    raise Failure()
                else:
                    tc.mark_status(Status.INVALID)


def test_failure_from_hypothesis_2():
    with pytest.raises(Failure):

        @run_test(max_examples=1000, database={}, random=Random(0))
        def _(tc):
            n1 = tc.choice(6)
            if n1 == 6:
                n2 = tc.weighted(0.0)
                if not n2:
                    raise Failure()
            elif n1 == 4:
                n3 = tc.choice(0)
                if n3 == 0:
                    raise Failure()
                else:
                    tc.mark_status(Status.INVALID)
            elif n1 == 2:
                raise Failure()
            else:
                tc.mark_status(Status.INVALID)
