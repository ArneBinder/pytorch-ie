import dataclasses
import logging
from typing import List

import torch
from pie_core import Annotation

from pytorch_ie.taskmodules.common import BatchableMixin
from pytorch_ie.taskmodules.common.mixins import RelationStatisticsMixin


def test_batchable_mixin():
    """Test the BatchableMixin class."""

    @dataclasses.dataclass
    class Foo(BatchableMixin):
        """A class that uses the BatchableMixin class."""

        a: List[int]

        @property
        def len_a(self):
            """Return the length of the list a."""
            return len(self.a)

    x = Foo(a=[1, 2, 3])
    y = Foo(a=[4, 5])

    batch = Foo.batch(
        values=[x, y], dtypes={"a": torch.int64, "len_a": torch.int64}, pad_values={"a": 0}
    )
    torch.testing.assert_close(batch["a"], torch.tensor([[1, 2, 3], [4, 5, 0]]))
    torch.testing.assert_close(batch["len_a"], torch.tensor([3, 2]))


def test_relation_statistics_mixin_show_statistics(caplog):
    """Test the RelationStatisticsMixin class."""

    class Foo(RelationStatisticsMixin):
        """A class that uses the RelationStatisticsMixin class."""

        pass

    @dataclasses.dataclass(eq=True, frozen=True)
    class TestAnnotation(Annotation):
        label: str
        score: float = dataclasses.field(default=1.0, compare=False)

    x = Foo(collect_statistics=True)

    relations = [
        TestAnnotation(label="A", score=1.0),
        TestAnnotation(label="B", score=0.5),
        TestAnnotation(label="C", score=0.0),
        TestAnnotation(label="D", score=0.3),
    ]
    # all available relations
    x.collect_all_relations(kind="available", relations=relations)
    # relations skipped for a reason ("test")
    x.collect_relation(kind="skipped_test", relation=relations[1])
    # mark two relations as used, one of them is skipped for another (unknown) reason
    x.collect_all_relations(kind="used", relations=[relations[0], relations[2]])

    statistics = x.get_statistics()

    assert statistics == {
        ("available", "A"): 1,
        ("available", "B"): 1,
        ("available", "D"): 1,
        ("available", "no_relation"): 1,
        ("skipped_other", "D"): 1,
        ("skipped_test", "B"): 1,
        ("used", "A"): 1,
        ("used", "no_relation"): 1,
    }

    with caplog.at_level(logging.INFO):
        x.show_statistics()
    assert caplog.messages[0] == (
        "Foo does not have a `none_label` attribute. "
        "Using default value 'no_relation'. "
        "`none_label` is used as the label for relations with score 0 in statistics and "
        "all relations with label different from `none_label` will be summarized to 'all_relations'. "
        "Set the `none_label` attribute before using statistics or "
        "overwrite `get_none_label_for_statistics()` function to get rid of this message."
    )
    assert caplog.messages[1] == (
        "statistics:\n"
        "|               |   available |   skipped_other |   skipped_test |   used |   used % |\n"
        "|:--------------|------------:|----------------:|---------------:|-------:|---------:|\n"
        "| A             |           1 |               0 |              0 |      1 |      100 |\n"
        "| B             |           1 |               0 |              1 |      0 |        0 |\n"
        "| D             |           1 |               1 |              0 |      0 |        0 |\n"
        "| no_relation   |           1 |               0 |              0 |      1 |      100 |\n"
        "| all_relations |           3 |               1 |              1 |      1 |       33 |"
    )


def test_relation_statistics_mixin_show_statistics_no_relations(caplog):
    """Test the RelationStatisticsMixin class with no predictions."""

    class Foo(RelationStatisticsMixin):
        """A class that uses the RelationStatisticsMixin class."""

        pass

    x = Foo(collect_statistics=True)

    # Test with no relations collected
    x.collect_all_relations(kind="available", relations=[])
    x.collect_all_relations(kind="used", relations=[])

    statistics = x.get_statistics()

    assert statistics == {}

    with caplog.at_level(logging.INFO):
        x.show_statistics()
    assert caplog.messages[0] == "statistics:\n" "|--:|\n" "| 0 |"


def test_relation_statistics_mixin_show_statistics_custom_none_label(caplog):
    """Test the RelationStatisticsMixin class with custom none_label."""

    class Foo(RelationStatisticsMixin):
        """A class that uses the RelationStatisticsMixin class.

        It also sets the `none_label` attribute which will be used by statistics.
        """

        def __init__(self, none_label: str = "no_relation", **kwargs):
            super().__init__(**kwargs)
            self.none_label = none_label

    @dataclasses.dataclass(eq=True, frozen=True)
    class TestAnnotation(Annotation):
        label: str
        score: float = dataclasses.field(default=1.0, compare=False)

    x = Foo(collect_statistics=True, none_label="None_Label")

    relations = [
        TestAnnotation(label="A", score=1.0),
        TestAnnotation(label="B", score=0.5),
        TestAnnotation(label="C", score=0.0),
        TestAnnotation(label="D", score=0.3),
    ]
    # all available relations
    x.collect_all_relations(kind="available", relations=relations)
    # relations skipped for a reason ("test")
    x.collect_relation(kind="skipped_test", relation=relations[1])
    # mark two relations as used, one of them is skipped for another (unknown) reason
    x.collect_all_relations(kind="used", relations=[relations[0], relations[2]])

    with caplog.at_level(logging.INFO):
        x.show_statistics()
    assert caplog.messages[0] == (
        "statistics:\n"
        "|               |   available |   skipped_other |   skipped_test |   used |   used % |\n"
        "|:--------------|------------:|----------------:|---------------:|-------:|---------:|\n"
        "| A             |           1 |               0 |              0 |      1 |      100 |\n"
        "| B             |           1 |               0 |              1 |      0 |        0 |\n"
        "| D             |           1 |               1 |              0 |      0 |        0 |\n"
        "| None_Label    |           1 |               0 |              0 |      1 |      100 |\n"
        "| all_relations |           3 |               1 |              1 |      1 |       33 |"
    )
