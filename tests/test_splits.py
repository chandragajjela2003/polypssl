"""Label splits: budget sizes, validation inside the budget, stratification, no overlap, determinism."""
import pytest

from polypssl.data.splits import BUDGETS, check_split, make_random_split, val_count


@pytest.fixture()
def pool():
    return ([{"id": f"cju{i}", "source": "Kvasir"} for i in range(900)]
            + [{"id": str(i), "source": "ClinicDB"} for i in range(1, 551)])


@pytest.mark.parametrize("budget", list(BUDGETS))
def test_budget_sizes_and_validation_inside(pool, budget):
    split = make_random_split(pool, budget, seed=0)
    check_split(split, pool)
    total = BUDGETS[budget]
    assert len(split["labelled_train"]) + len(split["labelled_val"]) == total
    assert len(split["labelled_val"]) == val_count(total)
    assert len(split["unlabelled"]) == 1450 - total


def test_five_percent_is_60_plus_12(pool):
    split = make_random_split(pool, "5", seed=0)
    assert (len(split["labelled_train"]), len(split["labelled_val"])) == (60, 12)


def test_stratified_by_source(pool):
    split = make_random_split(pool, "5", seed=3)
    labelled = split["labelled_train"] + split["labelled_val"]
    n_kvasir = sum(i.startswith("cju") for i in labelled)
    assert n_kvasir == 45 and len(labelled) - n_kvasir == 27   # 72 * 900/1450 = 44.7 -> 45


def test_deterministic_and_seed_dependent(pool):
    a, b, c = (make_random_split(pool, "5", s) for s in (1, 1, 2))
    assert a == b
    assert a["labelled_train"] != c["labelled_train"]
