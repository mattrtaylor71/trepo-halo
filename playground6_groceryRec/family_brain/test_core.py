import json
from datetime import datetime, timedelta, timezone
import pytest
from .core import project, corrected_state, encode, prompt, MAX_BYTES, validate_preference

NOW = datetime(2026, 9, 25, tzinfo=timezone.utc)


def state(actor="a", household="h", **changes):
    return dict(actor_id=actor, household_id=household, enabled=True, paused=False,
                generation=1, revision=1, learning_since=NOW-timedelta(days=90), facts={}, **changes)


def evidence(actor="a", **changes):
    row = dict(actor_id=actor, household_id="h", kind="recipe_saved", entity_id="r1",
               at=NOW-timedelta(days=1), name="Bean tacos", committed=True)
    row.update(changes)
    return row


def render(states=None, rows=(), actor="a", purpose="thyme"):
    return project(actor=actor, household="h", purpose=purpose,
                   states=states or [state()], evidence=rows, now=NOW)


def test_imported_member_a_history_never_reaches_b():
    assert render([state("a"), state("b")], [evidence()], actor="b")["observations"] == {}


def test_household_background_never_includes_private_history():
    assert render(rows=[evidence()], actor=None)["observations"] == {}


@pytest.mark.parametrize("change", [dict(household_id="other"), dict(actor_id="b"),
    dict(at=NOW-timedelta(days=800)), dict(at=NOW+timedelta(seconds=1)),
    dict(retracted=True), dict(committed=False), dict(kind="recipe_viewed"),
    dict(kind="receipt_uploaded"), dict(kind="product_searched"), dict(entity_id="")])
def test_unqualified_evidence_excluded(change):
    assert render(rows=[evidence(**change)])["observations"] == {}


def test_semantic_duplicate_saves_count_once_and_deletion_removes():
    c = render(rows=[evidence(), evidence(), evidence()])
    assert c["observations"]["saved_recipe_interest"] == ["Bean tacos"]
    assert render(rows=())["observations"] == {}
    assert "liked" not in encode(c)


def test_correction_replaces_previous_preference():
    s = corrected_state(state(), dict(action="set", key="cuisines", value=["Thai"]), NOW)
    s = corrected_state(s, dict(action="set", key="cuisines", value=["Mexican"]), NOW)
    assert render([s])["explicit_preferences"]["cuisines"]["value"] == ["Mexican"]
    assert "Thai" not in encode(render([s]))


def test_personal_pause_and_reset_do_not_touch_other_member():
    a = state("a")
    b = corrected_state(state("b"), dict(action="reset"), NOW)
    assert a["generation"] == 1
    assert render([a, b], [evidence()])["observations"]
    assert render([a, b], [evidence("b")], actor="b")["observations"] == {}
    a = corrected_state(a, dict(action="pause"), NOW)
    assert render([a], [evidence()])["observations"] == {}


def test_resume_does_not_relearn_paused_or_reset_history():
    s = corrected_state(state(), dict(action="pause"), NOW-timedelta(days=2))
    s = corrected_state(s, dict(action="resume"), NOW)
    assert render([s], [evidence()])["observations"] == {}


def test_private_preferences_require_same_actor_and_shared_preferences_opt_in():
    s = corrected_state(state(), dict(action="set", key="favorite_foods", value=["Peaches"]), NOW)
    assert render([s], actor=None)["explicit_preferences"] == {}
    assert render([s], actor="b")["explicit_preferences"] == {}
    s = corrected_state(s, dict(action="set", key="favorite_foods", value=["Peaches"], visibility="household"), NOW)
    assert render([s], actor=None)["explicit_preferences"]["favorite_foods"]["scope"] == "household"


def test_personal_choice_wins_regardless_of_subject_sort_order():
    a = corrected_state(state("a"), dict(action="set", key="cuisines", value=["Thai"]), NOW)
    b = corrected_state(state("b"), dict(action="set", key="cuisines", value=["Mexican"], visibility="household"), NOW)
    assert render([b, a])["explicit_preferences"]["cuisines"]["value"] == ["Thai"]


def test_expired_fact_and_departed_member_excluded():
    s = corrected_state(state(), dict(action="set", key="cuisines", value=["Thai"], expires_at=(NOW+timedelta(days=1)).isoformat()), NOW)
    c = project(actor="a", household="h", purpose="thyme", states=[s], now=NOW+timedelta(days=2))
    assert not c["explicit_preferences"]
    s["household_id"] = "different"
    assert not render([s])["explicit_preferences"]


def test_prompt_bounds_bytes_even_with_unicode_and_large_history():
    s = state()
    for key in ["cuisines", "favorite_foods", "avoid_foods", "preferred_products", "cooking_style"]:
        s = corrected_state(s, dict(action="set", key=key, value=[("🥑"*70)+str(i) for i in range(8)]), NOW)
    c = render([s], [evidence(entity_id=str(i), name="x"*1000) for i in range(192)])
    assert len(encode(c).encode("utf8")) <= MAX_BYTES
    assert json.loads(encode(c)) == c


@pytest.mark.parametrize("key,value", [("allergies", ["none"]), ("religion", ["x"]),
    ("max_prep_minutes", True), ("max_prep_minutes", 1000), ("cuisines", "Thai"),
    ("cuisines", ["x"*81]), ("cuisines", []), ("cuisines", [None])])
def test_no_hard_or_sensitive_inference_fields(key, value):
    with pytest.raises(ValueError):
        validate_preference(key, value)


def test_empty_context_has_no_prompt_and_rules_state_precedence():
    assert prompt(render()) == ""
    text = prompt(render(rows=[evidence()]))
    assert "Current dietary restrictions" in text
    assert "not liking or cooking" in text


def test_forget_is_field_scoped_and_explicit_preferences_survive_reset():
    s = corrected_state(state(), dict(action="set", key="cuisines", value=["Thai"]), NOW)
    s = corrected_state(s, dict(action="set", key="max_prep_minutes", value=25), NOW)
    s = corrected_state(s, dict(action="forget", key="cuisines"), NOW)
    s = corrected_state(s, dict(action="reset"), NOW)
    assert render([s])["explicit_preferences"] == {"max_prep_minutes": {"value": 25, "scope": "personal"}}
