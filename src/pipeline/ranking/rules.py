"""Business rules engine for post-ranking controls: exclusion, boosting, and pinning."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class RuleType(str, Enum):
    EXCLUDE = "exclude"
    BOOST = "boost"
    PIN = "pin"


@dataclass
class BusinessRule:
    """A business rule that alters recommendation candidate ranking."""

    rule_id: str
    rule_type: RuleType
    item_ids: set[int] = field(default_factory=set)
    genres: set[str] = field(default_factory=set)
    value: float = 1.0  # Multiplier for BOOST rules (e.g. 1.5x)
    position: int | None = None  # 1-indexed target position for PIN rules
    start_date: datetime | str | None = None
    end_date: datetime | str | None = None
    priority: int = 0  # Higher priority breaks ties in conflicts
    tenant_id: str = "default"

    def __post_init__(self) -> None:
        if isinstance(self.rule_type, str) and not isinstance(self.rule_type, RuleType):
            self.rule_type = RuleType(self.rule_type.lower())
        if self.item_ids is None:
            self.item_ids = set()
        elif not isinstance(self.item_ids, set):
            self.item_ids = set(self.item_ids)
        if self.genres is None:
            self.genres = set()
        elif not isinstance(self.genres, set):
            self.genres = {str(g).strip().lower() for g in self.genres}
        else:
            self.genres = {str(g).strip().lower() for g in self.genres}


@dataclass
class RuleConflict:
    """Represents a conflict detected between two or more business rules."""

    rule_ids: list[str]
    conflict_type: str
    description: str


def _parse_datetime(dt: datetime | str | None) -> datetime | None:
    if dt is None:
        return None
    if isinstance(dt, datetime):
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    # Parse ISO string
    parsed = datetime.fromisoformat(str(dt).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


class RuleEngine:
    """Rule engine applying business rules (exclusion, boost, pinning) after ranking."""

    def __init__(self, rules: list[BusinessRule] | None = None) -> None:
        self.rules: list[BusinessRule] = list(rules) if rules else []

    def add_rule(self, rule: BusinessRule) -> None:
        self.rules.append(rule)

    def remove_rule(self, rule_id: str, tenant_id: str | None = None) -> bool:
        """Remove a rule by ID, optionally restricted to a specific tenant."""
        initial_count = len(self.rules)
        self.rules = [
            r
            for r in self.rules
            if not (r.rule_id == rule_id and (tenant_id is None or r.tenant_id == tenant_id))
        ]
        return len(self.rules) < initial_count

    def is_rule_active(self, rule: BusinessRule, now: datetime | None = None) -> bool:
        if now is None:
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        start = _parse_datetime(rule.start_date)
        if start is not None and now < start:
            return False

        end = _parse_datetime(rule.end_date)
        if end is not None and now > end:
            return False

        return True

    def get_active_rules(
        self,
        now: datetime | None = None,
        tenant_id: str | None = None,
    ) -> list[BusinessRule]:
        return [
            r
            for r in self.rules
            if self.is_rule_active(r, now) and (tenant_id is None or r.tenant_id == tenant_id)
        ]

    def detect_conflicts(
        self,
        now: datetime | None = None,
        tenant_id: str | None = None,
    ) -> list[RuleConflict]:
        active_rules = self.get_active_rules(now, tenant_id=tenant_id)
        conflicts: list[RuleConflict] = []

        exclude_items: set[int] = set()
        for r in active_rules:
            if r.rule_type == RuleType.EXCLUDE:
                exclude_items.update(r.item_ids)

        # Check for exclude vs pin conflict
        for r in active_rules:
            if r.rule_type == RuleType.PIN:
                colliding_items = r.item_ids & exclude_items
                if colliding_items:
                    conflicts.append(
                        RuleConflict(
                            rule_ids=[r.rule_id],
                            conflict_type="exclude_overrides_pin",
                            description=(
                                f"Rule '{r.rule_id}' attempts to pin items {colliding_items} "
                                f"that are excluded by active exclusion rules."
                            ),
                        )
                    )

        # Check for multiple pins at same position
        pins_by_pos: dict[int, list[BusinessRule]] = {}
        for r in active_rules:
            if r.rule_type == RuleType.PIN and r.position is not None:
                pins_by_pos.setdefault(r.position, []).append(r)

        for pos, pin_rules in pins_by_pos.items():
            if len(pin_rules) > 1:
                conflicts.append(
                    RuleConflict(
                        rule_ids=[r.rule_id for r in pin_rules],
                        conflict_type="pin_collision",
                        description=(
                            f"Multiple rules attempt to pin items to position {pos}: "
                            f"{[r.rule_id for r in pin_rules]}."
                        ),
                    )
                )

        return conflicts

    def apply(
        self,
        items: list[Any],
        now: datetime | None = None,
        top_k: int | None = None,
        item_lookup: dict[int, dict] | None = None,
        tenant_id: str | None = None,
    ) -> list[Any]:
        """Apply active business rules: exclusion -> boost & sort -> pinning."""
        active_rules = self.get_active_rules(now, tenant_id=tenant_id)
        if not active_rules:
            return items[:top_k] if top_k is not None else items

        # Helper to get attributes uniformly whether item is dict or object
        def get_item_id(it: Any) -> int:
            return int(it["item_id"] if isinstance(it, dict) else getattr(it, "item_id"))

        def get_item_score(it: Any) -> float:
            sc = it["score"] if isinstance(it, dict) else getattr(it, "score", 0.0)
            return float(sc) if sc is not None else 0.0

        def set_item_score(it: Any, score: float) -> None:
            if isinstance(it, dict):
                it["score"] = score
            else:
                setattr(it, "score", score)

        def get_item_genres(it: Any) -> set[str]:
            genres_raw = it.get("genres", "") if isinstance(it, dict) else getattr(it, "genres", "")
            if isinstance(genres_raw, str):
                return {
                    g.strip().lower() for g in genres_raw.replace("|", ",").split(",") if g.strip()
                }
            if isinstance(genres_raw, set | list):
                return {str(g).strip().lower() for g in genres_raw if str(g).strip()}
            return set()

        # Step 1: Gather exclusion sets
        excluded_ids: set[int] = set()
        excluded_genres: set[str] = set()
        for r in active_rules:
            if r.rule_type == RuleType.EXCLUDE:
                excluded_ids.update(r.item_ids)
                excluded_genres.update(r.genres)

        # Filter items
        filtered: list[Any] = []
        for it in items:
            iid = get_item_id(it)
            if iid in excluded_ids:
                continue
            item_gen = get_item_genres(it)
            if excluded_genres and (item_gen & excluded_genres):
                continue
            filtered.append(dict(it) if isinstance(it, dict) else it)

        # Step 2: Apply BOOST rules
        boost_rules = [r for r in active_rules if r.rule_type == RuleType.BOOST]
        if boost_rules:
            for it in filtered:
                iid = get_item_id(it)
                item_gen = get_item_genres(it)
                current_score = get_item_score(it)
                multiplier = 1.0
                for br in boost_rules:
                    if (iid in br.item_ids) or (br.genres and (item_gen & br.genres)):
                        multiplier *= br.value
                set_item_score(it, current_score * multiplier)

            # Re-sort after boosting descending by score
            filtered.sort(key=get_item_score, reverse=True)

        # Step 3: Apply PIN rules
        pin_rules = [
            r
            for r in active_rules
            if r.rule_type == RuleType.PIN and r.position is not None and r.position >= 1
        ]
        if pin_rules:
            # Sort pin rules by priority descending
            pin_rules.sort(key=lambda r: r.priority, reverse=True)

            pinned_slots: dict[int, Any] = {}
            # Track items already placed via pinning
            pinned_item_ids: set[int] = set()

            for pr in pin_rules:
                for iid in sorted(pr.item_ids):
                    # Exclusion always takes precedence over pinning
                    if iid in excluded_ids:
                        continue
                    if iid in pinned_item_ids:
                        continue

                    # Check if item exists in filtered list
                    existing_it = next((it for it in filtered if get_item_id(it) == iid), None)
                    if existing_it is not None:
                        item_obj = existing_it
                        filtered = [it for it in filtered if get_item_id(it) != iid]
                    elif item_lookup and iid in item_lookup:
                        meta = item_lookup[iid]
                        if isinstance(items[0], dict) if items else True:
                            item_obj = {
                                "item_id": iid,
                                "title": meta.get("title", f"Item {iid}"),
                                "genres": meta.get("genres", ""),
                                "score": 1.0,
                            }
                        else:
                            # Model object like RecommendationItem
                            cls = type(items[0])
                            item_obj = cls(
                                item_id=iid,
                                title=meta.get("title", f"Item {iid}"),
                                genres=meta.get("genres", ""),
                                score=1.0,
                            )
                    else:
                        continue

                    target_pos = pr.position or 1
                    # Resolve pin collisions by finding next available slot
                    while target_pos in pinned_slots:
                        target_pos += 1

                    pinned_slots[target_pos] = item_obj
                    pinned_item_ids.add(iid)

            # Reassemble list with pins inserted at exact 1-indexed positions
            max_pos = max(pinned_slots.keys()) if pinned_slots else 0
            final_len = max(len(filtered) + len(pinned_slots), max_pos)
            result_list: list[Any] = []
            filtered_idx = 0

            for slot_1based in range(1, final_len + 1):
                if slot_1based in pinned_slots:
                    result_list.append(pinned_slots[slot_1based])
                elif filtered_idx < len(filtered):
                    result_list.append(filtered[filtered_idx])
                    filtered_idx += 1

            # Append any remaining unpinned items
            while filtered_idx < len(filtered):
                result_list.append(filtered[filtered_idx])
                filtered_idx += 1

            filtered = result_list

        return filtered[:top_k] if top_k is not None else filtered
