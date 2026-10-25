"""Citation guard: the LLM may only use codes and sources that retrieval returned.

Structured outputs guarantee the *shape* of the model's answer, not its truthfulness. The
guard enforces grounding mechanically:

* a candidate's ``hts_code`` must normalise to an HTS line that was in the context;
* every citation must be a ``chunk_id`` that was in the context (unknown ones are dropped);
* a candidate must cite its own tariff line (``hts:<code>``), as the system prompt requires.
  Citing only a neighbouring line (often the 8-digit parent of a 10-digit "Other") would
  let a code be "grounded" by text that does not describe it, so such a candidate is
  rejected rather than silently repaired;
* description and rate are replaced with the authoritative values from the tariff data, so
  a model that "remembers" a different rate cannot leak it into the answer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from landedcost.classify.models import LlmCandidate, LlmClassification
from landedcost.hts.codes import HtsCode, InvalidHtsCodeError
from landedcost.retrieval.hybrid import Candidate


@dataclass(frozen=True, slots=True)
class GuardedCandidate:
    candidate: Candidate
    rationale: str
    citations: tuple[str, ...]


@dataclass(slots=True)
class GuardOutcome:
    kept: list[GuardedCandidate] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    dropped_citations: list[str] = field(default_factory=list)


def apply_citation_guard(
    answer: LlmClassification,
    context_candidates: Mapping[str, Candidate],
    context_chunk_ids: frozenset[str],
) -> GuardOutcome:
    """Filter ``answer`` down to what the retrieved context supports.

    ``context_candidates`` maps canonical HTS codes to the candidates shown to the model;
    ``context_chunk_ids`` is every chunk id (HTS lines and rulings) in the prompt.
    """
    outcome = GuardOutcome()
    seen_codes: set[str] = set()
    for item in answer.candidates:
        reason = _reject_reason(item, context_candidates)
        if reason:
            outcome.rejected.append(reason)
            continue
        code = HtsCode.parse(item.hts_code).dotted
        if code in seen_codes:
            continue
        valid = tuple(dict.fromkeys(c for c in item.citations if c in context_chunk_ids))
        outcome.dropped_citations.extend(c for c in item.citations if c not in context_chunk_ids)
        if not valid:
            outcome.rejected.append(f"{code}: no citation refers to a retrieved source")
            continue
        own = context_candidates[code].chunk_id
        if own not in valid:
            outcome.rejected.append(f"{code}: does not cite its own tariff line ({own})")
            continue
        # The candidate's own line always comes first, then any supporting sources.
        valid = (own, *(c for c in valid if c != own))
        seen_codes.add(code)
        outcome.kept.append(
            GuardedCandidate(context_candidates[code], item.rationale.strip(), valid)
        )
    return outcome


def _reject_reason(item: LlmCandidate, context: Mapping[str, Candidate]) -> str | None:
    try:
        code = HtsCode.parse(item.hts_code).dotted
    except InvalidHtsCodeError:
        return f"{item.hts_code!r}: not a valid HTS code"
    if code not in context:
        return f"{code}: not among the retrieved tariff lines"
    return None
