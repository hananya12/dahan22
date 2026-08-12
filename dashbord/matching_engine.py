"""
vision/identity/matching_engine.py
----------------------------------
Decides whether a new appearance belongs to someone we already know.

The asymmetry that drives every design choice here
--------------------------------------------------
The two possible errors are NOT equally bad:

  * FALSE SPLIT (one person counted as two) inflates Visitors Today. It is
    visible, self-correcting over time, and merely embarrassing.
  * FALSE MERGE (two people collapsed into one) is silent and permanent. It
    corrupts that identity's gallery with a second person's appearance,
    which causes further merges, which corrupts it further. It destroys
    Journey and Average Visit Time, and there is no way to detect it after
    the fact.

So the engine is deliberately biased toward splitting, and is allowed to
return AMBIGUOUS rather than guess. An ambiguous track stays in probation and
gets re-evaluated with better crops. In a retail counter, "wait one more
second before deciding" costs nothing.

Three gates must all pass for a MATCH:
  1. absolute:  best score >= match_threshold
  2. relative:  best beats runner-up by min_margin_over_runner_up
  3. feasible:  the candidate is temporally/spatially reachable
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from config import MatchingConfig
from contracts import Embedding, MatchResult, Verdict
from person_profile import PersonProfile


class MatchingEngine:
    """Stateless scorer. All state lives in MemoryEngine — which means this
    class can be exhaustively unit-tested with hand-built profiles."""

    def __init__(self, config: MatchingConfig) -> None:
        self.config = config

    def match(
        self,
        query: Embedding,
        candidates: Sequence[PersonProfile],
        feasible: Optional[Callable[[PersonProfile], bool]] = None,
        exclude_ids: Iterable[str] = (),
        position: Optional[Tuple[float, float]] = None,
        now: Optional[float] = None,
        frame_width: Optional[float] = None,
    ) -> MatchResult:
        """Score *query* against *candidates* and return an explained verdict.

        Args:
            feasible: optional reachability predicate. Two identities cannot
                be the same person if they were seen 2 seconds apart at
                opposite ends of the store, or on two cameras 40 metres apart.
                Rejecting those BEFORE scoring is free accuracy — appearance
                models have no concept of physics, so we supply it.
            exclude_ids: identities already claimed by another live track in
                this same frame. One person cannot be in two places at once,
                so an id bound to a currently-visible track is not available.
        """
        excluded = set(exclude_ids)
        scored: List[Tuple[float, PersonProfile]] = []

        for profile in candidates:
            if profile.global_id in excluded:
                continue
            if not profile.gallery:
                continue
            if feasible is not None and not feasible(profile):
                continue

            score = profile.similarity_to(query)
            if self.config.use_motion_prior and position and frame_width and now:
                reachable, proximity = self._motion_prior(profile, position, now, frame_width)
                if not reachable:
                    # Physically impossible. Appearance does not get a vote.
                    continue
                score += self.config.motion_bonus * proximity
            scored.append((min(1.0, score), profile))

        if not scored:
            return MatchResult(verdict=Verdict.NEW, candidates_considered=0)

        scored.sort(key=lambda pair: pair[0], reverse=True)
        best_score, best = scored[0]
        runner_score, runner_id = (scored[1][0], scored[1][1].global_id) if len(scored) > 1 else (0.0, None)

        result = MatchResult(
            verdict=Verdict.NEW,
            global_id=best.global_id,
            score=best_score,
            runner_up_score=runner_score,
            runner_up_id=runner_id,
            candidates_considered=len(scored),
        )

        cfg = self.config

        # --- solo re-association -------------------------------------------
        if (cfg.solo_reassociation and position and frame_width and now
                and result.verdict is not Verdict.MATCH):
            solo = self._solo_candidate(scored, position, now, frame_width)
            if solo is not None:
                result.global_id = solo.global_id
                result.verdict = Verdict.MATCH
                result.score = max(best_score, cfg.match_threshold)
                return result

        if best_score < cfg.new_threshold:
            result.verdict = Verdict.NEW
            result.global_id = None
            return result

        if best_score < cfg.match_threshold:
            # In the grey band: plausible, not proven. Refuse to decide.
            result.verdict = Verdict.AMBIGUOUS
            return result

        if runner_id is not None and (best_score - runner_score) < cfg.min_margin_over_runner_up:
            # Two people fit equally well — the classic uniformed-staff case.
            result.verdict = Verdict.AMBIGUOUS
            return result

        result.verdict = Verdict.MATCH
        return result

    # ------------------------------------------------------------------

    def _solo_candidate(self, scored, position, now, frame_width):
        """The one identity this could possibly be — or None.

        Deliberately conservative: any hint of a second person in the recent
        past disqualifies the whole rule, because that is the only situation
        in which it could merge two human beings.
        """
        cfg = self.config
        window = cfg.solo_max_gap_seconds
        recent = [p for _, p in scored if now - p.last_seen <= window]
        if len(recent) != 1:
            return None                      # nobody, or more than one: abstain
        if len(scored) > 1:
            # Someone else exists in memory. Only proceed if they are long gone.
            others = [p for _, p in scored if p is not recent[0]]
            if any(now - p.last_seen <= window for p in others):
                return None

        candidate = recent[0]
        if candidate.last_position is None:
            return None
        dx = position[0] - candidate.last_position[0]
        dy = position[1] - candidate.last_position[1]
        if (dx * dx + dy * dy) ** 0.5 > cfg.solo_max_distance_frac * frame_width:
            return None
        return candidate

    def _motion_prior(self, profile: PersonProfile, position: Tuple[float, float],
                      now: float, frame_width: float) -> Tuple[bool, float]:
        """(reachable, proximity 0..1) for a candidate, from motion alone."""
        last = profile.last_position
        if last is None:
            return True, 0.0
        elapsed = max(0.05, now - profile.last_seen)
        reach = self.config.max_speed_frac_per_s * frame_width * elapsed
        distance = ((position[0] - last[0]) ** 2 + (position[1] - last[1]) ** 2) ** 0.5
        if distance > reach * self.config.motion_tolerance:
            return False, 0.0
        return True, max(0.0, 1.0 - distance / max(reach, 1.0))

    def evaluate_probation(self, votes: Sequence[MatchResult]) -> MatchResult:
        """Collapse several frames of evidence into one committed decision.

        A track in probation is matched on each of its first N embeddable
        frames. Committing requires a MAJORITY of those frames to agree on
        the SAME id — one strong frame is not enough, because the strongest
        single frame is often the one that got lucky.
        """
        if not votes:
            return MatchResult(verdict=Verdict.AMBIGUOUS)

        tally: dict[str, List[MatchResult]] = {}
        for vote in votes:
            if vote.verdict is Verdict.MATCH and vote.global_id:
                tally.setdefault(vote.global_id, []).append(vote)

        if not tally:
            # Nobody was ever confidently matched -> genuinely a new person.
            return MatchResult(
                verdict=Verdict.NEW,
                score=max(v.score for v in votes),
                candidates_considered=max(v.candidates_considered for v in votes),
            )

        winner_id, winner_votes = max(tally.items(), key=lambda kv: (len(kv[1]), max(v.score for v in kv[1])))

        if len(winner_votes) * 2 <= len(votes):
            # No majority: the evidence is split. Prefer a new identity —
            # a false split is recoverable, a false merge is not.
            return MatchResult(
                verdict=Verdict.NEW,
                score=max(v.score for v in winner_votes),
                global_id=None,
            )

        best = max(winner_votes, key=lambda v: v.score)
        return MatchResult(
            verdict=Verdict.MATCH,
            global_id=winner_id,
            score=best.score,
            runner_up_score=best.runner_up_score,
            runner_up_id=best.runner_up_id,
            candidates_considered=best.candidates_considered,
        )