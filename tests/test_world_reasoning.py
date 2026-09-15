from prvr_agent.schemas import CandidateWorldObservation, ProspectiveEventWorld, ProspectiveWorldSet, WorldEvidence
from prvr_agent.world_reasoning import fuse_candidate_score, prospective_world_score, revise_world_beliefs


def _worlds():
    return ProspectiveWorldSet(
        query="put cake in oven",
        worlds=[
            ProspectiveEventWorld(id="W1", query_anchor="anchor", prior=0.5),
            ProspectiveEventWorld(id="W2", query_anchor="anchor", prior=0.5),
        ],
    )


def test_visual_support_updates_world_posterior():
    observation = CandidateWorldObservation(
        query_satisfaction=0.8,
        counterfactual_risk=0.1,
        world_evidence=[
            WorldEvidence(world_id="W1", support=0.9, contradiction=0.0),
            WorldEvidence(world_id="W2", support=0.1, contradiction=0.7),
        ],
    )
    beliefs = revise_world_beliefs(_worlds(), observation)
    assert beliefs[0].posterior > beliefs[1].posterior
    assert prospective_world_score(beliefs) > 0


def test_counterfactual_risk_lowers_candidate_score():
    good = CandidateWorldObservation(query_satisfaction=0.9, counterfactual_risk=0.0)
    bad = CandidateWorldObservation(query_satisfaction=0.9, counterfactual_risk=0.9)
    assert fuse_candidate_score(0.8, good, 0.0) > fuse_candidate_score(0.8, bad, 0.0)
