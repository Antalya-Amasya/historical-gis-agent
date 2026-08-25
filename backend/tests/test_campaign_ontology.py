from backend.app.rag.campaign_ontology import HistoricalCampaignOntology
from backend.app.route_orchestrator import HistoricalCampaignIntentRegistry


def resolve(message: str):
    return HistoricalCampaignIntentRegistry(HistoricalCampaignOntology.default()).resolve(message)


def test_ontology_models_campaign_hierarchy_without_coordinates_or_routes():
    ontology = HistoricalCampaignOntology.default()
    alpine = ontology.get("hannibal_alpine_crossing")
    assert alpine and alpine.type == "movement"
    assert alpine.parent_campaign == "hannibal_italy_campaign"
    assert alpine.source_references
    assert not hasattr(alpine, "coordinates")


def test_alpine_crossing_matches_specific_movement_entity():
    intent = resolve("汉尼拔翻越阿尔卑斯山路线")
    assert intent and intent.model_dump() == {
        "intent": "historical_route", "campaign_id": "hannibal_italy_campaign",
        "entity": "hannibal_alpine_crossing", "route_type": "movement",
    }


def test_italian_campaign_matches_campaign_sequence_not_alpine_movement():
    intent = resolve("汉尼拔在意大利境内的战役路线")
    assert intent and intent.entity == "hannibal_italy_campaign"
    assert intent.route_type == "campaign_sequence"


def test_cannae_matches_battle_entity():
    intent = resolve("坎尼会战路线")
    assert intent and intent.entity == "battle_of_cannae"
    assert intent.route_type == "battle"


def test_unsupported_query_has_no_ontology_intent():
    assert resolve("亚瑟王路线") is None
