from dataclasses import dataclass


@dataclass(frozen=True)
class HistoricalPlaceAlias:
    canonical_name: str
    aliases: tuple[str, ...]


# Data only: usable for any event; matching never supplies coordinates or event order.
HISTORICAL_PLACE_ALIASES = (
    HistoricalPlaceAlias("Carthago Nova", ("carthago nova", "new carthage", "cartagena")),
    HistoricalPlaceAlias("Iberus", ("iberus", "iber", "ebro")),
    HistoricalPlaceAlias("Pyrenaei", ("pyrenaei", "pyrenees")),
    HistoricalPlaceAlias("Rhodanus", ("rhodanus", "rhone", "rhône")),
    HistoricalPlaceAlias("Massalia", ("massalia", "massilia", "marseilles")),
    HistoricalPlaceAlias("Alpes", ("alpes", "alps")),
    HistoricalPlaceAlias("Taurini", ("taurini", "taurinians")),
    HistoricalPlaceAlias("Padus", ("padus", "po valley", "river po", "po, italy")),
)
