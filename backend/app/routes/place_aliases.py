from dataclasses import dataclass
from backend.app.geography.place_registry import alias_records


@dataclass(frozen=True)
class HistoricalPlaceAlias:
    canonical_name: str
    aliases: tuple[str, ...]
    provenance: str = "audited_route_alias_registry"


# Data only: usable for any event; matching never supplies coordinates or event order.
HISTORICAL_PLACE_ALIASES = tuple(HistoricalPlaceAlias(name, values, provenance) for name, values, provenance in alias_records()) + (
    HistoricalPlaceAlias("Pyrenaei", ("pyrenaei", "pyrenees")),
    HistoricalPlaceAlias("Pyrenaei", ("pyrenaei", "pyrenees")),
    HistoricalPlaceAlias("Roma", ("roma", "rome")),
    HistoricalPlaceAlias("Taurini", ("taurini", "taurinians")),
    HistoricalPlaceAlias("Padus", ("padus", "po valley", "river po", "po, italy")),
    # Corpus-observed Roman Republican places.  These aliases establish only a
    # textual identity; Geography MCP must still resolve every coordinate.
    HistoricalPlaceAlias("Gallia", ("gallia", "gaul", "further gaul", "cisalpine gaul"), "frozen_corpus_observed"),
    HistoricalPlaceAlias("Genava", ("genava", "geneva"), "frozen_corpus_observed"),
    HistoricalPlaceAlias("Melodunum", ("melodunum",), "frozen_corpus_observed"),
    HistoricalPlaceAlias("Lutetia", ("lutetia",), "frozen_corpus_observed"),
    HistoricalPlaceAlias("Gergovia", ("gergovia",), "frozen_corpus_observed"),
    HistoricalPlaceAlias("Rubico", ("rubico", "rubicon"), "frozen_corpus_observed"),
    HistoricalPlaceAlias("Sequana", ("sequana", "seine"), "frozen_corpus_observed"),
)
