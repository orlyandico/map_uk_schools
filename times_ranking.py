"""
Times / Sunday Times Parent Power league-table rank lookup.

Loads the extracted Parent Power CSVs (see the "Times Parent Power ranks" section
of README.md) and matches a school from the DfE-based pipelines to its Times rank
by name + town.

Matching is two-stage:
  1. exact match on a normalised school name (town disambiguates ties);
  2. a token-subset fallback that also requires town agreement, to catch the
     Times/DfE naming differences ("King's College School" vs
     "King's College School, Wimbledon", "Bancroft's School" vs "Bancroft's").

Only the rank is exposed; the A-level / GCSE / SATs columns are already shown in
the maps from the DfE data.
"""

import csv
import re
import unicodedata

SECONDARY_CSV = "best_schools_2025_combined_secondary.csv"
PRIMARY_CSV = "best_schools_2025_top500_state_primary.csv"


def _norm(s):
    """Normalise a school name for matching."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("–", "-").replace("—", "-")
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\b(the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _towns_agree(a, b):
    if not a or not b:
        return False
    return a in b or b in a


class TimesRanking:
    """Rank lookup for one Parent Power table."""

    def __init__(self, csv_path):
        self.entries = []  # list of dicts: rank, name, town, tokens
        self._exact = {}   # normalised name -> list of entry indices
        try:
            with open(csv_path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    name = row.get("School", "")
                    n = _norm(name)
                    if not n:
                        continue
                    entry = {
                        "rank": (row.get("Rank") or "").strip(),
                        "name": n,
                        "town": _norm(row.get("Location", "")),
                        "tokens": set(n.split()),
                    }
                    self._exact.setdefault(n, []).append(len(self.entries))
                    self.entries.append(entry)
        except FileNotFoundError:
            pass

    @property
    def loaded(self):
        return bool(self.entries)

    def lookup(self, name, town=""):
        """Return the Times rank string (e.g. "23", "6=") or None."""
        n = _norm(name)
        if not n:
            return None
        t = _norm(town)

        cands = self._exact.get(n)
        if cands:
            if len(cands) == 1:
                return self.entries[cands[0]]["rank"]
            for i in cands:  # disambiguate ties by town
                if _towns_agree(self.entries[i]["town"], t):
                    return self.entries[i]["rank"]
            return self.entries[cands[0]]["rank"]

        # token-subset fallback, gated on town agreement to avoid false positives
        toks = set(n.split())
        for e in self.entries:
            if not _towns_agree(e["town"], t):
                continue
            if e["tokens"] <= toks or toks <= e["tokens"]:
                return e["rank"]
        return None


def load_secondary(path=SECONDARY_CSV):
    return TimesRanking(path)


def load_primary(path=PRIMARY_CSV):
    return TimesRanking(path)
