"""Functionality to convert XML recipes to JSON and to expand to Pandas."""

import fnmatch
import itertools
import json
import re

import libsonata
import numpy as np
import pandas as pd

from .legacy import XMLRecipe
from .parts.seeds import Seeds
from .parts.synapse_properties import SynapseProperties, SynapseRules
from .parts.touch_rules import TouchRule
from .property import Property, PropertyGroup

_CAMEL_CASE = re.compile(r"(?<=[^A-Z_])([A-Z][a-z0-9]*)")
_DIRECTIONAL_PREFIXES = ["src", "dst", "afferent", "efferent"]
_EDGE_DIRECTIONS = ["afferent", "efferent"]
_REPLACEMENTS = [
    (re.compile(r"\bfrom_"), "src_"),
    (re.compile(r"\bto_"), "dst_"),
    (re.compile(r"\bm_type\b"), "mtype"),
    (re.compile(r"_sclass\b"), "_synapse_class"),
    (re.compile(r"synapses_"), "synapse_"),
    (re.compile(r"^src_branch_type$"), "efferent_section_type"),
    (re.compile(r"^dst_branch_type$"), "afferent_section_type"),
    (re.compile(r"^seeds$"), "seed"),
    (re.compile(r"^type$"), "class"),
    (re.compile(r"^id$"), "class"),
    (re.compile(r"^p_a$"), "p_A"),
    (re.compile(r"^p_mu_a$"), "pMu_A"),
    (re.compile(r"^gsyn((?=_)|\b)"), "conductance"),
    (re.compile(r"_srsf((?=_)|\b)"), "_scale_factor"),
    (re.compile(r"^dtc((?=_)|\b)"), "decay_time"),
    (re.compile(r"^d((?=_)|\b)"), "depression_time"),
    (re.compile(r"^f((?=_)|\b)"), "facilitation_time"),
    (re.compile(r"^nrrp((?=_)|\b)"), "n_rrp_vesicles"),
    (re.compile(r"^u((?=_)|\b)"), "u_syn"),
]
_SECTION_TYPES = ["soma", "axon", "apical", "basal"]
_SPECIAL_REPLACEMENTS = {
    "afferent_section_type": {"dendrite": ["apical", "basal"]},
    "efferent_section_type": {"dendrite": ["apical", "basal"]},
}


def _snake_case(s):
    """Turn the given string into snake case."""
    return re.sub(_CAMEL_CASE, r"_\1", s).lower()


def _rename(s):
    """Transform the given string into a functionalizer compatible one."""
    snaked = _snake_case(s)
    for expr, repl in _REPLACEMENTS:
        snaked = re.sub(expr, repl, snaked)
    return snaked


def _is_directional(key):
    """Determine if a column name indicates a direction."""
    first = key.split("_", 1)[0]
    return first in _DIRECTIONAL_PREFIXES


def _to_integer_column(key):
    """Add `_i` suffix if appropriate - if the key is from a node population."""
    first = key.split("_", 1)[0]
    if first in _EDGE_DIRECTIONS:
        return key
    elif first in _DIRECTIONAL_PREFIXES and not key.endswith("_i"):
        return f"{key}_i"
    return key


def _all_specific_directional_keys(group):
    """Extracts all directional attributes that are not defaulted."""
    keys = set()
    if isinstance(group, (list, tuple)):
        _all_keys = set()
        for rule in group:
            _all_keys.update([k for k in rule.keys() if _is_directional(k)])
        for key in _all_keys:
            all_wildcards = all(r.get(key, "*") == "*" for r in group)
            if not all_wildcards:
                keys.add(key)
    else:
        _all_keys = [c for c in group.columns if _is_directional(c)]
        for key in _all_keys:
            values = set(group[key].unique())
            if values != set([-1]):
                keys.add(key)
    return sorted(keys)


class _Pandifier:
    def __init__(self, circuit_config, nodes):
        self.circuit_config = circuit_config
        cfg = libsonata.CircuitConfig.from_file(self.circuit_config)
        (self.src, self.dst) = [self._get_population(cfg, pop) for pop in nodes]

    @staticmethod
    def _get_population(cfg, pop):
        """Get a node population from a circuit configuration."""
        if not pop:
            if len(cfg.node_populations) != 1:
                raise ValueError("cannot infer node population")
            pop = next(iter(cfg.node_populations))
        return cfg.node_population(pop)

    def convert(self, group):
        """Converts an XML recipe group to a Pandas DataFrame."""
        values = {}
        for c in group.required:
            nodes, attr = _rename(c).split("_", 1)
            values[c] = getattr(self, nodes).enumeration_values(attr)
        return group.to_pandas(values).rename(columns=_rename)

    def get_values(self, columns):
        """Get all possible enumeration values for columns."""
        values = {}
        for col in columns:
            pop, attr = col.split("_", 1)
            if pop in _EDGE_DIRECTIONS:
                if attr != "section_type":
                    raise KeyError("only [ae]fferent_section_type allowed")
                values[col] = _SECTION_TYPES
            else:
                if attr.endswith("_i"):
                    attr = attr[:-2]
                values[col] = getattr(self, pop).enumeration_values(attr)
        return values

    def framify(self, group):
        """Convert group into a Pandas DataFrame.

        Expands any partial wildcards, i.e., anything but `*`. Columns relating to the
        source and target node population will be tanslated to the corresponding values of
        their `@library` enumeration.  Wildcards `*` will have the value `-1` assigned.
        """
        columns = _all_specific_directional_keys(group)
        values = self.get_values(columns)
        expanded_group = []
        for rule in group:
            expanded = [[] for _ in columns]
            for n, col in enumerate(columns):
                val = rule.get(col, "*")
                special_treatment = _SPECIAL_REPLACEMENTS.get(col, {})
                if val != "*" and "*" in val:
                    expanded[n] = list(fnmatch.filter(values[col], val))
                elif replacements := special_treatment.get(val):
                    expanded[n] = replacements
                else:
                    expanded[n] = [val]
            expanded_group.extend(
                (rule | dict(zip(columns, vals)) for vals in itertools.product(*expanded))
            )
        df = pd.DataFrame(expanded_group)
        filterable = [c for c in df.columns if _is_directional(c)]
        df = (
            df.fillna({c: "*" for c in filterable})
            .replace(
                {c: {v: i for i, v in enumerate(values.get(c, {}))} | {"*": -1} for c in filterable}
            )
            .rename(columns={c: _to_integer_column(c) for c in filterable})
        )
        for col in (_to_integer_column(c) for c in filterable):
            if df.dtypes[col] == "O":
                values = ", ".join([str(v) for v in df[col].unique()])
                raise ValueError(f"Column '{col}' has untranslated values: {values}")
        for col in df.columns:
            if df.dtypes[col] == "O":
                df[col] = df[col].astype("category")
            elif df.dtypes[col].kind == "i":
                df[col] = df[col].astype("int16")
        return df

    def matrify(self, group):
        """Construct a rule assignment matrix.

        Takes a list of rules, determines which columns relating to the source and target
        population are of essence (contain more than just general wildcards `*`).  The
        enumeration value count in the corresponding node population `@libray` enumeration
        then is used to construct a matrix, where the entries in the matrix set depending
        on the numerical values of the columns, with wildcards expanded.

        Returns a tuple containing the list of columns used (ordered corresponding to the
        matrix dimensions) and the matrix itself.
        """
        columns = _all_specific_directional_keys(group)
        values = self.get_values(columns)

        concrete = np.full(
            fill_value=len(group),
            shape=[len(values[c]) for c in columns],
            dtype="uint32",
        )

        default_key = [np.arange(len(values[c])) for c in columns]

        if isinstance(group, (list, tuple)):
            reverted = {}
            for k, vs in values.items():
                reverted[k] = {v: i for i, v in enumerate(vs)}

            for idx, rule in enumerate(group):
                key = default_key[:]
                for n, name in enumerate(columns):
                    attr = rule.get(name, "*")
                    special_treatment = _SPECIAL_REPLACEMENTS.get(name, {})
                    if attr != "*":
                        if attr in special_treatment:
                            filtered = special_treatment[attr]
                        else:
                            filtered = fnmatch.filter(values[name], attr)
                        indices = [reverted[name][i] for i in filtered]
                        key[n] = indices
                indices = np.ix_(*key)
                concrete[indices] = idx
        else:
            for idx, rule in enumerate(group.itertuples()):
                key = default_key[:]
                for n, name in enumerate(columns):
                    attr = getattr(rule, name, -1)
                    if attr != -1:
                        key[n] = [attr]
                indices = np.ix_(*key)
                concrete[indices] = idx

        return [_to_integer_column(c) for c in columns], concrete


def _get_property_attributes(o):
    """Return attributes value for an XML rule, taking all defaults into account."""
    properties = dict(o._values)
    for k, v in o._local_defaults.items():
        if o._attributes[k] != v:
            properties[k] = v
    return properties


class _RecipeEncoder(json.JSONEncoder):
    def __init__(self, output_dir=None, circuit_config=None, nodes=None):
        super().__init__(indent=2)

        self.output_dir = output_dir
        self.pandifier = None
        if circuit_config:
            self.pandifier = _Pandifier(circuit_config, nodes)

    def default(self, o):
        if isinstance(o, Seeds):
            return o.synapseSeed
        if isinstance(o, Property):
            properties = _get_property_attributes(o)
            result = {_rename(k): v for k, v in properties.items()}
            if isinstance(o, TouchRule):
                result.pop("dst_layer", None)
                result.pop("src_layer", None)
            return result
        if isinstance(o, PropertyGroup):
            if isinstance(o, SynapseRules) and self.pandifier:
                pth = (self.output_dir / _rename(o.__class__.__name__)).with_suffix(".parquet")
                self.pandifier.convert(o).to_parquet(pth)
                return str(pth.relative_to(self.output_dir))
            if hasattr(o._kind, "overrides"):
                o = sorted(o)
            return [self.default(p) for p in o]
        if isinstance(o, (XMLRecipe, SynapseProperties)):
            result = {}
            for name in o.__annotations__:
                if hasattr(o, name) and getattr(o, name) is not None:
                    result[_rename(name)] = self.default(getattr(o, name))
            return result
        raise ValueError(f"Unknown object in recipe: {o}")  # pragma: no cover
