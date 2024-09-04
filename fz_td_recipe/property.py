"""Basic components to build XML recipes."""

import fnmatch
import itertools
import logging
import re
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from functools import cached_property, lru_cache
from io import StringIO
from textwrap import indent
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from . import utils

logger = logging.getLogger(__name__)


class NotFound(Exception):
    """Exception for missing recipe component."""

    def __init__(self, cls):
        """Creates exception for missing `cls`."""
        super().__init__(f"cannot find {cls.__name__}")


class PropertyMeta(type):
    """Simple metadata class to ensure that attributes, aliases, and defaults are set."""

    def __new__(mcs, name, bases, dct):
        """Creates the new type."""
        cls = super().__new__(mcs, name, bases, dct)
        cls._local_defaults = dict(cls._attributes)

        for key, value in cls._attribute_alias.items():
            if value not in cls._attributes:
                raise TypeError(f"cannot alias non-existant {value} to {key} in {name}")

        return cls


class Property(metaclass=PropertyMeta):
    """Generic property class that should be inherited from."""

    _name: Union[str, None] = None

    _attributes: Dict[str, Any] = {}
    _attribute_alias: Dict[str, str] = {}

    _implicit = False

    def __init__(self, strict: bool = True, **kwargs):
        """Create a new property object.

        When passing `strict`, will raise an `AttributeError` if an attribute is present
        that is not defined in `_attributes` or `_attribute_alias`.
        """
        self._i = kwargs.pop("index", None)
        self._imlicit = self._implicit
        # Stores reference to local defaults.  Allows to change the _local_defaults class property
        self._local_defaults = self._local_defaults
        self._values: Dict[str, Any] = {}
        for key, value in kwargs.items():
            k = self._attribute_alias.get(key, key)
            if k in self._local_defaults:
                self._values[k] = self._converter(k)(value)
            elif strict:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{k}'")

    @classmethod
    @lru_cache
    def _converter(cls, attr):
        """Returns a type used to convert strings into the corresponding attribute."""
        kind = cls._local_defaults[attr]
        if not isinstance(kind, type):
            kind = type(kind)
        if kind is str:
            return lambda x: x
        return kind

    @classmethod
    @contextmanager
    def with_defaults(cls, defaults):
        """Temporarily set default values for attributes.

        To be used when many properties are read at the same time with different default
        values compared to the class wide defaults.  The property initialization will copy
        a reference to these local defaults.
        """
        old_values = cls._local_defaults
        cls._local_defaults = dict(old_values)
        for key, value in defaults.items():
            if key not in cls._local_defaults:
                raise TypeError(f"cannot set a default for non-existant attribute {key} in {cls}")
            cls._local_defaults[key] = cls._converter(key)(value)
        try:
            yield
        finally:
            cls._local_defaults = old_values

    def __delattr__(self, attr):
        """Deletes attribute `attr`, even if it is aliased."""
        if attr in self._values:
            del self._values[attr]

    def __getattr__(self, attr):
        """Gets attribute `attr`, taking inheritance into account."""
        try:
            return self.__dict__[attr]
        except KeyError:
            try:
                return self._values[attr]
            except KeyError:
                try:
                    default = self._local_defaults[attr]
                    if isinstance(default, type):
                        return None
                    return default
                except KeyError:
                    pass
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

    def __setattr__(self, attr, value):
        """Sets attribute `attr`, taking data types into account."""
        if attr.startswith("_"):
            super().__setattr__(attr, value)
        elif attr in self._local_defaults:
            self._values[attr] = self._converter(attr)(value)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

    def __getstate__(self):
        """State needed for pickling."""
        return (self._i, self._implicit, self._local_defaults, self._values)

    def __setstate__(self, state):
        """Restoration after pickling."""
        self._i, self._implicit, self._local_defaults, self._values = state

    def __repr__(self):
        """More legible representation."""
        return str(self)

    def __str__(self):
        """An XML like representation of the rule."""

        def vals():
            yield ""
            for key, value in self._values.items():
                yield f'{key}="{value}"'

        attrs = " ".join(vals())
        if self._implicit and len(attrs) == 0:
            return ""
        tag = self._name or type(self).__name__
        return f"<{tag}{attrs} />"

    def validate(self, _: Dict[str, List[str]] = None) -> bool:
        """Validates attribute values."""
        return True


class PropertyGroup(list):
    """A container to group settings."""

    _kind: Property
    _name: Optional[str] = None
    _alias: Optional[str] = None
    _defaults: Dict[str, Any] = {}

    def __init__(self, *args, **kwargs):
        """Creates a new property container."""
        self._defaults = kwargs.pop("defaults", {})
        super().__init__(*args, **kwargs)

    def __str__(self):
        """An XML like representation of the group."""

        def vals():
            yield ""
            for key, value in self._defaults.items():
                yield f'{key}="{value}"'

        attrs = " ".join(vals())
        inner = "\n".join(str(e) for e in self)
        if inner:
            inner = "\n" + indent(inner, "  ") + "\n"
        tag = self._name or type(self).__name__
        return f"<{tag}{attrs}>{inner}</{tag}>"

    def to_pandas(self):
        """Convert the property list to a Pandas DataFrame."""
        data = {a: [] for a in self._kind._attributes}
        for e in self:
            for k, vs in data.items():
                vs.append(getattr(e, k))
        result = pd.DataFrame()
        for k, vs in data.items():
            kind = self._kind._attributes[k]
            if kind is str or isinstance(kind, str):
                result[k] = pd.Series(vs, dtype="category")
            else:
                result[k] = vs
        return result

    @classmethod
    def load(cls, xml, strict: bool = True):
        """Load a list of properties defined by the class."""
        if isinstance(xml, str):
            xml = StringIO(xml)
            xml = utils.load_xml(xml)
        tag = cls._name or cls.__name__
        root = xml.findall(tag)
        if len(root) == 0:
            if cls._alias:
                root = xml.findall(cls._alias)
            if len(root) == 0:
                raise NotFound(cls)
        if len(root) > 1:
            raise ValueError(f"{cls._name} needs to be defind exactly once")

        valid_defaults = set(cls._kind._attributes) | set(cls._kind._attribute_alias)
        defaults = {}
        for k, v in root[0].attrib.items():
            if k in valid_defaults:
                defaults[k] = v
            elif strict:
                raise AttributeError(f"'{tag}' object cannot override attribute '{k}'")

        allowed = set([cls._kind._name])
        if hasattr(cls._kind, "_alias"):
            allowed.add(cls._kind._alias)
        items = [i for i in root[0].iter() if i.tag in allowed]
        with cls._kind.with_defaults(defaults):
            data = [
                cls._kind(strict=strict, index=i, **item.attrib) for i, item in enumerate(items)
            ]
        return cls(data, defaults=defaults)


def singleton(fct: Union[Callable, None] = None, implicit: bool = False) -> Callable:
    """Decorate a class to allow initializing with an XML element.

    Will search for the classes name in the provided element, and load

    Args:
        fct:      the callable object to decorate
        implicit: will create the non-existant decorated element with default parameters
    """

    def _deco(cls):
        def _loader(*args, **kwargs):
            strict = kwargs.pop("strict", True)
            tag = cls._name or cls.__name__
            if (len(args) > 0) == (len(kwargs) > 0):
                raise TypeError("check call signature")
            if args:
                if len(args) > 1:
                    raise TypeError("check call signature")
                items = args[0].findall(tag)
                if len(items) == 0:
                    if implicit:
                        obj = cls()
                        obj._implicit = True
                        return obj
                    raise NotFound(cls)
                if len(items) > 1:
                    raise ValueError(f"singleton expected for {cls}")
                return cls(strict=strict, **items[0].attrib)
            return cls(strict=strict, **kwargs)

        cls.load = _loader
        return cls

    if fct:
        return _deco(fct)
    return _deco


_DIRECTIONAL_ATTR = re.compile(r"^(from|to)([A-Z]\w*)$")


class PathwayProperty(Property):
    """Property that uses pattern matching to define connectivity."""

    @classmethod
    def columns(cls):
        """Column names for attributes used in the pathway.

        The attributes of the property that indicate a source / target invocation via the
        prefixes ``from`` and ``to``.

        >>> class rule(PathwayProperty):
        ...     _attributes = {'toMType': '*', 'fromEType': '*', 'something': str}
        >>> r = rule(toMType='B*', fromEType='*')
        >>> r.columns()
        ['fromEType', 'toMType']
        """
        return sorted(k for k in cls._attributes if _DIRECTIONAL_ATTR.match(k))

    def expand(self, values: Dict[str, List[str]]):
        """Expand the pathway for partial attribute wildcards.

        Expand the property given the possible matching attribute values in `values`,
        leaving pure wildcards intact.
        Returns an iterator over the property with expanded values.

        >>> class rule(PathwayProperty):
        ...     _attributes = {'toMType': '*', 'fromEType': '*', 'something': str}
        >>> r = rule(toMType='B*', fromEType='*')
        >>> list(r.expand({'toMType': ['Foo', 'Bar', 'Baz']}))
        [<rule toMType="Bar" fromEType="*" />, <rule toMType="Baz" fromEType="*" />]
        >>> r = rule(toMType='*', fromEType='*')
        >>> list(r.expand({'toMType': ['Foo', 'Bar', 'Baz']}))
        [<rule toMType="*" fromEType="*" />]
        """
        expansions = []
        for k, vs in values.items():
            val = getattr(self, k)
            if val == "*" or val in vs:
                expansions.append([(k, val)])
            elif "*" in val:
                expansions.append([(k, v) for v in fnmatch.filter(vs, val)])
            else:
                # recipe value is not in the passed list for this attribute
                return
        for cols in itertools.product(*expansions):
            replica = deepcopy(self)
            for k, v in cols:
                setattr(replica, k, v)
            yield replica

    def __call__(
        self, values: Dict[str, List[str]], keep_generic=False
    ) -> Iterator[Tuple[Dict[str, str], Any]]:
        """Expand the pathway for all possible attribute values.

        Expand the property given the possible matching attribute values in `values`.
        Returns an iterator over dictionaries with the attribute values matched and the
        property itself.

        >>> class rule(PathwayProperty):
        ...     _attributes = {'toMType': '*', 'fromEType': '*', 'something': str}
        >>> r = rule(toMType='B*', fromEType='*')
        >>> list(r({'toMType': ['Foo', 'Bar']}))
        [({'toMType': 'Bar'}, <rule toMType="B*" fromEType="*" />)]
        """
        expansions = [
            [(name, v) for v in fnmatch.filter(values[name], getattr(self, name))]
            for name in self.columns()
            if name in values
        ]
        for cols in itertools.product(*expansions):
            yield (dict(cols), self)


class PathwayPropertyGroup(PropertyGroup):
    """Property group for `PathwayProperty`."""

    def validate(self, values: Dict[str, List[str]]) -> bool:
        """Basic validation.

        Checks that all rules of the group are correct and all values passed are covered.

        >>> from fz_td_recipe.parts.touch_connections import ConnectionRules
        >>> rules = ConnectionRules.load('''
        ... <blueColumn>
        ...   <ConnectionRules>
        ...     <mTypeRule fromMType="*" toMType="B*"
        ...       bouton_reduction_factor="0" pMu_A="1.0" p_A="1.0" />
        ...     <mTypeRule fromMType="*" toMType="Bar"
        ...       bouton_reduction_factor="0" pMu_A="1.0" p_A="1.0" />
        ...     <mTypeRule fromMType="*" toMType="Bar"
        ...       toRegion="Spam" bouton_reduction_factor="0" pMu_A="1.0" p_A="1.0" />
        ...   </ConnectionRules>
        ... </blueColumn>
        ... ''')
        >>> values = {
        ...     'toMType': ['Bar', 'Baz'],
        ...     'toRegion': ['Ham', 'Eggs', 'Spam']
        ... }
        >>> rules.validate(values)
        True
        >>> values = {
        ...     'toMType': ['Foo', 'Bar', 'Baz'],
        ...     'toRegion': ['Ham', 'Eggs', 'Spam']
        ... }
        >>> rules.validate(values)  # Foo is not covered by the rules!
        False
        """

        def _check(name, covered):
            uncovered = set(values[name]) - covered
            if len(uncovered):
                logger.warning("The following %s are not covered: %s", name, ", ".join(uncovered))
                return False
            return True

        covered = defaultdict(set)
        valid = True
        for r in self:
            if not r.validate():
                valid = False
            expansion = list(r(values))
            if len(expansion) == 0:
                logger.warning("The following rule does not match anything: %s", r)
                valid = False
            for keys, _ in expansion:
                for name, key in keys.items():
                    covered[name].add(key)
        return valid and all(_check(name, covered[name]) for name in covered)

    def __delitem__(self, key, /):
        """Delete self[key]."""
        self.__dict__.pop("required", None)
        super().__delitem__(key)

    def __getitem__(self, key, /):
        """Returns the property corresponding to key."""
        self.__dict__.pop("required", None)
        return super().__getitem__(key)

    def __iadd__(self, value, /):
        """Implement self+=value."""
        self.__dict__.pop("required", None)
        return super().__iadd__(value)

    def __setitem__(self, key, value, /):
        """Set self[key] to value."""
        self.__dict__.pop("required", None)
        super().__setitem__(key, value)

    def append(self, rule, /):
        """Append rule to the end of the group."""
        self.__dict__.pop("required", None)
        super().append(rule)

    def clear(self, /):
        """Remove all items from group."""
        self.__dict__.pop("required", None)
        super().clear()

    def extend(self, iterable, /):
        """Extend group by appending rules from the iterable."""
        self.__dict__.pop("required", None)
        super().extend(iterable)

    def insert(self, index, rule, /):
        """Insert rule before index."""
        self.__dict__.pop("required", None)
        super().insert(index, rule)

    def pop(self, index=-1, /):
        """Remove and return rule at index (default last).

        Raises ValueError if the value is not present.
        """
        self.__dict__.pop("required", None)
        return super().pop(index)

    def remove(self, value, /):
        """Remove first occurrence of value.

        Raises ValueError if the value is not present.
        """
        self.__dict__.pop("required", None)
        return super().remove(value)

    @cached_property
    def columns(self) -> List[str]:
        """The columns of all properties in the list."""
        return self._kind.columns()

    @cached_property
    def required(self) -> List[str]:
        """The required attributes for this property group.

        Returns a list of attributes whose values are required for properly selecting
        rules, i.e., all attributes used by the enclosed properties, excluding any that
        are consistently set to ``"*"``.

        For example:

        >>> from fz_td_recipe.parts.touch_connections import ConnectionRules
        >>> ConnectionRules.load('''
        ... <blueColumn>
        ...   <ConnectionRules>
        ...     <mTypeRule fromMType="*" toMType="B*" />
        ...     <mTypeRule fromMType="*" toMType="Bar" />
        ...     <mTypeRule fromMType="*" toMType="Bar" toRegion="Spam" />
        ...   </ConnectionRules>
        ... </blueColumn>
        ... ''').required
        ['toMType', 'toRegion']
        """
        cols = []
        for col in self._kind.columns():
            for e in self:
                if getattr(e, col, "*") != "*":
                    cols.append(col)
                    break
        return cols

    def pathway_index(
        self,
        key: Dict[str, str],
        values: Dict[str, List[str]],
    ) -> int:
        """Create a pathway index from `key`.

        Generate an integer pathway identifier for the provided `key`, and reference
        `values` for all attributes given by :py:attr:`~required`.

        >>> from fz_td_recipe.parts.touch_connections import ConnectionRules
        >>> rules = ConnectionRules.load('''
        ... <blueColumn>
        ...   <ConnectionRules>
        ...     <mTypeRule fromMType="*" toMType="B*" />
        ...     <mTypeRule fromMType="*" toMType="Bar" />
        ...     <mTypeRule fromMType="*" toMType="Bar" toRegion="Spam" />
        ...   </ConnectionRules>
        ... </blueColumn>
        ... ''')
        >>> values = {
        ...     'toMType': ['Bar', 'Baz'],
        ...     'toRegion': ['Ham', 'Eggs', 'Spam']
        ... }
        >>> rules.pathway_index(
        ...     key={'toMType': 'Bar', 'toRegion': 'Spam'},
        ...     values=values
        ... )
        4
        >>> rules.pathway_index(
        ...     key={'toMType': 'Baz', 'toRegion': 'Spam'},
        ...     values=values
        ... )
        5
        >>> rules.pathway_index(
        ...     key={'toMType': 'Baz', 'toRegion': 'Eggs'},
        ...     values=values
        ... )
        3
        """
        factor = 1
        result = 0
        for attr in self.required:
            result += values[attr].index(key[attr]) * factor
            factor *= len(values[attr])
        return result

    def pathway_values(
        self,
        key: int,
        values: Dict[str, List[str]],
    ) -> Dict[str, str]:
        """Resolve the attribute values from pathway index `key`.

        Generate a dictionaly of attributes and values that need to be set given an
        integer pathway `key` and reference `values` for all attributes given by
        :py:attr:`~required`.

        >>> from fz_td_recipe.parts.touch_connections import ConnectionRules
        >>> rules = ConnectionRules.load('''
        ... <blueColumn>
        ...   <ConnectionRules>
        ...     <mTypeRule fromMType="*" toMType="B*" />
        ...     <mTypeRule fromMType="*" toMType="Bar" />
        ...     <mTypeRule fromMType="*" toMType="Bar" toRegion="Spam" />
        ...   </ConnectionRules>
        ... </blueColumn>
        ... ''')
        >>> values = {
        ...     'toMType': ['Bar', 'Baz'],
        ...     'toRegion': ['Ham', 'Eggs', 'Spam']
        ... }
        >>> rules.pathway_values(
        ...     key=4,
        ...     values=values
        ... )
        {'toMType': 'Bar', 'toRegion': 'Spam'}
        """
        result = {}
        for attr in self.required:
            idx = key % len(values[attr])
            key //= len(values[attr])
            result[attr] = values[attr][idx]
        return result

    # pylint: disable=arguments-differ
    def to_pandas(self, values: Dict[str, List[str]]) -> pd.DataFrame:
        """Convert the property list to as a Pandas DataFrame.

        Any directional attribute will be converted to an integer representation,
        corresponding to the index within the map in `values`.  The integer columns will
        have a `_i` suffix. Any generic wildcard in the form of just `*` will be assigned
        a value of -1.

        The passed `values` will also be used to expand any *partial* wildcards in the
        properties.  The corresponding property will be repeated for all matching values.

        Columns that consist of purely -1 will be dropped.

        >>> from fz_td_recipe.parts.touch_connections import ConnectionRules
        >>> rules = ConnectionRules.load('''
        ... <blueColumn>
        ...   <ConnectionRules>
        ...     <mTypeRule fromMType="*" toMType="B*" />
        ...     <mTypeRule fromMType="*" toMType="Bar" />
        ...     <mTypeRule fromMType="*" toMType="Bar" toRegion="Spam" />
        ...   </ConnectionRules>
        ... </blueColumn>
        ... ''')
        >>> values = {
        ...     'toMType': ['Bar', 'Baz'],
        ...     'toRegion': ['Ham', 'Eggs', 'Spam']
        ... }
        >>> df = rules.to_pandas(values)
        >>> len(df)
        4
        >>> list(df["toMType_i"])
        [0, 1, 0, 0]
        >>> list(df["toRegion_i"])
        [-1, -1, -1, 2]
        >>> "fromRegion_i" in df.columns
        False
        """
        exclude = set(self.columns) - set(self.required)
        data = {a: [] for a in self._kind._attributes if a not in exclude}
        for rule in self:
            for r in rule.expand(values):
                for k, vs in data.items():
                    vs.append(getattr(r, k))
        result = pd.DataFrame()
        for k, vs in values.items():
            if k in data:
                mapping = {k: n for n, k in enumerate(vs)} | {"*": -1}
                # result[k] = pd.Series(data[k], dtype="string")
                result[f"{k}_i"] = (
                    pd.Series(data.pop(k), dtype="string").map(mapping).astype("int16")
                )
        for k, vs in data.items():
            kind = self._kind._attributes[k]
            if kind is str or isinstance(kind, str):
                result[k] = pd.Series(vs, dtype="category")
            else:
                result[k] = vs
        return result

    def to_matrix(self, values: Dict[str, List[str]]) -> np.array:
        """Construct a flattened connection rule matrix.

        Requires being passed a full set of allowed parameter values passed
        as argument `values`.  Will raise a :class:`KeyError` if any
        required attributes listed by :py:attr:`~required` are not
        present.

        Returns a :class:`numpy.array` filled with indices into the
        :class:`PathwayPropertyGroup` for each possible pathway.
        Values greater than the last index of the
        :class:`~PathwayPropertyGroup` signify no match with any rule.

        >>> from fz_td_recipe.parts.touch_connections import ConnectionRules
        >>> rules = ConnectionRules.load('''
        ... <blueColumn>
        ...   <ConnectionRules>
        ...     <mTypeRule fromMType="*" toMType="B*" />
        ...     <mTypeRule fromMType="*" toMType="Bar" />
        ...     <mTypeRule fromMType="*" toMType="Bar" toRegion="Spam" />
        ...   </ConnectionRules>
        ... </blueColumn>
        ... ''')
        >>> values = {
        ...     'toMType': ['Bar', 'Baz'],
        ...     'toRegion': ['Ham', 'Eggs', 'Spam']
        ... }
        >>> m = rules.to_matrix(values)
        >>> m
        array([1, 0, 1, 0, 2, 0], dtype=uint32)
        >>> # The last rule matches Bar/Spam
        >>> idx = m[rules.pathway_index({'toMType': 'Bar', 'toRegion': 'Spam'}, values)]
        >>> idx
        np.uint32(2)
        >>> rules[idx]
        <rule fromMType="*" toMType="Bar" toRegion="Spam" />
        """
        if set(self.required) - set(values):
            raise KeyError(f"Missing keys: {', '.join(set(self.required) - set(values))}")
        reverted = {}
        for name, allowed in values.items():
            if name in self.required:
                reverted[name] = {v: i for i, v in enumerate(allowed)}
        concrete = np.full(
            fill_value=len(self),
            shape=[len(values[r]) for r in self.required],
            dtype="uint32",
        )
        default_key = [np.arange(len(values[name])) for name in self.required]
        for rule in self:
            key = default_key[:]
            for n, name in enumerate(self.required):
                attr = getattr(rule, name)
                if attr != "*":
                    filtered = fnmatch.filter(values[name], attr)
                    indices = [reverted[name][i] for i in filtered]
                    key[n] = indices
            indices = np.ix_(*key)
            if hasattr(rule, "overrides"):
                # Provide the opportunity to selectively override previous
                # rules
                active_rules = concrete[indices]
                for old_idx in np.unique(active_rules):
                    if old_idx >= len(self) or rule.overrides(self[old_idx]):
                        active_rules = np.where(active_rules == old_idx, rule._i, active_rules)
                concrete[indices] = active_rules
            else:
                concrete[indices] = rule._i
        return concrete.flatten(order="F")


class MTypeValidator:
    """Mixin to validate rules depending on morpholgy types.

    Will test both that all rules will match some morphology types, and
    that all morphology types passed to ``validate`` will be covered by at
    least one rule.
    If the inheriting class has an attribute ``_mtype_coverage`` set to
    ``False``, the latter part of the validation will be skipped.
    """

    def validate(self, values: Dict[str, List[str]]) -> bool:
        """Checks that all MType values are covered, and rules match MTypes."""

        def _check(mtypes, covered, kind):
            uncovered = set(mtypes) - covered
            if len(uncovered):
                logger.warning(
                    "The following %s MTypes are not covered: %s",
                    kind,
                    ", ".join(uncovered),
                )
                return False
            return True

        src_mtypes = values["fromMType"]
        dst_mtypes = values["toMType"]

        covered_src = set()
        covered_dst = set()
        valid = True
        for r in iter(self):
            values = list(r(src_mtypes, dst_mtypes))
            if len(values) == 0:
                logger.warning("The following rule does not match anything: %s", r)
                valid = False
            for src, dst, *_ in values:
                covered_src.add(src)
                covered_dst.add(dst)
        if not getattr(self, "_mtype_coverage", True):
            return valid
        return all(
            [
                valid,
                _check(src_mtypes, covered_src, "source"),
                _check(dst_mtypes, covered_dst, "target"),
            ]
        )
