"""Utility functions."""

import logging

from lxml import etree

logger = logging.getLogger(__name__)


def load_xml(recipe):
    """Extract the xml root from `recipe`, which may either be a path or string."""
    try:
        # Parse the given XML file:
        parser = etree.XMLParser(remove_comments=True)
        tree = etree.parse(recipe, parser)
    except (etree.XMLSyntaxError, etree.ParserError) as err:
        logger.warning("could not parse xml of recipe '%s'", recipe)
        raise err
    except IOError as err:
        logger.warning("error while reading xml: %s", err)
        raise err
    return tree.getroot()
