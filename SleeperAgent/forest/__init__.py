"""Library of simple routines."""

# from forest.witchcoven import Witch
from .witchcoven import Witch
# from forest.data import Kettle, KettleExternal
from .data import Kettle, KettleExternal
# from forest.victims import Victim
from .victims import Victim

from .options import options, options_imagenet



__all__ = ['Victim', 'Witch', 'Kettle', 'KettleExternal', 'options']
