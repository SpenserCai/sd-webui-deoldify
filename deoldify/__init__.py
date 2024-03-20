import sys
import logging
try:
    import modules.shared
    IS_INSIDE_WEBUI = True
except ImportError:
    IS_INSIDE_WEBUI = False

if not IS_INSIDE_WEBUI:
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().setLevel(logging.INFO)

from deoldify._device import _Device

device = _Device()