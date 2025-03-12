"""main entry point to ReX"""

from rex_xai._utils import get_device
from rex_xai.config import get_all_args
from rex_xai.database import initialise_rex_db

from rex_xai.explanation import explanation
from rex_xai.logger import logger, set_log_level
from rex_xai.config import validate_args


def main():
    """main entry point to ReX cmdline tool"""
    args = get_all_args()
    validate_args(args)
    set_log_level(args.verbosity, logger)

    device = get_device(args.gpu)

    logger.debug("running ReX with the following args:\n %s", args)

    db = None
    if args.db is not None:
        db = initialise_rex_db(args.db)

    explanation(args, device, db)
