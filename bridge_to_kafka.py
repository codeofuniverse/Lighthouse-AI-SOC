"""Run Wazuh bridge to publish alerts to Kafka."""

from __future__ import annotations

import asyncio
import logging
from wazuh_bridge import WazuhBridge

logger = logging.getLogger(__name__)


async def main() -> None:
    """Start the Wazuh bridge and stream alerts to Kafka."""
    logging.basicConfig(level=logging.INFO)

    bridge = WazuhBridge()
    await bridge.start()

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Wazuh bridge shutting down")
    finally:
        await bridge.stop()


if __name__ == "__main__":
    asyncio.run(main())
