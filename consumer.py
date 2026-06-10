"""Run Wazuh bridge to publish alerts to Kafka."""

from __future__ import annotations

import asyncio
import logging

from wazuh_bridge import WazuhBridge


async def main() -> None:
    """Start the bridge and keep it running until interrupted."""

    logging.basicConfig(level=logging.INFO)
    bridge = WazuhBridge()
    await bridge.start()
    try:
        while True:
            await asyncio.sleep(1)
    finally:
        await bridge.stop()


if __name__ == "__main__":
    asyncio.run(main())
