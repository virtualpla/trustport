from __future__ import annotations

from collections.abc import Iterator, Sequence

from trustport.berths.consignments import BatchTensors, collate
from trustport.manifest import Consignment


def iter_batches(
    items: Sequence[Consignment],
    batch_size: int,
    drop_last: bool = False,
) -> Iterator[BatchTensors]:
    total = len(items)
    end = total - (total % batch_size) if drop_last else total
    for start in range(0, end, batch_size):
        chunk = items[start : start + batch_size]
        if not chunk:
            continue
        yield collate(chunk)
