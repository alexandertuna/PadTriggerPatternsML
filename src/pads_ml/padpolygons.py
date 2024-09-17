from dataclasses import dataclass
from shapely.geometry import Polygon


def is_valid(line: str) -> bool:
    return line.startswith("Pad")


class PadPolygons:
    """
    PadPolygons is a container class for PadPolygons
    """
    def __init__(self, filename: str) -> None:
        self.filename = filename
        with open(self.filename) as fi:
            self.pads = tuple(
                PadPolygon(*PadLine(line).data) for line in fi if is_valid(line)
            )

    def __iter__(self):
        return iter(self.pads)


@dataclass()
class PadPolygon:
    """
    PadPolygon is a dataclass for holding attributes which describe a pad.
    """
    line: str
    id: int
    wheel: int
    sector: int
    pfeb: int
    tds_channel: int
    pad_channel: int
    x0: float
    y0: float
    z0: float
    x1: float
    y1: float
    z1: float
    x2: float
    y2: float
    z2: float
    x3: float
    y3: float
    z3: float

    def __post_init__(self):
        self.polygon = Polygon([
            (self.x0, self.y0),
            (self.x1, self.y1),
            (self.x3, self.y3),
            (self.x2, self.y2),
        ])

        # All the zs should be very similar
        zs = (self.z0, self.z1, self.z2, self.z3)
        if abs(min(zs) - max(zs)) > 10:
            raise Exception(f"Weird z values: {zs}")

        self.z = sum(zs) / len(zs)


class PadLine:
    """
    PadLine takes a line of text as input, and converts it to a list of attributes
      which describe a pad.
    """
    def __init__(self, line: str) -> None:
        if not is_valid(line):
            raise Exception("Cannot unpack invalid PadLine")
        self.line = line
        self.data = self.unpack()

    def is_valid(self) -> bool:
        return self.line.startswith("Pad")

    def unpack(self) -> tuple:
        (key, id, whl, sec, pfeb, tdsch, padch,
         x0, y0, z0, x1, y1, z1,
         x2, y2, z2, x3, y3, z3) = self.line.split()

        return (
            self.line,
            int(id, base=16),
            0 if whl=="A" else 1,
            int(sec),
            int(pfeb),
            int(tdsch),
            int(padch),
            float(x0),
            float(y0),
            float(z0),
            float(x1),
            float(y1),
            float(z1),
            float(x2),
            float(y2),
            float(z2),
            float(x3),
            float(y3),
            float(z3),
        )

