from pads_ml.pads import Pads
from pads_ml.lines import Lines
from pads_ml.traverser import Traverser

class Generator:
    def __init__(self, num: int):
        self.lines = Lines(num)
        self.pads = Pads("data/STGCPadTrigger.np.A05.txt")
        self.traverser = Traverser(self.lines.df, self.pads)
