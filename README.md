# PadTriggerPatternsML

The main data types used are:

- Pandas dataframe
  - This is used to a small amount of data per row
  - They are small and flexible
  - For example, the generated muon eta, phi
  - For example, the pad number of a hit on each layer (e.g. pad 100)
- Numpy array
  - This is used to store a bitmask for every pad (1739 pads) per row
  - The number of columns should always be 1739

