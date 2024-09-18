from pads_ml.pads import Pads

def main():

    # i/o
    filename_i = "data/STGCPadTrigger.np.txt"
    filename_o = "data/STGCPadTrigger.np.A05.txt"

    # get input pads
    pads = Pads(filename_i, create_polygons=False)

    # require wheel A, sector 5 (numbered from 1)
    df_A05 = pads.df[
        (pads.df["wheel"] == 0xA)
        & (pads.df["sector"] == 4)
    ]

    # write to disk
    df_A05.to_csv(filename_o, sep=' ', index=False)


if __name__ == "__main__":
    main()


