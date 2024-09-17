from pads_ml.padpolygons import PadPolygons

def main():

    filename_i = "data/STGCPadTrigger.txt"
    filename_o = "data/STGCPadTrigger.A05.txt"

    with open(filename_o, "w") as fi:
        all_pads = PadPolygons(filename_i)
        for pad in all_pads:
            if pad.wheel == 0 and pad.sector == 4:
                fi.write(pad.line)


if __name__ == "__main__":
    main()


