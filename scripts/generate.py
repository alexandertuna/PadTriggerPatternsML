import time
from pads_ml.generator import SignalGenerator

def main():
    now = time.strftime("%Y_%m_%d_%H_%M_%S")
    print(now)
    num = 100000
    gen = SignalGenerator(num, "data/STGCPadTrigger.np.A05.txt")
    # print(gen.lines.df)
    # print(gen.pads.df)
    print(gen.df)

    gen.df.to_parquet(f"tmp.parquet")

if __name__ == "__main__":
    main()


