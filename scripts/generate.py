import time
from pads_ml.generator import SignalGenerator, NoiseGenerator

def main():
    now = time.strftime("%Y_%m_%d_%H_%M_%S")
    print(now)
    num = 10000
    signal = SignalGenerator(num, "data/STGCPadTrigger.np.A05.txt")
    print(signal.df)

    noise = NoiseGenerator(num, "data/STGCPadTrigger.np.A05.txt")
    print(noise.df)

    signal.df.to_parquet(f"signal.parquet")
    noise.df.to_parquet(f"noise.parquet")

if __name__ == "__main__":
    main()


