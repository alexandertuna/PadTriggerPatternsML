import time
from pads_ml.generator import SignalGenerator, NoiseGenerator

def main():
    now = time.strftime("%Y_%m_%d_%H_%M_%S")
    num = 10000

    print("Generating signal ...")
    signal = SignalGenerator(num, "data/STGCPadTrigger.np.A05.txt")
    print(signal.df)

    print("Generating noise ...")
    noise = NoiseGenerator(num, "data/STGCPadTrigger.np.A05.txt")
    print(noise.df)

    signal.df.to_parquet(f"signal.{now}.{num}.parquet")
    noise.df.to_parquet(f"noise.{now}.{num}.parquet")

if __name__ == "__main__":
    main()


