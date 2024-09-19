from pads_ml.generator import Generator

def main():
    num = 5
    gen = Generator(num)
    print(gen.lines.df)
    print(gen.pads.df)
    print(gen.traverser.df)

if __name__ == "__main__":
    main()


