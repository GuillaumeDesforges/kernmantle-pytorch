import argparse
import torch

def main():
    parser = argparse.ArgumentParser("Pull a model from torch.hub")
    parser.add_argument('hubdir', type=str)
    parser.add_argument('github', type=str)
    parser.add_argument('model', type=str)

    args = parser.parse_args()
    github = args.github
    model = args.model
    hubdir = args.hubdir

    # actual script
    torch.hub.set_dir(hubdir)
    torch.hub.load(github, model)

if __name__ == "__main__":
    main()
