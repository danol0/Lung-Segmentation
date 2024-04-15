import argparse


def parse_args():
    """
    Parses input command line arguments

    Returns:
        Namespace: The parsed arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--data_dir", type=str, default="Dataset")
    parser.add_argument("--lambda_DSC", type=float, default=1)
    parser.add_argument("--lambda_BCE", type=float, default=2)
    parser.add_argument("--model_path", type=str, default="model.pt")

    opt = parser.parse_known_args()[0]
    print_options(parser, opt)
    return opt


def print_options(parser, opt):
    """
    Print the options with non-defaults

    Args:
        parser (ArgumentParser): The argument parser, for defaults
        opt (Namespace): The parsed arguments

    Returns:
        str: The string representation of the options
    """

    message = ""
    message += "----------------- Options ---------------\n"
    for k, v in sorted(vars(opt).items()):
        comment = ""
        default = parser.get_default(k)
        if v != default:
            comment = "\t[default: %s]" % str(default)
        message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
    message += "----------------- End -------------------"
    print(message)
