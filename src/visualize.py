import argparse
import GPy
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml


def show_convergence(log_dir, axis):
    eval_dict = load_eval(log_dir)

    eval_dict = {k: np.array(v) for k, v in eval_dict.items()}
    num_evals = eval_dict["y_eval"].shape[0]
    eval_iter = np.arange(num_evals) + 1

    axis.plot(eval_iter, eval_dict["y_best"], "C3", label="Best function value")
    axis.plot(eval_iter, eval_dict["y_eval"], "ko", label="Function value")
    axis.set_xlabel("# Iteration")
    axis.set_ylabel("Function value")
    axis.legend()


def show_model(log_dir, axis):
    # Load data from files
    gp = load_gp(log_dir)
    config = load_config(log_dir)
    eval_dict = load_eval(log_dir)

    # Visualization depends on dimensionality of the input space
    if config["input_dim"] == 1:
        show_model_1d(gp, config, eval_dict, axis)
    elif config["input_dim"] == 2:
        show_model_2d(gp, config, eval_dict, axis)
    else:
        print("Only 1- and 2-dimensional models can be plotted.")
        return


def show_model_1d(gp, config, eval_dict, axis):

    lb, ub = config["lower_bound"][0], config["upper_bound"][0]
    x_plot = np.linspace(lb, ub, 500)
    mean, var = gp.predict(x_plot[:, None])
    mean = mean.squeeze()
    std = np.sqrt(var).squeeze()

    axis.plot(x_plot, mean, label="GP mean")
    axis.fill_between(
        x_plot, mean + 2 * std, mean - std * std, label="95% confidence", alpha=0.3
    )
    axis.plot(eval_dict["x_eval"], eval_dict["y_eval"], "ko", label="Evaluated points")
    axis.plot(
        eval_dict["x_best"][-1], eval_dict["y_best"][-1], "C8*", label="Best point"
    )
    axis.set_xlabel("Optimzation parameter")
    axis.set_ylabel("Function value")
    axis.legend()


def show_model_2d(gp, config, eval_dict, axis):
    raise NotImplementedError("This feature is not yet implemented.")


def load_gp(log_dir):
    try:
        model_file = os.path.join(log_dir, "model.json")
        with open(model_file, "r") as f:
            model_dict = json.load(f)
            gp = GPy.models.GPRegression._from_dict(model_dict)
    except FileNotFoundError:
        print(f"The model file could not be found in: {log_dir}")
        exit(1)
    return gp


def load_config(log_dir):
    try:
        config_file = os.path.join(log_dir, "config.yaml")
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            print(config)
    except FileNotFoundError:
        print(f"The config file could not be found in: {log_dir}")
        exit(1)
    return config


def load_eval(log_dir):
    try:
        evaluations_file = os.path.join(log_dir, "evaluations.yaml")
        with open(evaluations_file, "r") as f:
            eval_dict = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print(f"The evaluations file could not be found in: {log_dir}")
        exit(1)
    return eval_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--log_dir",
        help="Logging directory with the results to plot",
        type=str,
        default="./logs/",
    )

    args = parser.parse_args()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    fig.suptitle(f"BayesOpt4ROS \n Logging directory: {args.log_dir}")
    show_convergence(args.log_dir, axes[0])
    show_model(args.log_dir, axes[1])
    plt.tight_layout()
    plt.show()