from visualdl import LogWriter


class LoggerScalar:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def plot_data(self, my_fantastic_logging):
        assert self.base_dir is not None, 'Please set base_dir first'
        epoch = my_fantastic_logging['epoch']
        with LogWriter(logdir=self.base_dir) as writer:
            if my_fantastic_logging['loss'] is not None:
                writer.add_scalar(
                    tag=f"{my_fantastic_logging['type']}/Loss",
                    step=epoch,
                    value=my_fantastic_logging['loss']
                )
            if my_fantastic_logging['lr'] is not None:
                writer.add_scalar(
                    tag=f"{my_fantastic_logging['type']}/Learning Rate",
                    step=epoch,
                    value=my_fantastic_logging['lr']
                )
            writer.add_scalar(
                tag=f"{my_fantastic_logging['type']}/Accuracy",
                step=epoch,
                value=my_fantastic_logging['acc']
            )
            writer.add_scalar(
                tag=f"{my_fantastic_logging['type']}/Sensitivity (Recall)",
                step=epoch,
                value=my_fantastic_logging['SE']
            )
            writer.add_scalar(
                tag=f"{my_fantastic_logging['type']}/Specificity",
                step=epoch,
                value=my_fantastic_logging['SP']
            )
            writer.add_scalar(
                tag=f"{my_fantastic_logging['type']}/Precision",
                step=epoch,
                value=my_fantastic_logging['PC']
            )
            writer.add_scalar(
                tag=f"{my_fantastic_logging['type']}/F1 Score",
                step=epoch,
                value=my_fantastic_logging['F1']
            )
            writer.add_scalar(
                tag=f"{my_fantastic_logging['type']}/Jaccard Similarity",
                step=epoch,
                value=my_fantastic_logging['JS']
            )
            writer.add_scalar(
                tag=f"{my_fantastic_logging['type']}/Dice Coefficient",
                step=epoch,
                value=my_fantastic_logging['DC']
            )