from torch.utils.tensorboard import SummaryWriter


class LoggerScalar:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def plot_data(self, my_fantastic_logging):
        assert self.base_dir is not None, 'Please set base_dir first'
        epoch = my_fantastic_logging['epoch']
        with SummaryWriter(log_dir=self.base_dir) as writer:
            if 'loss' in my_fantastic_logging:
                writer.add_scalar(
                    tag=f"{my_fantastic_logging['type']}/Loss",
                    scalar_value=my_fantastic_logging['loss'],
                    global_step=epoch
                )
            if 'lr' in my_fantastic_logging:
                writer.add_scalar(
                    tag=f"{my_fantastic_logging['type']}/Learning Rate",
                    scalar_value=my_fantastic_logging['lr'],
                    global_step=epoch
                )
            writer.add_scalar(
                tag=f"{my_fantastic_logging['type']}/Accuracy",
                scalar_value=my_fantastic_logging['acc'],
                global_step=epoch
            )
            writer.add_scalar(
                tag=f"{my_fantastic_logging['type']}/Sensitivity (Recall)",
                scalar_value=my_fantastic_logging['SE'],
                global_step=epoch
            )
            writer.add_scalar(
                tag=f"{my_fantastic_logging['type']}/Specificity",
                scalar_value=my_fantastic_logging['SP'],
                global_step=epoch
            )
            writer.add_scalar(
                tag=f"{my_fantastic_logging['type']}/Precision",
                scalar_value=my_fantastic_logging['PC'],
                global_step=epoch
            )
            writer.add_scalar(
                tag=f"{my_fantastic_logging['type']}/F1 Score",
                scalar_value=my_fantastic_logging['F1'],
                global_step=epoch
            )
            writer.add_scalar(
                tag=f"{my_fantastic_logging['type']}/Jaccard Similarity",
                scalar_value=my_fantastic_logging['JS'],
                global_step=epoch
            )
            writer.add_scalar(
                tag=f"{my_fantastic_logging['type']}/Dice Coefficient",
                scalar_value=my_fantastic_logging['DC'],
                global_step=epoch
            )
            writer.add_scalar(
                tag=f"{my_fantastic_logging['type']}/Mean Intersection over Union",
                scalar_value=my_fantastic_logging['MIOU'],
                global_step=epoch
            )
            writer.add_scalar(
                tag=f"{my_fantastic_logging['type']}/Area Under the Curve",
                scalar_value=my_fantastic_logging['AUC'],
                global_step=epoch
            )
            writer.add_scalar(
                tag=f"{my_fantastic_logging['type']}/Average Precision",
                scalar_value=my_fantastic_logging['AP'],
                global_step=epoch
            )
