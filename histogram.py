import torch
import matplotlib.pyplot as plt
import numpy as np

class ConvDataCollector:
    """Accumulates weights, inputs, and outputs across all conv calls in a forward pass."""

    def __init__(self):
        self.conv_weights = []
        self.conv_inputs = []
        self.conv_outputs = []
        self.bn_outputs = []
        self.relu_outputs = []
        self.shortcut_outputs = []

    def reset(self):
        self.conv_weights.clear()
        self.conv_inputs.clear()
        self.conv_outputs.clear()
        self.bn_outputs.clear()
        self.relu_outputs.clear()
        self.shortcut_outputs.clear()

    def conv_call(self, conv, x):
        self.conv_weights.append(conv.weight.data.detach().float().cpu().flatten())
        self.conv_inputs.append(x.detach().float().cpu().flatten())
        out = conv(x)
        self.conv_outputs.append(out.detach().float().cpu().flatten())
        return out

    def bn_call(self, bn, x):
        out = bn(x)
        self.bn_outputs.append(out.detach().float().cpu().flatten())
        return out

    def relu_call(self, relu, x):
        out = relu(x)
        self.relu_outputs.append(out.detach().float().cpu().flatten())
        return out

    def shortcut_call(self, relu, x):
        out = relu(x)
        self.shortcut_outputs.append(out.detach().float().cpu().flatten())
        return out

    def plot_histograms_overlay(self, other, save_path=None, title=None, label_self='sim', label_other='float'):
        datasets_self = [
            torch.cat(self.conv_weights).numpy(),
            torch.cat(self.conv_inputs).numpy(),
            torch.cat(self.conv_outputs).numpy(),
            torch.cat(self.bn_outputs).numpy(),
            torch.cat(self.relu_outputs).numpy(),
            torch.cat(self.shortcut_outputs).numpy(),
        ]
        datasets_other = [
            torch.cat(other.conv_weights).numpy(),
            torch.cat(other.conv_inputs).numpy(),
            torch.cat(other.conv_outputs).numpy(),
            torch.cat(other.bn_outputs).numpy(),
            torch.cat(other.relu_outputs).numpy(),
            torch.cat(other.shortcut_outputs).numpy(),
        ]
        panel_labels = ['Conv Weights', 'Conv Inputs', 'Conv Outputs', 'BN Outputs', 'ReLu Outputs', 'Res Output']

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, d_self, d_other, label in zip(axes.flatten(), datasets_self, datasets_other, panel_labels):
            nz_self  = int((d_self  == 0).sum())
            nz_other = int((d_other == 0).sum())
            d_self  = d_self[d_self   != 0]
            d_other = d_other[d_other != 0]

            combined = np.concatenate([d_self, d_other])
            lo = float(np.percentile(combined, 0.5))
            hi = float(np.percentile(combined, 99.5))

            ax.hist(d_self,  bins=200, range=(lo, hi), alpha=0.5, label=f'{label_self} (zeros: {nz_self:,})',  color='steelblue')
            ax.hist(d_other, bins=200, range=(lo, hi), alpha=0.5, label=f'{label_other} (zeros: {nz_other:,})', color='darkorange')
            ax.set_xlim(lo, hi)
            ax.set_title(label)
            ax.set_xlabel('Value')
            ax.set_ylabel('Count')
            ax.legend(fontsize=8)

        if title is not None:
            fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()

    def plot_diff(self, other, save_path=None, title=None, label_self='sim', label_other='float'):
        pairs = [
            (self.conv_weights,     other.conv_weights,     'Conv Weights'),
            (self.conv_inputs,      other.conv_inputs,      'Conv Inputs'),
            (self.conv_outputs,     other.conv_outputs,     'Conv Outputs'),
            (self.bn_outputs,       other.bn_outputs,       'BN Outputs'),
            (self.relu_outputs,     other.relu_outputs,     'ReLU Outputs'),
            (self.shortcut_outputs, other.shortcut_outputs, 'Res Output'),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, (s, o, label) in zip(axes.flatten(), pairs):
            s_f = torch.cat(s).float()
            o_f = torch.cat(o).float()
            diff = (o_f - s_f).numpy()
            n_zeros = int((diff == 0).sum())
            diff = diff[diff != 0]
            if diff.size == 0 or diff.sum() == 0:
                ax.set_axis_off()
                ax.set_title(label)
                ax.text(0.5, 0.5, 'No differences', transform=ax.transAxes, ha='center', va='center')
                continue
            min_diff = float(diff.min())
            max_diff = float(diff.max())
            lo, hi = float(np.percentile(diff, 0.1)), float(np.percentile(diff, 99.9))
            ax.hist(diff, bins=200, range=(lo, hi), color='steelblue')
            ax.set_xlim(lo, hi)
            ax.set_title(label)
            ax.set_xlabel(f'{label_other} - {label_self}  [min={min_diff:.4g}, max={max_diff:.4g}]  (zeros: {n_zeros:,})')
            ax.set_ylabel('Count')

        if title is not None:
            fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()
