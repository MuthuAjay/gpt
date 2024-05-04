import torch
import matplotlib.pyplot as plt
from typing import List


class Plot:
    def plot_activation_distribution(self,
                                     layers: List[object],
                                     act_fn: object) -> None:
        """Plot the distribution of activations for layers using a specific activation function.

        Args:
            layers (List[object]): List of layer instances in the neural network.
            act_fn (object): Activation function instance to analyze.

        Returns:
            None
        """
        # Visualize the histograms
        plt.figure(figsize=(20, 4))
        legends = []
        for i, layer in enumerate(layers[:-1]):
            if isinstance(layer, act_fn):
                t = layer.out
                print('layer %d (%10s): mean %+.2f, std %2.f, saturated: %.2f%%' % (i,
                                                                                    layer.__class__.__name__,
                                                                                    t.mean(),
                                                                                    t.std(),
                                                                                    (
                                                                                        t.abs() > 0.97).float().mean()
                                                                                    * 100)
                      )
                hy, hx = torch.histogram(t, density=True)
                plt.plot(hx[:-1].detach(), hy.detach())
                legends.append(f'layer {i} {layer.__class__.__name__}')
        plt.legend(legends)
        plt.title('activation distribution')

    def plot_gradient_distribution(self,
                                   layers: List[object],
                                   act_fn: object) -> None:
        """Plot the distribution of gradients for layers using a specific activation function.

        Args:
            layers (List[object]): List of layer instances in the neural network.
            act_fn (object): Activation function instance to analyze.

        Returns:
            None
        """
        # Visualize the histograms
        plt.figure(figsize=(20, 4))
        legends = []
        for i, layer in enumerate(layers[:-1]):
            if isinstance(layer, act_fn):
                t = layer.out.grad
                print('layer %d (%10s): mean %+f, std %e' % (i,
                                                             layer.__class__.__name__,
                                                             t.mean(),
                                                             t.std(),
                                                             ))
                hy, hx = torch.histogram(t, density=True)
                plt.plot(hx[:-1].detach(), hy.detach())
                legends.append(f'layer {i} {layer.__class__.__name__}')
        plt.legend(legends)
        plt.title('gradient distribution')

    def weights_gradient_distribution(self,
                                      parameters: List[torch.Tensor]) -> None:
        """Plot the distribution of gradients for weights.

        Args:
            parameters (List[torch.Tensor]): List of weight tensors.

        Returns:
            None
        """
        plt.figure(figsize=(20, 4))
        legends = []
        for i, p in enumerate(parameters):
            t = p.grad
            if p.ndim == 2:
                print('weight %10s } mean %+f | std %e | grad:data ratio %e' % (
                    tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
                hy, hx = torch.histogram(t, density=True)
                plt.plot(hx[:-1].detach(), hy.detach())
                legends.append(f"{i} {tuple(p.shape)}")
        plt.legend(legends)
        plt.title('weights gradient distribution')

    def plot_updates_to_gradients_ratio(self,
                                        parameters: List[torch.Tensor],
                                        ud: List) -> None:
        """Plot the ratio of gradient updates to data.

        Args:
            parameters (List[torch.Tensor]): List of weight tensors.
            ud (List): List of updates to gradients ratio.

        Returns:
            None
        """
        plt.figure(figsize=(20, 4))
        legends = []

        for i, p in enumerate(parameters):
            if p.ndim == 2:
                plt.plot([ud[j][i] for j in range(len(ud))])
                legends.append('param %d' % i)

        plt.plot([0, len(ud)], [-3, -3], 'k')  # these ratios should be ~1e-3, indicate on plot
        plt.legend(legends)
