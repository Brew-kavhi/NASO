from abc import ABC, abstractmethod


class PruningInterface(ABC):
    @abstractmethod
    def __init__(self, to_prune, *args, **kwargs):
        """
        Initialization methid for the pruning wrapper. This function gets the to_prune object, that si the layer that should be pruned
        """

    @abstractmethod
    def weight_mask_op(self, *args, **kwargs):
        """
        This function actuaslly pruness the weioghts and is thererfor obvisouly the place to implement the logci. This method is important as the pruning callbackkms will try to call this method on the layer to dynamically adjust the pruning.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        Returns:
            nothing
        """

    @abstractmethod
    def sparsity(self) -> float:
        """
        This is necessary to get the sparsity of this layer. There is no specified way to calculate it, it just needs to be returned here.

        Returns:
            float, representing the sparsity of this layer after pruning
        """

    @abstractmethod
    def conditional_mask_update(self):
        """
        This function updates the mask and is called in the callback after each epoch. So this is where the logic to update the mask happens and to decide when to update it.
        """

    @abstractmethod
    def strip_pruning(self):
        """This prodives the logic to remove the pruning layer and all associatedweights"""
