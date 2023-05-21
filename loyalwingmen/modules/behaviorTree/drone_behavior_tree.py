from enum import Enum
import random
import math
from modules.behaviorTree.behavior_tree_base import (
    BehaviorTree,
    LeafNode,
    SelectorNode,
    SequenceNode,
    ExecutionStatus,
)


class DroneBehaviorTree(BehaviorTree):
    """
    Represents a behavior tree of a roomba cleaning robot.
    """

    def __init__(self):
        super().__init__()
        # Todo: construct the tree here

        """
            Set Root
        """

        root = SelectorNode("root")

        self.root = root

        """
            Seq1
        """

        seq1 = SequenceNode("seq1")
        root.add_child(seq1)

        leaf11 = MoveForwardNode()
        leaf12 = MoveInSpiralNode()

        seq1.add_child(leaf11)
        seq1.add_child(leaf12)

        """
            Seq2
        """

        seq2 = SequenceNode("seq2")
        root.add_child(seq2)

        leaf21 = GoBackNode()
        leaf22 = RotateNode()

        seq2.add_child(leaf21)
        seq2.add_child(leaf22)


class MoveForwardNode(LeafNode):
    def __init__(self):
        super().__init__("MoveForward")

    def enter(self, agent):
        pass

    def execute(self, agent):
        pass


class MoveInSpiralNode(LeafNode):
    def __init__(self):
        super().__init__("MoveInSpiral")
        # Todo: add initialization code

    def enter(self, agent):
        pass

    def execute(self, agent):
        pass


class GoBackNode(LeafNode):
    def __init__(self):
        super().__init__("GoBack")
        # Todo: add initialization code

    def enter(self, agent):
        pass

    def execute(self, agent):
        pass


class RotateNode(LeafNode):
    def __init__(self):
        super().__init__("Rotate")
        # Todo: add initialization code

    def enter(self, agent):
        pass

    def execute(self, agent):
        pass
