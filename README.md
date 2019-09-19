# Residual-CycleGAN-for-Identity-preserving-Face-Attribute-Editing

## ABSTRACT
Face attributes describe the variation of face components. Its complexity and variety have a great inﬂuence on the tasks of face detection, identifcation and verifcation. Large amount of attribute-annotated data is needed for training an attribute-irrelevant deep-learning based system. We propose a framework for face attribute editing which aims at modifying attributes on face images and generating more self-annotated data. Cycle-consistent adversarial networks (cycleGAN) is incorporated to our framework which can learn the mapping of none-attribute image domain X to target-attribute image domain Y in the absence of pared examples. Instead of learning to generate the whole images directly, we modify the generative network into a residual form which is proposed to learn residual images between the discrepant domains. The proposed structure can ease the burden of generative networks which needs not to learn to reconstruct all the
details from scratch. To preserve the identity information before and after editing, L1 loss, total variation loss (TV loss) and identity-preserving loss (IP-loss) are incorporated into the training of generation jointly. Experiments on the CelebA dataset demonstrate the efciency of the proposed method, and the residual learning corresponded with IP-loss can help to editing the face attribute on high-resolution images with identity information and facial details preserved.

## framework

## experiment

