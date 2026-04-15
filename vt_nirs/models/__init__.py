"""
VT-NIRS: Virtual Twin for Non-Invasive Respiratory Support
===========================================================
Censoring-aware adversarial framework for individualized treatment effect
estimation of NIRS vs IMV in acute respiratory failure.

Base architecture: GANITE (Yoon et al., ICML 2018)
Novel modification: Survival-decomposed adversarial ITE with Transformer encoder

References:
  [Base] Yoon J, Jordon J, van der Schaar M. "GANITE: Estimation of
         Individualized Treatment Effects using Generative Adversarial Nets."
         ICML 2018.
  [Encoder] Vaswani A et al. "Attention Is All You Need." NeurIPS 2017.
  [Survival] Fine JP, Gray RJ. "A Proportional Hazards Model for the
             Subdistribution of a Competing Risk." JASA 1999.
  [Lab pattern] Yadav P et al. "Spatiotemporal GNN with Dynamic Adjacency
                Enhancement for ARDS Prediction." JBI 2026.
"""

from .encoder import TemporalTransformerEncoder, PositionalEncoding
from .encoder_sa import SurvivalAwareTransformerEncoder
from .generator import CounterfactualGenerator
from .discriminator import TreatmentDiscriminator
from .ite_predictor import ITEPredictor
from .vt_nirs import VTNIRSModel
from .baselines import CustomTLearner, CustomCausalForest, run_baselines
