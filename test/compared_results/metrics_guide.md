# Evaluation Metrics Guide

## General Metrics (all features)

**Pearson R**
Linear correlation between vspy and VoiceSauce values. Ranges from -1 to 1. A value of 1 means the two outputs move together perfectly, but does not catch systematic offsets (e.g. one system always outputting 10 dB higher will still score r=1).

**CCC (Concordance Correlation Coefficient)**
Like Pearson R but also penalizes systematic bias. A value of 1 means perfect agreement in both direction and scale. If CCC is much lower than Pearson R for a feature, there is a consistent offset between the two systems.

**Mean Error**
Average of (vspy - VoiceSauce) across all valid frames. A positive value means vspy is systematically higher; negative means vspy is systematically lower. Close to zero means no consistent bias.

**MAE (Mean Absolute Error)**
Average of |vspy - VoiceSauce| across all valid frames. Reported in the same units as the feature (Hz for F0/formants, dB for amplitudes). Captures the typical size of disagreement regardless of direction.

**Coverage**
Proportion of merged frames (0 to 1) where both systems produced a non-NaN value. A low coverage means the two systems disagree substantially on which frames are valid (e.g. one marks a frame as unvoiced while the other does not).

---

## F0-Specific Metrics

**VDE (Voicing Decision Error)**
Proportion of frames where one system labels a frame as voiced and the other labels it as unvoiced. A high VDE means the two systems disagree on when the speaker is phonating.

**GPE (Gross Pitch Error)**
Among frames where both systems agree the frame is voiced, the proportion where their F0 estimates differ by more than 20%. Captures octave errors and large tracking failures.

**FPE Mean / FPE Std (Fine Pitch Error)**
Mean and standard deviation of (vspy F0 - VoiceSauce F0) on agreed-voiced frames only, in Hz. FPE mean is the average bias on voiced frames; FPE std captures how much that error varies frame to frame.
