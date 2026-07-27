from vspy import vspy, Path

source = Path("test/wav")

# VoiceSauce defaults (vs_Initialize.m): F0algorithm = 'F0 (Straight)',
# FMTalgorithm = 'Formants (Snack)'.  The reference output was produced with
# those, so CPP / Energy / HNR / SoE / the harmonics are all driven by strF0.
df = vspy(source, textgrid_dir=Path("test/textgrid"), tier="words", output_csv="test/output.csv", f0_source='straight', formant_source='snack')

print(df.shape)
print(df.head(20))

