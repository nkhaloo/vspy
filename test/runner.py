from vspy import vspy, Path

source = Path("test/wav")

df = vspy(source, textgrid_dir=Path("test/textgrid"), tier="words", output_csv="test/output.csv", f0_source='snack', formant_source='snack')

print(df.shape)
print(df.head(20))

