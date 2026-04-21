from vspy import vspy, Path

source = Path("test/soundfiles")

# make sure to add the option of tuning hyperparamters for F0 and formants to this 
df = vspy(source, textgrid_dir=Path("test/textgrids"), tier="vowel")

print(df.shape)
print(df.head(20))

