import kagglehub

path = kagglehub.dataset_download(
    "shreyashautomation/pokemon-stats-dataset",
    output_dir="/home/vboxuser/PokemonEDA/data"
)

print("Path to dataset files:", path)
