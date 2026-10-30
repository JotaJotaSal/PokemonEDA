#Initializing SparkSession
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, greatest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

spark = SparkSession.builder \
    .appName("PokemonEDA") \
    .getOrCreate()
#Create DataFrame
file_path = "/home/vboxuser/PokemonEDA/data/pokemon_complete_stats.csv"

pokemon_df = spark.read.csv(
    file_path,
    header=True,
    inferSchema=True
)

print("Total number of Pokemon:", pokemon_df.count())

#Print out the schema structure and the first DataFrame's 5 rows
pokemon_df.printSchema()
pokemon_df.show(5)

#Add region column based on Pokemon ID
pokemon_df = pokemon_df.withColumn(
    "region",
    when(col("id").between(1, 151), "Kanto")
    .when(col("id").between(152, 251), "Johto")
    .when(col("id").between(252, 386), "Hoenn")
    .when(col("id").between(387, 493), "Sinnoh")
    .when(col("id").between(494, 649), "Teselia")
    .when(col("id").between(650, 721), "Kalos")
    .when(col("id").between(722, 809), "Alola")
    .when(col("id").between(810, 905), "Galar")
    .when(col("id").between(906, 1025), "Paldea")
    .otherwise("Other")
)

#Display first 10 rows of Pokemon with their assigned regions
pokemon_df.select("id", "name", "generation", "region").show(10, truncate=False)

# Filter Pokemon by capture rate and generation
filtered_df = pokemon_df.filter(
    (pokemon_df.capture_rate > 45) &
    (pokemon_df.capture_rate < 120) &
    (pokemon_df.generation.isin(
        "generation-i",
        "generation-ii",
        "generation-iii",
        "generation-iv",
	"generation-v"
    ))
)

print("Filtered Pokemon count:", filtered_df.count())
filtered_df.show(5, truncate=False, vertical=False)

# Identify support Pokemon with the Prankster ability
support_df = filtered_df.filter(
    col("abilities").contains("prankster")
)

# Calculate the bulk score
support_df = support_df.withColumn(
    "bulk_score",
    col("hp") + col("defense") + col("special_defense")
)

# Order Pokemon by their bulk score
support_df = support_df.orderBy(
    col("bulk_score").desc()
)

# Display the five best support Pokemon
print("Pokemon with Prankster ability:", support_df.count())

support_df.select(
    "name",
    "abilities",
    "hp",
    "defense",
    "special_defense",
    "bulk_score"
).show(5, truncate=False)

#Filter the pokemons with the type needed
type_df = filtered_df.filter(
    (col("type_1").isin("dark", "ghost", "fire")) |
    (col("type_2").isin("dark", "ghost", "fire"))
)

#Display the total number of selected Pokemon
print("Dark, Ghost, and Fire Pokemon count:", type_df.count())

#Filter Pokemon by selected abilities
attacking_df = type_df.filter(
    col("abilities").contains("flash-fire") |
    col("abilities").contains("moxie") |
    col("abilities").contains("blaze") |
    col("abilities").contains("water-absorb") |
    col("abilities").contains("adaptability") |
    col("abilities").contains("intimidate")
)

print("Pokemon with selected abilities:", attacking_df.count())

#Calculate the highest offensive stat
attacking_df = attacking_df.withColumn(
    "best_attack",
    greatest(col("attack"), col("special_attack"))
)

#Order Pokemon by offensive potential
attacking_df = attacking_df.orderBy(
    col("best_attack").desc()
)

#Display the five strongest offensive candidates
attacking_df.select(
    "name",
    "type_1",
    "type_2",
    "abilities",
    "attack",
    "special_attack",
    "best_attack"
).show(5, truncate=False)

# Visualize the five strongest offensive candidates
plot_df = attacking_df.select("name", "best_attack").limit(5).toPandas()

plt.bar(plot_df["name"], plot_df["best_attack"])
plt.title("Top 5 Pokemon by Offensive Stat")
plt.xlabel("Pokemon")
plt.ylabel("Best Attack")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("/home/vboxuser/PokemonEDA/top_5_attack.png")

print("Chart saved: top_5_attack.png")

# Visualize support Pokemon by bulk score
support_plot_df = support_df.select(
    "name",
    "bulk_score"
).toPandas()

plt.bar(support_plot_df["name"], support_plot_df["bulk_score"])
plt.title("Support Pokemon by Bulk Score")
plt.xlabel("Pokemon")
plt.ylabel("Bulk Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("/home/vboxuser/PokemonEDA/support_pokemon.png")

print("Chart saved: support_pokemon.png")

#Stop the session
spark.stop()
