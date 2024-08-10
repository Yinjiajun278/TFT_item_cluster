This project is entirely written by Claude-3-Sonnet. I have no idea what it is doing, my only job is telling it what I want, and copy error messages to it.
I feel like more people should try programming - with no training.

Summary by Claude:
This project is a Python-based tool for analyzing character builds in a game. It processes build data for various characters, identifies core items, clusters non-core items, and provides insights into item usage patterns.

## File Structure

- `analyse_2.py`: Main script containing all the analysis logic
- `data/`: Directory containing character build data files
- `all_legal_items.txt`: File listing all legal items in the game
- `all_normal_items.txt`: File listing all normal items in the game
- `output.txt`: Output file where analysis results are saved

## Usage

1. Ensure all required libraries are installed:
   ```
   pip install numpy scipy scikit-learn
   ```

2. Place your character build data files in the `data/` directory. Each file should be named after the character and contain build information.

3. Update `all_legal_items.txt` and `all_normal_items.txt` with the current game items if necessary.

4. Run the script:
   ```
   python analyse_2.py
   ```

5. The analysis results will be saved in `output.txt`.

## Output

The script generates an `output.txt` file containing:

- Character-specific analysis:
  - Best in Slot (BIS) build
  - Core items
  - Non-core item clusters
  - Items that can be built multiple times
- Overall cluster hierarchy
- Top 5 non-overlapping clusters across all characters

## Customization

You can adjust the following parameters in the script:

- Threshold for core item identification (default: 0.7)
- Threshold for cluster membership (default: 0.75)
- Number of top clusters to display (default: 6)

## Note

This script assumes a specific format for the input files and may need adjustments if the data format changes.

10-Aug-2024
