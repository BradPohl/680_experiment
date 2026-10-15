# DSPy Muay Thai Strike Classifier

A DSPy-based system using BootstrapFewShot optimizer to help Google Gemini classify Muay Thai strike combinations in videos.

## Overview

This project implements a video classification system that:
- Uses Google Gemini 2.5 Flash to analyze Muay Thai training videos
- Leverages DSPy's BootstrapFewShot optimizer to automatically select the best few-shot examples
- Classifies sequences of strikes (e.g., ["jab", "cross", "lead_hook", "roundhouse"])
- Evaluates performance on validation sets from both shadowboxing and heavy bag training videos

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

1. Get your Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a `.env` file in the project root:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` and add your API key:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

### 3. Data Structure

The project expects the following data structure:
```
data/
  combos/
    shadow/
      *.mov (video files)
      true_labels.json
    heavy_bag/
      *.mov (video files)
      true_labels.json
```

Each `true_labels.json` file contains a mapping of video IDs to strike sequences:
```json
{
  "1": ["jab", "cross", "lead_hook", "roundhouse"],
  "2": ["jab", "cross", "lowkick"]
}
```

## Usage

### Running the Notebook

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Open `{notebook}.ipynb`

3. Run all cells sequentially. The notebook is organized into sections:
   - **Section 1**: Setup and configuration
   - **Section 2**: Data loading and train/validation split
   - **Section 3**: DSPy signature and classifier module
   - **Section 4**: BootstrapFewShot optimizer
   - **Section 5**: Evaluation on validation set
   - **Section 6**: Results analysis and export

### Train/Validation Split

The notebook automatically splits the data:
- **Training**: 12 videos from shadow + 12 videos from heavy_bag (remaining after validation)
- **Validation**: 7 videos from shadow + 7 videos from heavy_bag

The split is randomized with a fixed seed (42) for reproducibility.

## Strike Types

The classifier recognizes the following Muay Thai strikes:
- `jab`
- `cross`
- `lead_hook`
- `rear_hook`
- `roundhouse`
- `lowkick`
- `lead_uppercut`
- `rear_elbow`
- `lead_elbow`
- `knee`
- `switch_kick`
- `switch_knee`
- `teep`

## Output

After running the notebook, results are saved to the `results/` directory:
- `evaluation_results.json`: Validation metrics, predictions, and per-strike statistics
- `optimized_prompts.json`: Optimized few-shot examples selected by BootstrapFewShot

## Evaluation Metrics

The system calculates:
- **Exact Match Accuracy**: Percentage of videos where the predicted sequence exactly matches the true sequence
- **Average Sequence Accuracy**: Average positional accuracy across all validation examples
- **Per-Strike Accuracy**: Accuracy for each individual strike type

## Notes

- Video processing through Gemini API may take time depending on video length and API rate limits
- The BootstrapFewShot optimizer automatically selects the best examples from the training set
- You can adjust optimizer parameters (`max_bootstrapped_demos`, `max_labeled_demos`) in Section 4
- The notebook includes error handling for API failures and JSON parsing issues

## Requirements

- Python 3.8+
- Google Gemini API key
- DSPy framework
- Jupyter notebook

