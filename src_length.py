import csv

def count_numbers_from_csv(csv_path):
    counts = {i: 0 for i in range(1, 51)}  # Initialize counts for 1 to 50
    
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        lengths = [int(row['src_len']) for row in reader]
    
    # Filter lengths <= 50 for total count
    filtered_lengths = [l for l in lengths if 1 <= l <= 50]
    total = len(filtered_lengths)
    
    # Count occurrences only for numbers 1 to 50
    for length in filtered_lengths:
        counts[length] += 1
    
    # Convert counts to percentages
    percentages = {length: (count / total) * 100 if total > 0 else 0 for length, count in counts.items()}
    
    return percentages

def print_percentages(percentages):
    print("Sentence Length Percentage Distribution (1 to 50):")
    for length in range(1, 51):
        print(f"Length {length}: {percentages[length]:.2f}%")

if __name__ == "__main__":
    csv_file_path = "src_lengths.csv"  # update with your CSV file path
    percentages = count_numbers_from_csv(csv_file_path)

    print_percentages(percentages)
